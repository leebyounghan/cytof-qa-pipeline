#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sim-disagreement smoothing for the per-cluster LLM predictions.

Greedy local update on the sim-weighted disagreement graph over meta
clusters: each iteration flips the cluster with the highest
diff/(diff+same) ratio (≥ threshold) to the plurality label among its
different-label neighbors, weighted by cosine similarity on the raw
protein expression of each cluster's representative cell. Loop until
every cluster is below threshold or ``max_iterations`` is hit.

Edge **selection** is xy-spatial (top-K Euclidean neighbors between
rep cells in the gating plane); edge **weight** is cosine sim on raw
protein. A thick orange edge → two spatially close, biologically
similar clusters that the LLM labeled differently → prime candidate
for a flip.

This module produces ``pred_smoothed.json`` and nothing else. For the
before/after visual diagnostic, run ``src.llm.plot.disagreement_graph``
separately on the two pred files.

Usage::

    python -m src.llm.postprocess.smooth \\
        --eval_path   results/gpt-5.4_hvp10/Acute2020/pred.json \\
        --dataset     Acute2020 \\
        --benchmark   benchmark/ --data-dir data/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.bench import BenchmarkLoader, parse_perturbations
from src.llm.inference.output_schema import group_by_sample_step
from src.llm.utils.task_input_adapter import _TaskInputDataset, _step_from_task

DEFAULT_TOP_K_XY = 3
DEFAULT_STOP_THRESHOLD = 0.5
DEFAULT_MAX_ITERATIONS = 500


# ── Cosine-sim / edge utilities ─────────────────────────────────────────────

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity on rows. Returns (n, n) with diag≈1."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Xn = X / norms
    return Xn @ Xn.T


def top_k_spatial_edges(
    x: np.ndarray,
    y: np.ndarray,
    sim: np.ndarray,
    k: int,
) -> list[tuple[int, int, float]]:
    """Deduplicated ``(i, j, sim_ij)`` edges for top-``k`` (x, y)-nearest
    neighbor pairs per node.

    *Selection* = Euclidean distance in the 2D plane (the task's native
    space). *Weight* = cosine similarity, used downstream for
    alpha/linewidth scaling and the smoothing step.
    """
    n = len(x)
    if n < 2 or k <= 0:
        return []
    pts = np.column_stack([x, y]).astype(np.float64)
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    k_eff = min(k, n - 1)
    nn_idx = np.argpartition(d2, kth=k_eff - 1, axis=1)[:, :k_eff]

    edge_set: set[tuple[int, int]] = set()
    for i in range(n):
        for j in nn_idx[i]:
            pair = (i, int(j)) if i < int(j) else (int(j), i)
            edge_set.add(pair)
    return [(i, j, float(sim[i, j])) for i, j in sorted(edge_set)]


# ── Smoothing core ──────────────────────────────────────────────────────────

def smooth_labels_by_disagreement(
    labels: list[str | None],
    edges: list[tuple[int, int, float]],
    stop_threshold: float = DEFAULT_STOP_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[list[str | None], dict]:
    """Greedy local smoothing on the sim-weighted disagreement graph.

    See module docstring for the exact update rule. Per-iteration cost is
    O(|E|) for the ratio scan + O(deg) for the orange vote, so total cost
    is O(iterations · |E|) — trivial for a few dozen clusters.

    Returns ``(adjusted_labels, info)`` where ``info`` tracks
    ``n_flipped``, ``iterations``, ``max_iter_hit``, and the flip
    history.
    """
    n = len(labels)
    adjusted: list[str | None] = list(labels)

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i, j, s in edges:
        adj[i].append((j, s))
        adj[j].append((i, s))

    flips: list[dict] = []
    it = 0
    while it < max_iterations:
        # Scan all nodes: find one with max orange_ratio ≥ stop_threshold.
        best_i = -1
        best_ratio = -1.0
        for i in range(n):
            cur = adjusted[i]
            if cur is None:
                continue
            diff_sum = same_sum = 0.0
            for j, s in adj[i]:
                pj = adjusted[j]
                if pj is None:
                    continue
                if pj == cur:
                    same_sum += s
                else:
                    diff_sum += s
            total = diff_sum + same_sum
            if total <= 0:
                continue
            ratio = diff_sum / total
            if ratio >= stop_threshold and ratio > best_ratio:
                best_ratio = ratio
                best_i = i

        if best_i < 0:
            break  # every node below threshold → converged

        # Plurality over ORANGE neighbors only (sim-weighted)
        cur = adjusted[best_i]
        vote: dict[str, float] = defaultdict(float)
        for j, s in adj[best_i]:
            pj = adjusted[j]
            if pj is None or pj == cur:
                continue
            vote[pj] += s
        if not vote:
            # Degenerate: ratio says ≥ 0.5 but no valid orange labels —
            # defensive break to avoid infinite loop.
            break
        new_label = max(vote.items(), key=lambda kv: kv[1])[0]

        flips.append({
            'iter':         int(it),
            'cluster':      int(best_i),
            'before':       cur,
            'after':        new_label,
            'ratio_before': float(best_ratio),
        })
        adjusted[best_i] = new_label
        it += 1

    return adjusted, {
        'n_flipped':    len(flips),
        'iterations':   it,
        'max_iter_hit': it >= max_iterations,
        'history':      flips,
    }


# ── Per-step driver ─────────────────────────────────────────────────────────

def smooth_step(
    ti,
    mapping,
    records_for_step: list[dict],
    top_k_xy: int = DEFAULT_TOP_K_XY,
    stop_threshold: float = DEFAULT_STOP_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> tuple[list[dict], dict]:
    """Run smoothing for one (sample, step). No plotting.

    Returns ``(adjusted_records, info)``. Each adjusted record carries the
    new ``prediction`` plus ``prediction_original`` and ``flipped`` fields.
    """
    dataset = _TaskInputDataset(ti)
    step = _step_from_task(ti)

    empty_info = {
        'sample':       ti.sample,
        'step':         int(ti.step),
        'n_clusters':   0,
        'n_flipped':    0,
        'iterations':   0,
        'max_iter_hit': False,
    }

    parent_mask = dataset.get_parent_mask(step)
    parent_idx = np.where(parent_mask)[0]
    if len(parent_idx) == 0:
        return [], empty_info

    x_all, y_all = dataset.get_xy_vals(step)
    x_vals = x_all[parent_idx]
    y_vals = y_all[parent_idx]

    protein_expr, _ = dataset.get_protein_expr()
    parent_protein = (protein_expr[parent_idx] if protein_expr.size
                      else protein_expr)

    rep_parent_local = mapping['rep_parent_local']
    rec_lookup = {str(r['cell_id']): r for r in records_for_step}
    # Only include clusters that actually received a prediction — if
    # c2s_prompt was run with --n-cells N, the later clusters were never
    # shown to the LLM and should be dropped from the smoothing graph.
    rep_locals: list[int] = []
    cluster_meta: list[str] = []
    for slot, local in enumerate(rep_parent_local):
        local = int(local)
        if local < 0:
            continue
        cid = f'cell_{slot + 1}'
        if cid not in rec_lookup:
            continue
        rep_locals.append(local)
        cluster_meta.append(cid)

    if not rep_locals:
        return [], empty_info

    rep_locals_arr = np.asarray(rep_locals, dtype=np.int64)

    labels_before: list[str | None] = [
        rec_lookup[cid]['prediction'] for cid in cluster_meta
    ]

    if parent_protein.size:
        X_rep = parent_protein[rep_locals_arr]
        sim = cosine_similarity_matrix(X_rep)
    else:
        sim = np.zeros((len(rep_locals_arr), len(rep_locals_arr)),
                       dtype=np.float32)

    edges = top_k_spatial_edges(
        x_vals[rep_locals_arr], y_vals[rep_locals_arr],
        sim, k=top_k_xy,
    )
    labels_after, smooth_info = smooth_labels_by_disagreement(
        labels_before, edges,
        stop_threshold=stop_threshold,
        max_iterations=max_iterations,
    )

    adjusted_records: list[dict] = []
    for idx, cid in enumerate(cluster_meta):
        orig = rec_lookup.get(cid)
        if orig is None:
            continue
        new = dict(orig)
        new['prediction_original'] = orig['prediction']
        new['prediction'] = labels_after[idx]
        new['flipped'] = bool(
            labels_after[idx] != orig['prediction']
            and orig['prediction'] is not None
        )
        adjusted_records.append(new)

    info = {
        'sample':       ti.sample,
        'step':         int(ti.step),
        'n_clusters':   len(rep_locals),
        'n_flipped':    smooth_info['n_flipped'],
        'iterations':   smooth_info['iterations'],
        'max_iter_hit': smooth_info['max_iter_hit'],
    }
    return adjusted_records, info


# ── Dataset-level driver ────────────────────────────────────────────────────

def smooth_dataset(
    eval_path: str,
    dataset_name: str,
    benchmark_dir: str,
    data_dir: str,
    top_k_xy: int = DEFAULT_TOP_K_XY,
    stop_threshold: float = DEFAULT_STOP_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    loader: BenchmarkLoader | None = None,
) -> Path:
    """Walk the eval JSON, smooth each step, write ``pred_smoothed.json``
    next to the input ``eval_path``. Returns the smoothed-pred path.

    A pre-built ``loader`` (with the same perturbation used at c2s time)
    must be passed for Setting 2 runs so the parent cells fed to the
    similarity graph align with ``cell2cluster__{slug}.npz``.
    """
    preds = json.loads(Path(eval_path).read_text())
    by_step = group_by_sample_step(preds)

    pred_meta = preds.get('meta', {}) if isinstance(preds, dict) else {}
    pred_slug = pred_meta.get('perturbation')
    loader = loader if loader is not None else BenchmarkLoader(benchmark_dir, data_dir)
    loader_slug = loader.perturbation_name
    if pred_slug != loader_slug:
        raise SystemExit(
            f'[ERROR] perturbation mismatch between pred.json and loader: '
            f'pred={pred_slug!r}, loader={loader_slug!r}. Pass the same '
            f'--perturbation spec to src.llm.postprocess.smooth as was '
            f'used when generating c2s.'
        )

    mapping_filename = (f'cell2cluster__{loader_slug}.npz'
                        if loader_slug else 'cell2cluster.npz')

    cell_ptr: dict[tuple[str, int, str], dict] = {}
    for sn, sample_entry in preds.get('samples', {}).items():
        for step_str, step_entry in sample_entry.get('steps', {}).items():
            for cell in step_entry.get('cells', []):
                cell_ptr[(sn, int(step_str), cell['cell_id'])] = cell

    all_adjusted: list[dict] = []
    total_flips = total_clusters = 0
    for (sample, step_num), records in sorted(by_step.items()):
        step_str = f'step_{step_num:02d}'
        step_dir = Path(benchmark_dir) / dataset_name / sample / step_str
        task_json = step_dir / 'task.json'
        mapping_path = step_dir / mapping_filename
        missing = [p.name for p in (task_json, mapping_path) if not p.exists()]
        if missing:
            print(f'    [SKIP] {sample}/{step_str}: missing {missing}')
            continue

        ti = loader.load_task(str(task_json))
        mapping = np.load(mapping_path)

        adj_records, info = smooth_step(
            ti, mapping, records,
            top_k_xy=top_k_xy,
            stop_threshold=stop_threshold,
            max_iterations=max_iterations,
        )
        all_adjusted.extend(adj_records)

        for r in adj_records:
            key = (sample, step_num, r['cell_id'])
            ptr = cell_ptr.get(key)
            if ptr is None:
                continue
            ptr['prediction_original'] = r.get('prediction_original')
            ptr['prediction']          = r.get('prediction')
            ptr['flipped']             = r.get('flipped', False)

        total_flips += info['n_flipped']
        total_clusters += info['n_clusters']
        print(f'    [ok] {sample}/{step_str}: '
              f'{info["n_clusters"]} clusters · flips={info["n_flipped"]}'
              f'{" [max_iter]" if info["max_iter_hit"] else ""}')

    pct = 100.0 * total_flips / max(total_clusters, 1)
    print(f'  Total flips: {total_flips} / {total_clusters} clusters '
          f'({pct:.1f}%)')

    smoothed_path = Path(eval_path).with_name(
        Path(eval_path).stem + '_smoothed.json'
    )
    smoothed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(smoothed_path, 'w') as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f'  Smoothed pred → {smoothed_path}')
    print(f'  Compute cluster metrics with: '
          f'python -m src.llm.postprocess.anchor_metric --eval_path {smoothed_path}')
    return smoothed_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Sim-disagreement smoothing → pred_smoothed.json',
    )
    parser.add_argument('--eval_path', nargs='+', required=True,
                        help='pred.json from run_openai (repeatable)')
    parser.add_argument('--dataset', nargs='+', required=True,
                        help='Dataset name per --eval_path')
    parser.add_argument('--benchmark', default='benchmark/')
    parser.add_argument('--data-dir', default='data/', dest='data_dir')
    parser.add_argument('--top-k-xy', type=int,
                        default=DEFAULT_TOP_K_XY, dest='top_k_xy',
                        help='Edges per cluster (top-K by Euclidean in '
                             '(x_marker, y_marker)). Default '
                             f'{DEFAULT_TOP_K_XY}.')
    parser.add_argument('--stop-threshold', type=float,
                        default=DEFAULT_STOP_THRESHOLD,
                        dest='stop_threshold',
                        help='Smoothing stops when every cluster has '
                             'diff_sim / (diff_sim + same_sim) < this. '
                             f'Default {DEFAULT_STOP_THRESHOLD}.')
    parser.add_argument('--max-iterations', type=int,
                        default=DEFAULT_MAX_ITERATIONS,
                        dest='max_iterations',
                        help='Safety cap on smoothing iterations '
                             f'(default {DEFAULT_MAX_ITERATIONS}).')
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'used at c2s time so parent cells + cluster indices align).',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    from src.hard_depletions import add_hard_depletion_args, resolve_hard_depletion_args
    add_hard_depletion_args(parser)
    args = parser.parse_args()

    if len(args.eval_path) != len(args.dataset):
        parser.error('--eval_path and --dataset counts must match')

    resolve_hard_depletion_args(args)

    perturbation_name, perturbation = parse_perturbations(
        args.perturbation or [], name_override=args.perturbation_name,
    )
    loader = BenchmarkLoader(
        args.benchmark, args.data_dir,
        perturbation=perturbation, perturbation_name=perturbation_name,
    )
    if perturbation is not None:
        print(f'Perturbation: {perturbation_name} ({perturbation.describe()})')

    for ep, ds in zip(args.eval_path, args.dataset):
        print(f'\n[{ds}] smoothing from {ep}')
        smooth_dataset(
            eval_path=ep, dataset_name=ds,
            benchmark_dir=args.benchmark, data_dir=args.data_dir,
            top_k_xy=args.top_k_xy,
            stop_threshold=args.stop_threshold,
            max_iterations=args.max_iterations,
            loader=loader,
        )


if __name__ == '__main__':
    main()
