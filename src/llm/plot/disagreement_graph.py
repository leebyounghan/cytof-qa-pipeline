#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cluster sim-disagreement graph plot.

Per (sample, step), draws the meta clusters (rendered at their rep-cell
positions) colored by their LLM label with top-K (x, y)-nearest-neighbor
edges whose **weight (linewidth + alpha)** equals the pair's cosine
similarity on the rep cells' raw protein expression. Edges are light
gray when labels agree, orange when they disagree.

Pass ``--smoothed_path`` to also load a second pred file and render a
2-panel before/after diagnostic in the same axes.

Output layout::

    {output_dir}/{dataset}/{sample}/step_NN/c2s_clusters.png

Usage::

    # Single-snapshot plot (most common)
    python -m src.llm.plot.disagreement_graph \\
        --eval_path   results/gpt-5.4_hvp10/Acute2020/pred.json \\
        --dataset     Acute2020 \\
        --benchmark   benchmark/ --data-dir data/ \\
        --output_dir  results/gpt-5.4_hvp10

    # Before/after diagnostic (smoothed_path comes from postprocess.smooth)
    python -m src.llm.plot.disagreement_graph \\
        --eval_path     results/gpt-5.4_hvp10/Acute2020/pred.json \\
        --smoothed_path results/gpt-5.4_hvp10/Acute2020/pred_smoothed.json \\
        --dataset       Acute2020 \\
        --benchmark     benchmark/ --data-dir data/ \\
        --output_dir    results/gpt-5.4_hvp10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.bench import BenchmarkLoader, parse_perturbations
from src.llm.inference.output_schema import group_by_sample_step
from src.llm.postprocess.smooth import (
    DEFAULT_TOP_K_XY,
    cosine_similarity_matrix,
    top_k_spatial_edges,
)
from src.llm.utils.task_input_adapter import _TaskInputDataset, _step_from_task


def _panel_palette(options: list[str]):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import cm
    cmap = cm.get_cmap('tab20', max(len(options), 1))
    color_lookup = {lab: cmap(i % 20) for i, lab in enumerate(options)}
    return color_lookup, '#cccccc'


def _edge_weight(sim_val: float) -> tuple[float, float]:
    """(alpha, linewidth) for a given cosine similarity.

    Alpha ranges [0.15, 0.95]; linewidth ranges [0.30, 2.50] for
    sim ∈ [0, 1] (arcsinh cytometry stays non-negative). Defensive
    clip for pathological negative sims.
    """
    s = float(sim_val) if sim_val == sim_val else 0.0  # NaN → 0
    s = float(np.clip(s, 0.0, 1.0))
    return 0.15 + 0.80 * s, 0.30 + 2.20 * s


def _draw_panel(
    ax,
    x_parent, y_parent, x_rep, y_rep,
    labels, edges,
    options, color_lookup, none_color,
    x_marker, y_marker,
    max_bg: int = 5000,
) -> tuple[int, int]:
    """Draw one panel: parent backdrop + edges + cluster scatter + legend.

    Returns ``(n_same_edges, n_diff_edges)`` so the caller can build the
    panel title.
    """
    n_bg = len(x_parent)
    if n_bg > max_bg:
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(n_bg, max_bg, replace=False)
    else:
        bg_idx = np.arange(n_bg)

    ax.scatter(
        x_parent[bg_idx], y_parent[bg_idx], s=2, c='lightgray',
        alpha=0.3, rasterized=True, linewidths=0, zorder=1,
    )

    n_same = n_diff = 0
    for i, j, s in edges:
        li, lj = labels[i], labels[j]
        same = (li is not None and lj is not None and li == lj)
        alpha, lw = _edge_weight(s)
        if same:
            n_same += 1
            ax.plot([x_rep[i], x_rep[j]], [y_rep[i], y_rep[j]],
                    color=(0.50, 0.50, 0.50), lw=lw, alpha=alpha, zorder=2)
        else:
            n_diff += 1
            ax.plot([x_rep[i], x_rep[j]], [y_rep[i], y_rep[j]],
                    color=(1.0, 0.45, 0.0), lw=lw, alpha=max(alpha, 0.35),
                    zorder=3)

    labels_arr = np.asarray(
        [l if l is not None else '__none__' for l in labels], dtype=object,
    )
    for lab in list(options) + ['__none__']:
        mask = labels_arr == lab
        if not mask.any():
            continue
        color = (color_lookup.get(lab, none_color)
                 if lab != '__none__' else none_color)
        display = lab if lab != '__none__' else 'None (parse-fail)'
        ax.scatter(
            x_rep[mask], y_rep[mask], s=110, c=[color],
            edgecolors='black', linewidths=0.7, label=display, zorder=5,
        )

    ax.set_xlabel(x_marker)
    ax.set_ylabel(y_marker)
    ax.legend(loc='best', fontsize=8, framealpha=0.85)
    return n_same, n_diff


def plot_cluster_graph(
    x_parent: np.ndarray,
    y_parent: np.ndarray,
    x_rep: np.ndarray,
    y_rep: np.ndarray,
    labels_before: list[str | None],
    sim: np.ndarray,
    options: list[str],
    x_marker: str,
    y_marker: str,
    step_num: int,
    output_path: str,
    labels_after: list[str | None] | None = None,
    top_k_xy: int = DEFAULT_TOP_K_XY,
    max_bg: int = 5000,
) -> dict:
    """Single panel when ``labels_after`` is None, else 2-panel
    before/after. Edges (xy-top-K) are computed once from geometry and
    reused across both panels — only their same/diff classification
    changes after smoothing.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    color_lookup, none_color = _panel_palette(options)
    edges = top_k_spatial_edges(x_rep, y_rep, sim, k=top_k_xy)
    sim_vals = (np.asarray([s for _, _, s in edges], dtype=np.float64)
                if edges else np.array([]))

    if labels_after is None:
        fig, ax = plt.subplots(figsize=(9, 8))
        n_same, n_diff = _draw_panel(
            ax, x_parent, y_parent, x_rep, y_rep,
            labels_before, edges, options, color_lookup, none_color,
            x_marker, y_marker, max_bg=max_bg,
        )
        sim_tag = (f'  |  sim on edges: [{sim_vals.min():.3f}, '
                   f'{sim_vals.max():.3f}]' if edges else '')
        ax.set_title(
            f'Step {step_num}: {x_marker} vs {y_marker}\n'
            f'{len(labels_before)} clusters · top-{top_k_xy} (x,y)-nearest '
            f'edges (same={n_same}, diff={n_diff}){sim_tag} · '
            f'edge weight = cosine sim',
            fontsize=10,
        )
        stats = {
            'n_edges_same':       n_same,
            'n_edges_diff':       n_diff,
            'n_edges_same_after': None,
            'n_edges_diff_after': None,
        }
    else:
        fig, axes = plt.subplots(1, 2, figsize=(17, 8),
                                 sharex=True, sharey=True)
        n_same_b, n_diff_b = _draw_panel(
            axes[0], x_parent, y_parent, x_rep, y_rep,
            labels_before, edges, options, color_lookup, none_color,
            x_marker, y_marker, max_bg=max_bg,
        )
        axes[0].set_title(
            f'Before smoothing  —  same={n_same_b}, diff={n_diff_b}',
            fontsize=10,
        )
        n_same_a, n_diff_a = _draw_panel(
            axes[1], x_parent, y_parent, x_rep, y_rep,
            labels_after, edges, options, color_lookup, none_color,
            x_marker, y_marker, max_bg=max_bg,
        )
        axes[1].set_title(
            f'After  smoothing  —  same={n_same_a}, diff={n_diff_a}',
            fontsize=10,
        )
        sim_tag = (f'sim on edges: [{sim_vals.min():.3f}, '
                   f'{sim_vals.max():.3f}]' if edges else '')
        fig.suptitle(
            f'Step {step_num}: {x_marker} vs {y_marker}  —  '
            f'top-{top_k_xy} (x,y)-nearest edges, weight = cosine sim  |  '
            f'{sim_tag}',
            fontsize=11,
        )
        stats = {
            'n_edges_same':       n_same_b,
            'n_edges_diff':       n_diff_b,
            'n_edges_same_after': n_same_a,
            'n_edges_diff_after': n_diff_a,
        }

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    stats['sim_min'] = float(sim_vals.min()) if edges else None
    stats['sim_max'] = float(sim_vals.max()) if edges else None
    stats['sim_median'] = float(np.median(sim_vals)) if edges else None
    return stats


# ── Per-step driver ─────────────────────────────────────────────────────────

def plot_step(
    ti,
    c2s_data: dict,
    mapping,
    records_for_step: list[dict],
    smoothed_records_for_step: list[dict] | None,
    plot_path: str,
    top_k_xy: int = DEFAULT_TOP_K_XY,
) -> dict | None:
    """Draw the disagreement-graph plot for one (sample, step)."""
    dataset = _TaskInputDataset(ti)
    step = _step_from_task(ti)

    parent_mask = dataset.get_parent_mask(step)
    parent_idx = np.where(parent_mask)[0]
    if len(parent_idx) == 0:
        return None

    x_all, y_all = dataset.get_xy_vals(step)
    x_vals = x_all[parent_idx]
    y_vals = y_all[parent_idx]

    protein_expr, _ = dataset.get_protein_expr()
    parent_protein = (protein_expr[parent_idx] if protein_expr.size
                      else protein_expr)

    rep_parent_local = mapping['rep_parent_local']
    rec_lookup = {str(r['cell_id']): r for r in records_for_step}
    rep_locals: list[int] = []
    cluster_meta: list[str] = []  # cell_id per cluster
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
        return None

    rep_locals_arr = np.asarray(rep_locals, dtype=np.int64)

    labels_before: list[str | None] = [
        rec_lookup[cid]['prediction'] for cid in cluster_meta
    ]

    labels_after: list[str | None] | None = None
    if smoothed_records_for_step is not None:
        smooth_lookup = {str(r['cell_id']): r
                         for r in smoothed_records_for_step}
        labels_after = [
            (smooth_lookup[cid]['prediction'] if cid in smooth_lookup else None)
            for cid in cluster_meta
        ]

    if parent_protein.size:
        X_rep = parent_protein[rep_locals_arr]
        sim = cosine_similarity_matrix(X_rep)
    else:
        sim = np.zeros((len(rep_locals_arr), len(rep_locals_arr)),
                       dtype=np.float32)

    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    return plot_cluster_graph(
        x_parent=x_vals, y_parent=y_vals,
        x_rep=x_vals[rep_locals_arr],
        y_rep=y_vals[rep_locals_arr],
        labels_before=labels_before,
        labels_after=labels_after,
        sim=sim,
        options=c2s_data.get('options', []),
        x_marker=c2s_data['x_marker'],
        y_marker=c2s_data['y_marker'],
        step_num=int(ti.step),
        output_path=plot_path,
        top_k_xy=top_k_xy,
    )


# ── Dataset-level driver ────────────────────────────────────────────────────

def plot_disagreement_dataset(
    eval_path: str,
    dataset_name: str,
    benchmark_dir: str,
    data_dir: str,
    output_dir: str,
    smoothed_path: str | None = None,
    top_k_xy: int = DEFAULT_TOP_K_XY,
    loader: BenchmarkLoader | None = None,
) -> None:
    preds = json.loads(Path(eval_path).read_text())
    by_step = group_by_sample_step(preds)
    by_step_smoothed: dict[tuple[str, int], list[dict]] | None = None
    if smoothed_path:
        smoothed = json.loads(Path(smoothed_path).read_text())
        by_step_smoothed = group_by_sample_step(smoothed)

    pred_meta = preds.get('meta', {}) if isinstance(preds, dict) else {}
    pred_slug = pred_meta.get('perturbation')
    loader = loader if loader is not None else BenchmarkLoader(benchmark_dir, data_dir)
    loader_slug = loader.perturbation_name
    if pred_slug != loader_slug:
        raise SystemExit(
            f'[ERROR] perturbation mismatch between pred.json and loader: '
            f'pred={pred_slug!r}, loader={loader_slug!r}. Pass the same '
            f'--perturbation spec as was used when generating c2s.'
        )
    c2s_filename = (f'c2s__{loader_slug}.json'
                    if loader_slug else 'c2s.json')
    mapping_filename = (f'cell2cluster__{loader_slug}.npz'
                        if loader_slug else 'cell2cluster.npz')

    out_root = Path(output_dir) / dataset_name

    n_written = 0
    for (sample, step_num), records in sorted(by_step.items()):
        step_str = f'step_{step_num:02d}'
        step_dir = Path(benchmark_dir) / dataset_name / sample / step_str
        task_json = step_dir / 'task.json'
        c2s_path = step_dir / c2s_filename
        mapping_path = step_dir / mapping_filename
        missing = [p.name for p in (task_json, c2s_path, mapping_path)
                   if not p.exists()]
        if missing:
            print(f'    [SKIP] {sample}/{step_str}: missing {missing}')
            continue

        ti = loader.load_task(str(task_json))
        c2s_data = json.loads(c2s_path.read_text())
        mapping = np.load(mapping_path)

        smoothed_recs = (by_step_smoothed.get((sample, step_num))
                         if by_step_smoothed is not None else None)

        plot_path = out_root / sample / step_str / 'c2s_clusters.png'
        stats = plot_step(
            ti, c2s_data, mapping, records, smoothed_recs,
            plot_path=str(plot_path),
            top_k_xy=top_k_xy,
        )
        if stats is None:
            continue
        if smoothed_recs is not None:
            print(f'    [ok] {sample}/{step_str}: edges (before→after) '
                  f'same={stats["n_edges_same"]}→{stats["n_edges_same_after"]}, '
                  f'diff={stats["n_edges_diff"]}→{stats["n_edges_diff_after"]}'
                  f'  → {plot_path}')
        else:
            print(f'    [ok] {sample}/{step_str}: edges '
                  f'same={stats["n_edges_same"]} '
                  f'diff={stats["n_edges_diff"]}  → {plot_path}')
        n_written += 1

    print(f'\n  {n_written} plots written under {out_root}/')


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Cluster sim-disagreement graph plot '
                    '(optionally before/after smoothing)',
    )
    parser.add_argument('--eval_path', nargs='+', required=True,
                        help='pred.json from run_openai (repeatable)')
    parser.add_argument('--dataset', nargs='+', required=True,
                        help='Dataset name per --eval_path')
    parser.add_argument('--smoothed_path', nargs='+', default=None,
                        help='Optional pred_smoothed.json from postprocess.'
                             'smooth (one per --eval_path); enables 2-panel '
                             'before/after rendering.')
    parser.add_argument('--benchmark', default='benchmark/')
    parser.add_argument('--data-dir', default='data/', dest='data_dir')
    parser.add_argument('--output_dir', required=True,
                        help='Plot output root — files land under '
                             '{output_dir}/{dataset}/{sample}/step_NN/'
                             'c2s_clusters.png')
    parser.add_argument('--top-k-xy', type=int,
                        default=DEFAULT_TOP_K_XY, dest='top_k_xy',
                        help='Edges per cluster (top-K by Euclidean in '
                             '(x_marker, y_marker)). Default '
                             f'{DEFAULT_TOP_K_XY}.')
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'used at c2s time).',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    from src.hard_depletions import add_hard_depletion_args, resolve_hard_depletion_args
    add_hard_depletion_args(parser)
    args = parser.parse_args()

    if len(args.eval_path) != len(args.dataset):
        parser.error('--eval_path and --dataset counts must match')
    if args.smoothed_path and len(args.smoothed_path) != len(args.eval_path):
        parser.error('--smoothed_path must match --eval_path count')

    resolve_hard_depletion_args(args)

    perturbation_name, perturbation = parse_perturbations(
        args.perturbation or [], name_override=args.perturbation_name,
    )
    loader = BenchmarkLoader(
        args.benchmark, args.data_dir,
        perturbation=perturbation, perturbation_name=perturbation_name,
    )

    smoothed_paths = args.smoothed_path or [None] * len(args.eval_path)
    for ep, ds, sp in zip(args.eval_path, args.dataset, smoothed_paths):
        print(f'\n[{ds}] plotting from {ep}'
              f'{" + " + sp if sp else ""}')
        plot_disagreement_dataset(
            eval_path=ep, dataset_name=ds,
            benchmark_dir=args.benchmark, data_dir=args.data_dir,
            output_dir=args.output_dir,
            smoothed_path=sp,
            top_k_xy=args.top_k_xy,
            loader=loader,
        )


if __name__ == '__main__':
    main()
