#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Propagate per-cluster LLM predictions to all parent cells via
cluster-label lookup.

Input: ``pred.json`` written by ``src.llm.inference.run_openai`` —
hierarchical ``{meta, samples}`` of per-cluster records (one record per
meta cluster emitted by ``src.llm.c2s``).

For every parent cell we read its meta-cluster slot from
``cell2cluster.npz['cluster_labels']`` (parallel to ``parent_indices``,
values 0..K-1) and inherit the LLM label attached to that slot. O(1) per
cell — no 1-NN search, no protein features.

Clusters with ``prediction=None`` (LLM parse failures) yield ``None``
for every parent cell assigned to them; ``src.eval`` treats those as
wrong. Steps with no usable clusters emit a None-filled label list.

Usage::

    python -m src.llm.postprocess.propagate \\
        --eval_path   results/gpt-5.4_hvp10/Acute2020/pred.json \\
        --dataset     Acute2020 \\
        --benchmark   benchmark/ --data-dir data/ \\
        --method      gpt-5.4_hvp10 \\
        --output_dir  results/gpt-5.4_hvp10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.bench import BenchmarkLoader, parse_perturbations
from src.llm.inference.output_schema import group_by_sample_step
from src.llm.utils.task_input_adapter import _TaskInputDataset, _step_from_task


def _coerce_pred(pred):
    """Some 4B-class models return a dict instead of a plain label string
    (e.g. {"name": "Cleanup1"}). Coerce to str or None so downstream
    set/sort operations stay hashable. Mirrors the clustering paradigm's
    handling in src/llm_flowsom/postprocess/propagate.py.
    """
    if pred is None or isinstance(pred, str):
        return pred
    if isinstance(pred, dict):
        for k in ('name', 'label', 'category', 'cell_type', 'value'):
            v = pred.get(k)
            if isinstance(v, str):
                return v
        return None
    return str(pred)


def _propagate_step(
    ti,
    mapping: np.lib.npyio.NpzFile,
    cluster_lookup: dict[str, str | None],
) -> tuple[list[str | None], dict]:
    """Return (``predicted_labels`` length ``n_parent``, info dict).

    Each parent cell inherits the LLM label of its assigned meta cluster
    (from ``cluster_labels[i]`` → ``cell_{slot + 1}`` → LLM prediction).

    ``cluster_lookup`` keys are ``cell_id`` strings (``'cell_1'``…) as
    emitted by ``run_openai``; values are the LLM-predicted category or
    ``None`` for parse failures.
    """
    dataset = _TaskInputDataset(ti)
    step = _step_from_task(ti)

    parent_mask = dataset.get_parent_mask(step)
    parent_idx = np.where(parent_mask)[0]
    n_parent = int(len(parent_idx))
    info: dict = {
        'method':           'cluster_label',
        'n_parent_cells':   n_parent,
        'n_clusters_total': 0,
        'n_clusters_used':  0,
        'n_parse_fail':     0,
    }
    if n_parent == 0:
        return [], info

    cluster_labels = mapping['cluster_labels']  # (n_parent,), int 0..K-1
    if len(cluster_labels) != n_parent:
        raise SystemExit(
            f'[ERROR] cluster_labels length {len(cluster_labels)} != '
            f'n_parent {n_parent} for {ti.dataset}/{ti.sample}/step_{ti.step}'
        )
    n_slots = int(cluster_labels.max()) + 1 if cluster_labels.size else 0

    # slot → LLM label (or None). Missing cell_ids (e.g. c2s_prompt truncation)
    # stay None so the parent cells assigned to them fall through to None.
    slot_labels: list[str | None] = [None] * n_slots
    n_clusters_total = 0
    n_parse_fail = 0
    for slot in range(n_slots):
        cid = f'cell_{slot + 1}'
        if cid not in cluster_lookup:
            continue
        n_clusters_total += 1
        pred = cluster_lookup.get(cid)
        if pred is None:
            n_parse_fail += 1
            continue
        slot_labels[slot] = pred

    n_clusters_used = sum(1 for l in slot_labels if l is not None)
    info['n_clusters_total'] = n_clusters_total
    info['n_clusters_used']  = n_clusters_used
    info['n_parse_fail']     = n_parse_fail

    if n_clusters_used == 0:
        return [None] * n_parent, info

    predicted_labels: list[str | None] = [
        slot_labels[int(c)] for c in cluster_labels
    ]

    info['categories_in_clusters'] = sorted({
        l for l in slot_labels if l is not None
    })
    return predicted_labels, info


def export_dataset(
    eval_path: str,
    dataset_name: str,
    benchmark_dir: str,
    data_dir: str,
    method: str,
    output_dir: str,
    loader: BenchmarkLoader | None = None,
    mapping_filename: str | None = None,
) -> int:
    """Write prediction.json for every (sample, step) present in ``eval_path``.

    A pre-built ``loader`` (with the same perturbation used at c2s time) must
    be passed for Setting 2 runs so parent cells are rebuilt consistently
    with ``cell2cluster__{slug}.npz``.
    """
    preds = json.loads(Path(eval_path).read_text())
    by_step = group_by_sample_step(preds)

    # Cross-check: if the pred.json records a perturbation, the loader must
    # carry the same slug. Reconstructing parent cells from parquet without
    # the perturbation would invalidate every cluster_labels index.
    pred_meta = preds.get('meta', {}) if isinstance(preds, dict) else {}
    pred_slug = pred_meta.get('perturbation')
    loader = loader if loader is not None else BenchmarkLoader(benchmark_dir, data_dir)
    loader_slug = loader.perturbation_name
    if pred_slug != loader_slug:
        raise SystemExit(
            f'[ERROR] perturbation mismatch between pred.json and loader: '
            f'pred={pred_slug!r}, loader={loader_slug!r}. Pass the same '
            f'--perturbation spec to src.llm.postprocess.propagate as was '
            f'used when generating c2s.'
        )

    if mapping_filename is None:
        mapping_filename = (f'cell2cluster__{loader_slug}.npz'
                            if loader_slug else 'cell2cluster.npz')

    out_root = Path(output_dir)
    n_written = 0
    n_skipped = 0

    for (sample, step_num), records in sorted(by_step.items()):
        step_str = f'step_{step_num:02d}'
        step_dir = Path(benchmark_dir) / dataset_name / sample / step_str
        task_json = step_dir / 'task.json'
        mapping_path = step_dir / mapping_filename

        missing = [p.name for p in (task_json, mapping_path)
                   if not p.exists()]
        if missing:
            print(f'    [SKIP] {sample}/{step_str}: missing {missing}')
            n_skipped += 1
            continue

        ti = loader.load_task(str(task_json))
        mapping = np.load(mapping_path)

        cluster_lookup = {
            str(r['cell_id']): _coerce_pred(r['prediction']) for r in records
        }

        predicted_labels, info = _propagate_step(ti, mapping, cluster_lookup)

        ti.save_prediction(
            predicted_labels,
            output_dir=out_root,
            method=method,
            info=info,
        )

        print(f'    [ok] {sample}/{step_str}: {info["n_parent_cells"]} cells, '
              f'{info["n_clusters_used"]}/{info["n_clusters_total"]} clusters '
              f'(parse_fail={info["n_parse_fail"]})')
        n_written += 1

    print(f'  → {n_written} prediction.json written ({n_skipped} skipped)')
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Propagate LLM cluster predictions to all parent cells '
                    'via cell2cluster.npz label lookup; emit src.eval-'
                    'compatible prediction.json per step.',
    )
    parser.add_argument('--eval_path', nargs='+', required=True,
                        help='pred.json from run_openai (repeatable)')
    parser.add_argument('--dataset', nargs='+', required=True,
                        help='Dataset name per --eval_path')
    parser.add_argument('--benchmark', type=Path, default=Path('benchmark/'))
    parser.add_argument('--data-dir', type=Path, default=Path('data/'),
                        dest='data_dir')
    parser.add_argument('--method', required=True,
                        help='Method tag recorded in prediction.json + used '
                             'as the leaf directory name.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'passed to src.llm.c2s). Reapplied when rebuilding parent '
             'cells so cluster_labels align with the clusters.',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    parser.add_argument('--mapping-filename', dest='mapping_filename',
                        default=None,
                        help='Override the cluster-mapping npz filename '
                             '(default: cell2cluster.npz). Use cell2grid.npz '
                             'for grid-baseline runs.')
    from src.hard_depletions import add_hard_depletion_args, resolve_hard_depletion_args
    add_hard_depletion_args(parser)
    args = parser.parse_args()

    if len(args.eval_path) != len(args.dataset):
        parser.error('--eval_path and --dataset must have matching counts')

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

    total = 0
    for eval_path, dataset_name in zip(args.eval_path, args.dataset):
        print(f'\n[{dataset_name}] propagating from {eval_path}')
        total += export_dataset(
            eval_path=eval_path,
            dataset_name=dataset_name,
            benchmark_dir=args.benchmark,
            data_dir=args.data_dir,
            method=args.method,
            output_dir=args.output_dir,
            loader=loader,
            mapping_filename=args.mapping_filename,
        )

    print(f'\nDone. Total {total} prediction.json → {args.output_dir}')


if __name__ == '__main__':
    main()
