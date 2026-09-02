#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anchor-level metric computation over the eval JSON.

Scope: how well did the LLM label the 100 anchors per step, *before*
propagation. The cell-level benchmark metric (after propagation, over
all parent cells) lives in ``src.eval`` and is the official benchmark
number; this module is a diagnostic complement.

- ``compute_metrics`` — case-insensitive accuracy (+ parse-fail count).
- ``compute_eval_metrics`` — walk a hierarchical pred.json, return the
  full metric structure (overall + by_sample + by_step + per_step +
  matrix) as a fresh dict, **without mutating** the input.
- ``print_eval_summary`` — terminal-friendly summary of the metric block.

CLI: ``python -m src.llm.postprocess.anchor_metric --eval_path pred.json``
reads the pred file, computes metrics, writes ``pred_anchor_metrics.json``
next to it (or the path given via ``--output``), and prints the summary.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

_NOISE = '__noise__'


def _norm(s):
    return s.strip().lower() if isinstance(s, str) else s


def _fmt_pct(v) -> str:
    if v is None:
        return 'n/a'
    return f'{v * 100:.2f}%'


def compute_metrics(gts: list, preds: list) -> dict:
    """Case-insensitive accuracy + parse-fail count.

    None predictions count as wrong (and are tallied in ``n_parse_fail``).
    """
    n = len(gts)
    n_parse_fail = sum(1 for p in preds if p is None)
    gt_n = [_norm(g) for g in gts]
    pd_n = [_norm(p) if p is not None else _NOISE for p in preds]
    correct = sum(1 for g, p in zip(gt_n, pd_n) if g is not None and g == p)
    return {
        'n_cells':      n,
        'n_parse_fail': n_parse_fail,
        'accuracy':     round(correct / n, 4) if n else 0.0,
    }


def compute_eval_metrics(eval_data: dict) -> dict:
    """Walk a hierarchical pred.json (read-only) and return a fresh metrics
    dict. Schema::

        {
          "meta":       {model, ablation, dataset_path, n_samples, n_steps},
          "n_cells":    int,
          "n_parse_fail": int,
          "overall":    {accuracy},
          "by_sample":  {<name>: {n_cells, n_parse_fail, accuracy}},
          "by_step":    {<step>: {n_cells, n_parse_fail, accuracy}},
          "per_step":   {<sample>: {<step>: {n_cells, n_parse_fail, accuracy}}},
          "matrix":     {samples[], steps[], accuracy[][]}
        }

    The input ``eval_data`` is not mutated.
    """
    samples = eval_data.get('samples', {})

    by_step_pool_gt: dict[str, list] = defaultdict(list)
    by_step_pool_pd: dict[str, list] = defaultdict(list)
    all_gt: list = []
    all_pd: list = []
    by_sample: dict = {}
    per_step: dict = {}

    for sample_name, sample_entry in samples.items():
        s_gt, s_pd = [], []
        per_step[sample_name] = {}
        for step_str, step_entry in sample_entry.get('steps', {}).items():
            st_gt = [c['gt'] for c in step_entry.get('cells', [])]
            st_pd = [c['prediction'] for c in step_entry.get('cells', [])]
            step_m = compute_metrics(st_gt, st_pd)
            per_step[sample_name][step_str] = step_m

            by_step_pool_gt[step_str].extend(st_gt)
            by_step_pool_pd[step_str].extend(st_pd)
            s_gt.extend(st_gt)
            s_pd.extend(st_pd)
        by_sample[sample_name] = compute_metrics(s_gt, s_pd)
        all_gt.extend(s_gt)
        all_pd.extend(s_pd)

    overall = compute_metrics(all_gt, all_pd)
    by_step = {}
    for step_str in sorted(by_step_pool_gt.keys(), key=int):
        by_step[step_str] = compute_metrics(
            by_step_pool_gt[step_str], by_step_pool_pd[step_str],
        )

    sample_list = sorted(samples.keys())
    step_list = sorted(
        {int(s) for sample in samples.values()
         for s in sample.get('steps', {}).keys()},
    )
    mat_acc = [[None] * len(step_list) for _ in sample_list]
    for i, sn in enumerate(sample_list):
        for j, stp in enumerate(step_list):
            se = per_step.get(sn, {}).get(str(stp))
            if se is None:
                continue
            mat_acc[i][j] = se.get('accuracy')

    pred_meta = eval_data.get('meta', {}) or {}
    return {
        'meta': {
            'model':        pred_meta.get('model'),
            'ablation':     pred_meta.get('ablation'),
            'dataset_path': pred_meta.get('dataset_path'),
            'n_samples':    len(sample_list),
            'n_steps':      len(step_list),
        },
        'n_cells':       overall['n_cells'],
        'n_parse_fail':  overall['n_parse_fail'],
        'overall': {
            'accuracy': overall['accuracy'],
        },
        'by_sample': by_sample,
        'by_step':   by_step,
        'per_step':  per_step,
        'matrix': {
            'samples':  sample_list,
            'steps':    step_list,
            'accuracy': mat_acc,
        },
    }


def print_eval_summary(metrics: dict, saved_to: str | None = None) -> None:
    meta = metrics.get('meta', {})
    ov = metrics.get('overall', {})
    print(f'\n{"=" * 60}')
    print(f'Model         : {meta.get("model")}')
    print(f'Ablation      : {meta.get("ablation")}')
    print(f'Samples       : {meta.get("n_samples")}')
    print(f'Steps (union) : {meta.get("n_steps")}')
    print(f'Cells (total) : {metrics.get("n_cells", 0):,}')
    print(f'Parse failures: {metrics.get("n_parse_fail", 0):,}')
    print(f'Accuracy      : {_fmt_pct(ov.get("accuracy"))}')
    if saved_to:
        print(f'Saved to      : {saved_to}')
    print(f'{"=" * 60}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute anchor-level metrics from a pred.json. '
                    'Writes pred_anchor_metrics.json (default: alongside pred.json).',
    )
    parser.add_argument('--eval_path', required=True,
                        help='pred.json (or pred_smoothed.json) to score.')
    parser.add_argument('--output', default=None,
                        help='Output anchor-metrics JSON path (default: '
                             '<eval_path stem>_anchor_metrics.json next to input).')
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    eval_data = json.loads(eval_path.read_text())
    metrics = compute_eval_metrics(eval_data)

    out_path = (Path(args.output) if args.output
                else eval_path.with_name(eval_path.stem + '_anchor_metrics.json'))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print_eval_summary(metrics, saved_to=str(out_path))


if __name__ == '__main__':
    main()
