#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cell-count-weighted accuracy for anchor predictions.

Each anchor (= the representative cell of one cluster) actually stands in
for the ``n_cells`` parent cells belonging to that cluster, so a wrong
anchor on a large cluster mislabels many more cells at the cell-level
propagate stage. The plain accuracy in ``compute_eval_metrics`` weights
every anchor equally and therefore diverges from cell-level accuracy.
This module computes a weighted_accuracy where each anchor is weighted by
its cluster size.

Formula: ``weighted_accuracy = sum_i (w_i * 1[pred_i == gt_i]) / sum_i w_i``
where ``w_i`` = the cluster size of anchor i (``n_cells`` in ``c2s.json``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _norm(s):
    return s.strip().lower() if isinstance(s, str) else s


def load_cluster_sizes(
    cache_dir: Path,
    cohort: str,
    samples: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Walk ``cache_dir/{cohort}/{sample}/step_*/c2s.json`` and return
    ``{sample: {step_str: {cell_id_str: n_cells}}}``.

    ``cell_id_str`` matches the ``cell_id`` field in pred.json (``cell_<cid>``).
    Missing entries default to weight 1 in :func:`compute_weighted_metrics`.
    """
    out: dict[str, dict[str, dict[str, int]]] = {}
    cohort_dir = cache_dir / cohort
    if not cohort_dir.is_dir():
        return out
    for sample in samples:
        sample_dir = cohort_dir / sample
        if not sample_dir.is_dir():
            continue
        sample_map: dict[str, dict[str, int]] = {}
        for step_dir in sorted(sample_dir.iterdir()):
            if not step_dir.is_dir() or not step_dir.name.startswith('step_'):
                continue
            c2s_path = step_dir / 'c2s.json'
            if not c2s_path.exists():
                continue
            data = json.loads(c2s_path.read_text())
            step_str = str(int(data['step']))
            cells = data.get('cells') or {}
            sample_map[step_str] = {
                f'cell_{cid}': int(c.get('n_cells', 1) or 1)
                for cid, c in cells.items()
            }
        out[sample] = sample_map
    return out


def compute_weighted_metrics(
    eval_data: dict,
    weights_map: dict[str, dict[str, dict[str, int]]],
) -> dict:
    """Compute weighted accuracy + total_weight at overall / by_sample /
    by_step / per_step / matrix levels. Read-only over ``eval_data``.

    Returns::

        {
          'overall':   {weighted_accuracy, total_weight, weighted_correct},
          'by_sample': {<name>: {weighted_accuracy, total_weight}},
          'by_step':   {<step>: {weighted_accuracy, total_weight}},
          'per_step':  {<sample>: {<step>: {weighted_accuracy, total_weight}}},
          'matrix':    {samples[], steps[], weighted_accuracy[][]}
        }
    """
    by_sample: dict[str, dict] = {}
    by_step_pool: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    per_step: dict[str, dict[str, dict]] = {}
    overall_correct = 0.0
    overall_weight = 0

    samples = eval_data.get('samples', {}) or {}
    for sample_name, sample_entry in samples.items():
        s_correct, s_weight = 0.0, 0
        per_step[sample_name] = {}
        sample_w = weights_map.get(sample_name, {})
        for step_str, step_entry in sample_entry.get('steps', {}).items():
            step_w = sample_w.get(step_str, {})
            step_correct, step_weight = 0.0, 0
            for c in step_entry.get('cells', []):
                w = step_w.get(c.get('cell_id'), 1)
                gt = _norm(c.get('gt'))
                pd = _norm(c.get('prediction'))
                if gt is not None and pd is not None and gt == pd:
                    step_correct += w
                step_weight += w
            per_step[sample_name][step_str] = {
                'weighted_accuracy': (
                    round(step_correct / step_weight, 4) if step_weight else 0.0
                ),
                'total_weight': step_weight,
            }
            s_correct += step_correct
            s_weight += step_weight
            by_step_pool[step_str][0] += step_correct
            by_step_pool[step_str][1] += step_weight
        by_sample[sample_name] = {
            'weighted_accuracy': (
                round(s_correct / s_weight, 4) if s_weight else 0.0
            ),
            'total_weight': s_weight,
        }
        overall_correct += s_correct
        overall_weight += s_weight

    by_step = {
        step_str: {
            'weighted_accuracy': round(c / w, 4) if w else 0.0,
            'total_weight': w,
        }
        for step_str, (c, w) in sorted(
            by_step_pool.items(), key=lambda kv: int(kv[0]),
        )
    }

    sample_list = sorted(samples.keys())
    step_list = sorted(
        {int(s) for sample in samples.values()
         for s in sample.get('steps', {}).keys()},
    )
    mat = [[None] * len(step_list) for _ in sample_list]
    for i, sn in enumerate(sample_list):
        for j, stp in enumerate(step_list):
            se = per_step.get(sn, {}).get(str(stp))
            if se is not None:
                mat[i][j] = se.get('weighted_accuracy')

    return {
        'overall': {
            'weighted_accuracy': (
                round(overall_correct / overall_weight, 4)
                if overall_weight else 0.0
            ),
            'total_weight': overall_weight,
            'weighted_correct': round(overall_correct, 4),
        },
        'by_sample': by_sample,
        'by_step':   by_step,
        'per_step':  per_step,
        'matrix': {
            'samples':           sample_list,
            'steps':             step_list,
            'weighted_accuracy': mat,
        },
    }


def merge_into_metrics(metrics: dict, weighted: dict) -> dict:
    """In-place augment ``metrics`` (from compute_eval_metrics) with the
    weighted_* fields. Adds ``weighted_accuracy`` next to ``accuracy`` /
    ``f1_macro`` at every level the plain metric exists, plus a
    ``weighted_accuracy`` matrix alongside the existing matrices.
    """
    ov_w = weighted.get('overall', {})
    metrics.setdefault('overall', {})['weighted_accuracy'] = (
        ov_w.get('weighted_accuracy')
    )
    metrics['overall']['total_weight'] = ov_w.get('total_weight')

    for name, entry in (weighted.get('by_sample') or {}).items():
        if name in metrics.get('by_sample', {}):
            metrics['by_sample'][name]['weighted_accuracy'] = (
                entry.get('weighted_accuracy')
            )
            metrics['by_sample'][name]['total_weight'] = (
                entry.get('total_weight')
            )

    for step_str, entry in (weighted.get('by_step') or {}).items():
        if step_str in metrics.get('by_step', {}):
            metrics['by_step'][step_str]['weighted_accuracy'] = (
                entry.get('weighted_accuracy')
            )
            metrics['by_step'][step_str]['total_weight'] = (
                entry.get('total_weight')
            )

    for sn, steps in (weighted.get('per_step') or {}).items():
        target = metrics.get('per_step', {}).get(sn, {})
        for step_str, entry in steps.items():
            if step_str in target:
                target[step_str]['weighted_accuracy'] = (
                    entry.get('weighted_accuracy')
                )
                target[step_str]['total_weight'] = entry.get('total_weight')

    mat_w = weighted.get('matrix', {}).get('weighted_accuracy')
    if mat_w is not None and 'matrix' in metrics:
        metrics['matrix']['weighted_accuracy'] = mat_w
    return metrics
