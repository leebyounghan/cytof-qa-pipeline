#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Propagate per-category rectangular gates to all parent cells.

Input: ``pred.json`` written by ``src.llm_gate.inference.run_openai`` —
hierarchical ``{meta, samples}`` of per-step gates (one box per named
category per gating step).

For every parent cell at (x, y) we find the named categories whose
rectangle contains it. Tie-break:

- ``smallest`` (default) — the smallest-area gate wins (most specific).
- ``priority``            — first listed in the step's ``options`` order
                            (excluding ``Unassigned``) wins.

Cells inside no gate become ``Unassigned``. Categories the LLM did not
emit a valid gate for (parse failures, missing entries, degenerate
boxes) simply contribute no membership — those cells fall through to
other matching gates or to ``Unassigned``. Categories explicitly marked
absent (``{"absent": true, ...}``) behave the same at runtime but are
counted separately in the per-step info (``n_gates_absent``,
``categories_absent``) for diagnostic transparency.

Usage::

    python -m src.llm_gate.postprocess.propagate \\
        --eval_path   results/gate_gpt-5.4_full/Acute2020/pred.json \\
        --dataset     Acute2020 \\
        --benchmark   benchmark/ --data-dir data/ \\
        --method      gate_gpt-5.4_full \\
        --output_dir  results/gate_gpt-5.4_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from src.bench import BenchmarkLoader, parse_perturbations
from src.llm_gate.inference.output_schema import (
    _polygon_area, group_by_sample_step, is_absent_gate,
)


TIEBREAKS = ('smallest', 'priority')


def _gate_area(gate: dict) -> float:
    if 'vertices' in gate:
        return _polygon_area(gate['vertices'])
    return ((gate['x_max'] - gate['x_min'])
            * (gate['y_max'] - gate['y_min']))


def _gate_contains(gate: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean membership mask for parent cells.

    Both axis-aligned and polygon gates use **closed** boundaries — cells
    exactly on an edge / vertex count as inside. ``MplPath.contains_points``
    excludes boundary points by default; a tiny positive ``radius`` adds
    them back without materially expanding the polygon.
    """
    if 'vertices' in gate:
        path = MplPath(np.asarray(gate['vertices'], dtype=np.float64))
        pts = np.column_stack([x, y])
        return path.contains_points(pts, radius=1e-9)
    return (
        (x >= gate['x_min']) & (x <= gate['x_max']) &
        (y >= gate['y_min']) & (y <= gate['y_max'])
    )


def _ordered_gates(
    gates: dict, options: list[str], tiebreak: str,
) -> list[tuple[str, dict]]:
    """Return [(category, gate), ...] in *application order*.

    A category may map to either a single gate dict or a list of gate
    dicts (multi-rectangle / multi-polygon, union semantics). Lists are
    flattened — each rectangle becomes its own ``(category, gate)``
    entry — so the smallest-area tiebreak runs over individual
    rectangles, not categories.

    Later items overwrite earlier ones for cells in the intersection,
    so the desired winner must come **last**.

    - ``smallest``: largest area first → smallest area wins.
    - ``priority``: reversed ``options`` order → first in options wins.
                    Within a category, larger rectangles come first so
                    the smallest sub-rectangle wins for ties inside one
                    category as well.
    """
    flat: list[tuple[str, dict]] = []
    for c, g in gates.items():
        if g is None or c == 'Unassigned' or is_absent_gate(g):
            continue
        if isinstance(g, list):
            for sub in g:
                if isinstance(sub, dict) and not is_absent_gate(sub):
                    flat.append((c, sub))
        elif isinstance(g, dict):
            flat.append((c, g))

    if tiebreak == 'smallest':
        flat.sort(key=lambda kv: -_gate_area(kv[1]))
        return flat
    if tiebreak == 'priority':
        order = {c: i for i, c in enumerate(options) if c != 'Unassigned'}
        # Sort by (priority desc) so first-in-options is applied last;
        # within the same category, larger area first → smallest wins.
        flat.sort(key=lambda kv: (-order.get(kv[0], len(order)),
                                  -_gate_area(kv[1])))
        return flat
    raise ValueError(f'unknown tiebreak: {tiebreak!r}')


def _propagate_step(
    ti,
    step_record: dict,
    tiebreak: str,
) -> tuple[list[str | None], dict]:
    """Return (``predicted_labels`` length n_parent, info dict)."""
    xy = ti.X_2d
    if xy is None or len(xy) == 0:
        return [], {
            'method':           f'gate_rect_{tiebreak}',
            'n_parent_cells':   0,
            'n_gates_total':    0,
            'n_gates_valid':    0,
            'n_categories':     0,
        }

    n_parent = int(len(xy))
    x = xy[:, 0]
    y = xy[:, 1]

    gates = step_record.get('gates', {}) or {}
    options = step_record.get('options', []) or list(ti.categories)
    targets = [c for c in options if c != 'Unassigned']
    n_total = len(targets)
    ordered = _ordered_gates(gates, options, tiebreak)
    cats_with_gate = sorted({c for c, _ in ordered})
    n_valid = len(cats_with_gate)
    n_rectangles = len(ordered)
    cats_absent = sorted(c for c in targets if is_absent_gate(gates.get(c)))

    info: dict = {
        'method':           f'gate_rect_{tiebreak}',
        'n_parent_cells':   n_parent,
        'n_gates_total':    n_total,
        'n_gates_valid':    n_valid,
        'n_gates_absent':   len(cats_absent),
        'n_rectangles':     n_rectangles,
        'n_categories':     n_valid,
        'categories_with_gate':  cats_with_gate,
        'categories_absent':     cats_absent,
    }

    predicted: list[str | None] = ['Unassigned'] * n_parent
    if n_valid == 0:
        return predicted, info

    pred_arr = np.array(predicted, dtype=object)
    for cat, g in ordered:
        inside = _gate_contains(g, x, y)
        if inside.any():
            pred_arr[inside] = cat
    return pred_arr.tolist(), info


def export_dataset(
    eval_path: str,
    dataset_name: str,
    benchmark_dir: str,
    data_dir: str,
    method: str,
    output_dir: str,
    tiebreak: str,
    loader: BenchmarkLoader | None = None,
) -> int:
    preds = json.loads(Path(eval_path).read_text())
    by_step = group_by_sample_step(preds)

    pred_meta = preds.get('meta', {}) if isinstance(preds, dict) else {}
    pred_slug = pred_meta.get('perturbation')
    loader = (loader if loader is not None
              else BenchmarkLoader(benchmark_dir, data_dir))
    loader_slug = loader.perturbation_name
    if pred_slug != loader_slug:
        raise SystemExit(
            f'[ERROR] perturbation mismatch between pred.json and loader: '
            f'pred={pred_slug!r}, loader={loader_slug!r}. Pass the same '
            f'--perturbation spec to src.llm_gate.postprocess.propagate '
            f'as was used when generating gate_task.'
        )

    out_root = Path(output_dir)
    n_written = 0
    n_skipped = 0

    for (sample, step_num), record in sorted(by_step.items()):
        step_str = f'step_{step_num:02d}'
        step_dir = Path(benchmark_dir) / dataset_name / sample / step_str
        task_json = step_dir / 'task.json'
        if not task_json.exists():
            print(f'    [SKIP] {sample}/{step_str}: missing task.json')
            n_skipped += 1
            continue

        ti = loader.load_task(str(task_json))
        if ti is None:
            print(f'    [SKIP] {sample}/{step_str}: load_task returned None')
            n_skipped += 1
            continue

        predicted_labels, info = _propagate_step(ti, record, tiebreak)

        ti.save_prediction(
            predicted_labels,
            output_dir=out_root,
            method=method,
            info=info,
        )

        print(f'    [ok] {sample}/{step_str}: {info["n_parent_cells"]} cells, '
              f'{info["n_gates_valid"]}/{info["n_gates_total"]} gates valid')
        n_written += 1

    print(f'  → {n_written} prediction.json written ({n_skipped} skipped)')
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Propagate LLM rectangular gates to all parent cells; '
                    'emit src.eval-compatible prediction.json per step.',
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
    parser.add_argument('--tiebreak', choices=TIEBREAKS, default='smallest',
                        help='Resolution rule when a cell falls inside '
                             'multiple gates. Default: smallest (smallest-'
                             'area gate wins).')
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'passed to src.llm_gate.task). Reapplied when rebuilding '
             'parent cells so x/y values align with the gates.',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    import json
    from src.hard_depletions import add_hard_depletion_args, resolve_hard_depletion_args
    from src.hard_shifts import add_hard_shift_args, resolve_hard_shift_args
    add_hard_depletion_args(parser)
    add_hard_shift_args(parser)
    args = parser.parse_args()

    if len(args.eval_path) != len(args.dataset):
        parser.error('--eval_path and --dataset must have matching counts')

    resolve_hard_depletion_args(args)
    resolve_hard_shift_args(
        args,
        meta_channels_lookup=lambda ds: json.loads(
            (Path('data') / ds / 'parquet' / 'meta.json').read_text()
        )['channels'],
    )

    if getattr(args, '_resolved_perturbation', None) is not None:
        perturbation = args._resolved_perturbation
        perturbation_name = args.perturbation_name
    else:
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
            tiebreak=args.tiebreak,
            loader=loader,
        )

    print(f'\nDone. Total {total} prediction.json → {args.output_dir}')


if __name__ == '__main__':
    main()
