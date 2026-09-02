#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-Gate — per-step task JSON generator.

For each gating step, emit a ``gate_task.json`` describing the parent
population's (x, y) plane in a form the LLM can use to draw axis-aligned
rectangular gates per category.

Pipeline per gating step:

  1. Load parent cells via ``BenchmarkLoader`` / ``TaskInput``.
  2. Compute per-axis basic statistics (min / max / quantiles) over the
     parent cells.
  3. (Reserved) Compute richer axis-distribution summaries — 1D
     histogram, 2D heatmap, mode hints, ... — that the prompt builder
     surfaces to the LLM. This part is intentionally left as a TODO; the
     representation is being decided separately. Until then, only the
     basic stats are persisted.
  4. Persist ``gate_task.json``. No precomputed cell→cluster mapping is
     needed: at propagate time we read parent cells fresh from the
     benchmark task and apply the gates directly.

Output is **independent** of any LLM call — it is purely a description
of the gating step's parent population.

CLI::

    python -m src.llm_gate.task --benchmark benchmark/ --data-dir data/
    python -m src.llm_gate.task --benchmark benchmark/ --data-dir data/ \\
                                --datasets Acute2020 --n-workers 8
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from src.bench import BenchmarkLoader, TaskInput
from src.llm_gate.utils.axis_distribution import compute_axis_bins
from src.llm_gate.utils.density_clouds import detect_density_clouds
from src.llm_gate.utils.joint_distribution import detect_joint_peaks_valleys
from src.llm_gate.utils.task_input_adapter import (
    _TaskInputDataset,
    _add_unassigned,
    _step_from_task,
)

warnings.filterwarnings('ignore', category=UserWarning)


DEFAULT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _axis_summary(
    vals: np.ndarray, quantiles=DEFAULT_QUANTILES,
) -> dict:
    """Per-axis basic stats — min / max / quantiles.

    Richer summaries (histogram bins, mode hints, 2D heatmap) are added
    later when the prompt representation is decided.
    """
    if vals.size == 0:
        return {
            'min': None, 'max': None,
            'quantiles': {f'q{int(q * 100)}': None for q in quantiles},
        }
    qvals = np.quantile(vals, quantiles)
    return {
        'min':  float(vals.min()),
        'max':  float(vals.max()),
        'quantiles': {
            f'q{int(q * 100)}': float(v) for q, v in zip(quantiles, qvals)
        },
    }


def _gt_category_counts(
    gt_labels: np.ndarray | None,
    parent_idx: np.ndarray,
    options: list[str],
) -> dict[str, int]:
    """Per-category GT cell count over the parent population.

    Diagnostic only — surfaced into ``gate_task.json`` for inspection
    and never rendered into the LLM prompt.
    """
    if gt_labels is None or len(parent_idx) == 0:
        return {c: 0 for c in options}
    sub = gt_labels[parent_idx]
    counts: dict[str, int] = {c: 0 for c in options}
    for v in sub:
        s = str(v) if v is not None else ''
        if not s or s.lower() == 'nan':
            s = 'Unassigned'
        counts[s] = counts.get(s, 0) + 1
    return counts


# ── Core pipeline ───────────────────────────────────────────────────────────

def generate_gate_task(
    dataset: _TaskInputDataset,
    step: dict,
    modality: str = 'cytof',
    cofactor: float | None = None,
    cohort: str | None = None,
) -> dict | None:
    """Build the per-step ``gate_task.json`` payload."""
    parent_mask = dataset.get_parent_mask(step)
    parent_idx = np.where(parent_mask)[0]
    n_parent = int(len(parent_idx))
    if n_parent == 0:
        return None

    x_all, y_all = dataset.get_xy_vals(step)
    x_vals = x_all[parent_idx].astype(np.float32)
    y_vals = y_all[parent_idx].astype(np.float32)

    xm_disp = dataset.get_display_name(step['x_marker'])
    ym_disp = dataset.get_display_name(step['y_marker'])

    options = _add_unassigned(step.get('annotation categories', []))
    gt_labels = dataset.get_gt_labels(step)
    gt_counts = _gt_category_counts(gt_labels, parent_idx, options)

    x64 = x_vals.astype(np.float64)
    y64 = y_vals.astype(np.float64)
    joint = detect_joint_peaks_valleys(x64, y64)

    return dict(
        step=step['step'],
        cohort=cohort,
        x_marker=xm_disp,
        y_marker=ym_disp,
        parent=step.get('parent', 'ALL'),
        note=step.get('note', ''),
        tips=step.get('tips', ''),
        options=options,
        n_parent_cells=n_parent,
        modality=modality,
        cofactor=cofactor,
        x_summary=_axis_summary(x_vals),
        y_summary=_axis_summary(y_vals),
        axis_distribution={
            'x': compute_axis_bins(x64),
            'y': compute_axis_bins(y64),
        },
        joint_distribution=joint,
        density_clouds=detect_density_clouds(x64, y64, joint_info=joint),
        gt_category_counts=gt_counts,
    )


# ── CLI / orchestration ─────────────────────────────────────────────────────

def _output_filename(perturb_name: str | None) -> str:
    return (f'gate_task__{perturb_name}.json'
            if perturb_name else 'gate_task.json')


_CYTOF_COHORTS = frozenset({'Acute2020', 'Acute2021', 'Bjornson', 'Vaccine'})


def _detect_modality(cohort: str, task: dict) -> str:
    """Cohort-based modality auto-detection.

    Hard-coded CyTOF cohorts win first; otherwise the task note's
    ``CyTOF`` mention triggers cytof; everything else is flow.
    """
    if cohort in _CYTOF_COHORTS:
        return 'cytof'
    note = (task.get('note') or '') if isinstance(task, dict) else ''
    return 'cytof' if 'CyTOF' in note else 'flow'


@dataclass(frozen=True)
class _WorkerConfig:
    modality: str  # 'auto', 'cytof', or 'flow'
    perturbation_name: str | None
    filename: str
    skip_existing: bool = False


def _process_one_task(
    ti: TaskInput, cfg: _WorkerConfig,
) -> tuple[str, int | None]:
    dataset = _TaskInputDataset(ti)
    step = _step_from_task(ti)
    step_dir = ti._step_dir
    label = f'{ti.dataset}/{ti.sample}/{ti.step_dir_name}'

    out_path = step_dir / cfg.filename
    if cfg.skip_existing and out_path.exists():
        return f'{label} [skip-existing]', -1

    modality = (_detect_modality(ti.dataset, ti.task)
                if cfg.modality == 'auto' else cfg.modality)

    result = generate_gate_task(
        dataset, step,
        modality=modality,
        cofactor=ti.cofactor,
        cohort=ti.dataset,
    )
    if not result:
        return label, None

    if cfg.perturbation_name:
        result['perturbation'] = cfg.perturbation_name

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return label, int(result['n_parent_cells'])


def run_bench(args: argparse.Namespace) -> None:
    loader = BenchmarkLoader.from_cli(args)
    cfg = _WorkerConfig(
        modality=args.modality,
        perturbation_name=loader.perturbation_name,
        filename=_output_filename(loader.perturbation_name),
        skip_existing=args.skip_existing,
    )

    n_workers = max(1, args.n_workers)
    tasks = loader.iter_tasks_from_cli(args)
    worker = partial(_process_one_task, cfg=cfg)

    total, skipped = 0, 0

    def _report(label: str, n: int | None) -> None:
        nonlocal total, skipped
        if n is None:
            skipped += 1
            print(f'  [skip] {label} (empty parent)')
        elif n < 0:
            skipped += 1
            print(f'  [skip] {label}')
        else:
            total += 1
            print(f'  [ok]   {label}: {n} parent cells')

    if n_workers == 1:
        for ti in tasks:
            _report(*worker(ti))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for label, n in ex.map(worker, tasks, chunksize=1):
                _report(label, n)

    suffix = (f' (perturbation: {loader.perturbation_name})'
              if loader.perturbation_name else '')
    print(f'\n[DONE] {total} gate_task items written{suffix}; '
          f'{skipped} skipped; n_workers={n_workers}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate per-step gate_task.json describing each '
                    'gating step\'s parent population for the LLM-gate '
                    'pipeline (rectangular axis-aligned gates).',
    )
    BenchmarkLoader.add_cli_args(parser)
    parser.add_argument('--modality', choices=['auto', 'cytof', 'flow'],
                        default='auto',
                        help='Recorded in gate_task.json. Default "auto" '
                             'detects from cohort: CyTOF for Acute2020/2021/'
                             'Bjornson/Vaccine and any cohort whose task note '
                             'mentions "CyTOF"; everything else is Flow.')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Parallel worker processes. Default: 4')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip steps whose gate_task.json already exists.')
    args = parser.parse_args()
    run_bench(args)


if __name__ == '__main__':
    main()
