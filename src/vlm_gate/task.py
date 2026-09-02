#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent-Gate — per-step task JSON + background scatter generator.

For each gating step, emit a ``gate_task.json`` describing the parent
population's (x, y) plane in a form the agent can use to draw
axis-aligned rectangular gates per category, **plus** a decoration-free
``scatter_bg.png`` of the parent population that the ``render_gate_overlay``
tool re-wraps with the agent's proposed gate boxes.

Pipeline per gating step:

  1. Load parent cells via ``BenchmarkLoader`` / ``TaskInput``.
  2. Compute per-axis basic statistics (min / max / quantiles) over the
     parent cells.
  3. Compute richer axis-distribution summaries — 1D histogram + KDE
     peak/valley hints, 2D joint peaks, density clouds — that the prompt
     builder surfaces to the agent.
  4. Render ``scatter_bg.png`` — a clean, axes-fill-figure density
     scatter of the parent (x, y) cells whose pixel bounds map *exactly*
     to ``scatter_extent`` ``[xmin, xmax, ymin, ymax]`` (also persisted
     in the JSON). The agentic ``render_gate_overlay`` tool loads this
     PNG with ``imshow(extent=...)`` so it never re-renders the (slow)
     dense scatter — it only overlays the proposed gate boxes.
  5. Persist ``gate_task.json``. No precomputed cell→cluster mapping is
     needed: at propagate time we read parent cells fresh from the
     benchmark task and apply the gates directly.

Output is **independent** of any LLM call — it is purely a description
of the gating step's parent population.

CLI::

    python -m src.vlm_gate.task --benchmark benchmark/ --data-dir data/
    python -m src.vlm_gate.task --benchmark benchmark/ --data-dir data/ \\
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

import matplotlib
matplotlib.use('Agg')  # headless — task.py runs in ProcessPoolExecutor workers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.bench import BenchmarkLoader, TaskInput
from src.vlm_gate.utils.axis_distribution import compute_axis_bins
from src.vlm_gate.utils.density_clouds import detect_density_clouds
from src.vlm_gate.utils.joint_distribution import detect_joint_peaks_valleys
from src.vlm_gate.utils.task_input_adapter import (
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


# ── Background scatter ──────────────────────────────────────────────────────
#
# The agentic ``render_gate_overlay`` tool needs the parent population's
# actual (x, y) scatter to draw boxes on. Re-rendering a 60k-point
# scatter on every tool call would be far too slow, so ``task.py``
# pre-renders it ONCE here as a decoration-free PNG whose pixel bounds
# map *exactly* to ``scatter_extent``. The tool then does
# ``imshow(bg, extent=scatter_extent, aspect='auto', origin='upper')``
# and overlays the gate boxes in data coordinates — matplotlib handles
# the data→pixel mapping, so as long as the tool reuses the same extent
# the overlay lands on the right cells.

_BG_FIGSIZE = 6.4         # inches (square)
_BG_DPI = 100             # → 640×640 px background
_BG_MAX_POINTS = 60_000   # subsample cap for render speed


def _compute_density(x: np.ndarray, y: np.ndarray, bins: int = 200) -> np.ndarray:
    """Per-point 2-D histogram density (same approach as src.preprocess)."""
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    return hist[xi, yi]


def _scatter_extent(x: np.ndarray, y: np.ndarray) -> list[float]:
    """``[xmin, xmax, ymin, ymax]`` data extent, padded for degenerate axes."""
    xmn, xmx = float(np.min(x)), float(np.max(x))
    ymn, ymx = float(np.min(y)), float(np.max(y))
    if xmx <= xmn:
        xmn, xmx = xmn - 0.5, xmx + 0.5
    if ymx <= ymn:
        ymn, ymx = ymn - 0.5, ymx + 0.5
    return [xmn, xmx, ymn, ymx]


def _render_background_scatter(
    x_vals: np.ndarray, y_vals: np.ndarray,
    out_path: Path, extent: list[float],
) -> None:
    """Render a decoration-free density scatter whose pixel bounds == extent.

    The axes fill the entire figure (``add_axes([0, 0, 1, 1])``), all
    decorations are off, and ``savefig`` is called WITHOUT
    ``bbox_inches='tight'`` — so the saved PNG's pixels correspond
    one-to-one to the ``[xmin, xmax, ymin, ymax]`` data rectangle. The
    overlay tool reuses this exact extent with ``imshow``.
    """
    xmn, xmx, ymn, ymx = extent
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    if len(x) > _BG_MAX_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), _BG_MAX_POINTS, replace=False)
        x, y = x[idx], y[idx]

    fig = plt.figure(figsize=(_BG_FIGSIZE, _BG_FIGSIZE), dpi=_BG_DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    if len(x) > 0:
        density = _compute_density(x, y)
        order = np.argsort(density)
        ax.scatter(
            x[order], y[order], s=2, alpha=0.45,
            c=density[order], cmap='viridis', linewidths=0, rasterized=True,
        )
    ax.set_xlim(xmn, xmx)
    ax.set_ylim(ymn, ymx)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=_BG_DPI)  # NO bbox_inches='tight' — keep extent
    plt.close(fig)


# ── Core pipeline ───────────────────────────────────────────────────────────

def generate_gate_task(
    dataset: _TaskInputDataset,
    step: dict,
    modality: str = 'cytof',
    cofactor: float | None = None,
    cohort: str | None = None,
    scatter_png_path: Path | None = None,
) -> dict | None:
    """Build the per-step ``gate_task.json`` payload.

    When ``scatter_png_path`` is given, also render the decoration-free
    background scatter there and record ``scatter_bg`` (filename) +
    ``scatter_extent`` in the returned payload.
    """
    parent_mask = dataset.get_parent_mask(step)
    parent_idx = np.where(parent_mask)[0]
    n_parent = int(len(parent_idx))
    if n_parent == 0:
        return None

    x_all, y_all = dataset.get_xy_vals(step)
    x_vals = x_all[parent_idx].astype(np.float32)
    y_vals = y_all[parent_idx].astype(np.float32)

    scatter_extent = _scatter_extent(x_vals, y_vals)
    scatter_bg_name: str | None = None
    if scatter_png_path is not None:
        _render_background_scatter(
            x_vals, y_vals, scatter_png_path, scatter_extent,
        )
        scatter_bg_name = scatter_png_path.name

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
        scatter_bg=scatter_bg_name,
        scatter_extent=scatter_extent,
    )


# ── CLI / orchestration ─────────────────────────────────────────────────────

def _output_filename(perturb_name: str | None) -> str:
    return (f'gate_task__{perturb_name}.json'
            if perturb_name else 'gate_task.json')


def _scatter_filename(perturb_name: str | None) -> str:
    """Background scatter PNG name — carries the perturbation slug so a
    perturbed run's scatter (different (x, y)) never shadows the clean one."""
    return (f'scatter_bg__{perturb_name}.png'
            if perturb_name else 'scatter_bg.png')


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
    scatter_filename: str
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
        scatter_png_path=step_dir / cfg.scatter_filename,
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
        scatter_filename=_scatter_filename(loader.perturbation_name),
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
        description='Generate per-step gate_task.json + scatter_bg.png '
                    'describing each gating step\'s parent population for '
                    'the agent-gate pipeline (rectangular axis-aligned '
                    'gates, drawn via the render_gate_overlay tool).',
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
