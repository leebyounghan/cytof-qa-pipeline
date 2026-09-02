#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cell-to-Sentence (grid baseline) — uniform (x, y) bins → per-bin mean.

Each non-empty bin in a data-driven uniform grid over the step's (x, y)
plane becomes one slot. The slot's cell sentence is built from the bin's
mean protein vector (HVP markers, sorted high→low). This is the v1
"grid + per-bin mean" approach revived as a side-by-side baseline to the
canonical two-stage clustering pipeline (``src.llm.c2s``).

Pipeline per gating step:

  1. Load parent cells (via bench loader).
  2. Compute top-HVP markers over parent (axis markers excluded).
  3. Bin parent cells in (x, y) on a uniform integer-aligned grid
     (``--bin-size``, default 1.0). One slot per non-empty bin.
  4. For each bin: protein mean across cells in the bin; cell sentence
     from the bin mean's HVP values (high→low). ``cell_index`` /
     ``parquet_row`` point at the cell whose protein vector is nearest
     (Euclidean) to the bin mean — this keeps downstream readers
     (``src.llm.inference.data_loader``)
     working unchanged. ``x``/``y`` = mean of the bin's parent cells.
  5. Canonical top-left → bottom-right slot reordering on (x, y).
  6. Persist ``c2s_grid.json``, ``cell2grid.npz``, and
     ``c2s_grid_scatter.png``.

    python -m src.llm.c2s_grid --benchmark benchmark/ --data-dir data/
    python -m src.llm.c2s_grid --benchmark benchmark/ --data-dir data/ \\
                              --bin-size 0.5 --datasets Acute2020
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
from src.llm.utils.task_input_adapter import (
    _TaskInputDataset,
    _add_unassigned,
    _step_from_task,
)
from src.llm.utils.features import compute_top_hvp
from src.llm_gate.utils.axis_distribution import compute_axis_bins
from src.llm_gate.utils.joint_distribution import detect_joint_peaks_valleys

warnings.filterwarnings('ignore', category=UserWarning)

DEFAULT_BIN_SIZE = 1.0
DEFAULT_N_BINS: int | None = None
DEFAULT_TOP_HVP_N: int | None = None
DEFAULT_BASE_SEED = 42

_CYTOF_COHORTS = frozenset({'Acute2020', 'Acute2021', 'Bjornson', 'Vaccine'})


def _detect_modality(cohort: str, step: dict) -> str:
    if cohort in _CYTOF_COHORTS:
        return 'cytof'
    note = (step.get('note') or '') if isinstance(step, dict) else ''
    return 'cytof' if 'CyTOF' in note else 'flow'


# ── Grid binning ────────────────────────────────────────────────────────────

def _assign_grid_from_thresholds(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_thresholds: list[float],
    y_thresholds: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Grid whose interior edges are explicit thresholds (e.g. flowDensity).

    Outer edges are clamped to data min/max. Thresholds that fall outside
    the data range are dropped. ``bin_ids = col * n_y + row``.
    """
    x_lo = float(np.min(x_vals))
    x_hi = float(np.max(x_vals))
    y_lo = float(np.min(y_vals))
    y_hi = float(np.max(y_vals))
    if x_hi <= x_lo:
        x_hi = x_lo + 1e-9
    if y_hi <= y_lo:
        y_hi = y_lo + 1e-9
    xt = sorted(t for t in x_thresholds if x_lo < float(t) < x_hi)
    yt = sorted(t for t in y_thresholds if y_lo < float(t) < y_hi)
    x_edges = np.array([x_lo, *xt, x_hi], dtype=np.float64)
    y_edges = np.array([y_lo, *yt, y_hi], dtype=np.float64)
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    col_idx = np.clip(np.digitize(x_vals, x_edges) - 1, 0, n_x - 1)
    row_idx = np.clip(np.digitize(y_vals, y_edges) - 1, 0, n_y - 1)
    bin_ids = col_idx * n_y + row_idx
    return col_idx, row_idx, bin_ids, x_edges, y_edges, n_x, n_y


def _assign_uniform_grid(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bin_size: float | None = None,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Uniform grid over (x, y).

    Two modes (exactly one must be set):
    - ``n_bins``: fixed n_bins × n_bins; edges span the data range exactly.
    - ``bin_size``: integer-aligned edges (..., -1, 0, 1, 2, ...) at the
      given step.

    Returns ``(col_idx, row_idx, bin_ids, x_edges, y_edges, n_x, n_y)``.
    ``bin_ids = col * n_y + row`` (col-major, matches v1 c2s_utils).
    """
    if (bin_size is None) == (n_bins is None):
        raise ValueError('pass exactly one of bin_size or n_bins')
    if n_bins is not None:
        x_lo = float(np.min(x_vals))
        x_hi = float(np.max(x_vals))
        y_lo = float(np.min(y_vals))
        y_hi = float(np.max(y_vals))
        if x_hi <= x_lo:
            x_hi = x_lo + 1.0
        if y_hi <= y_lo:
            y_hi = y_lo + 1.0
        n_x = n_y = int(n_bins)
    else:
        x_lo = np.floor(float(np.min(x_vals)) / bin_size) * bin_size
        x_hi = np.ceil(float(np.max(x_vals)) / bin_size) * bin_size
        y_lo = np.floor(float(np.min(y_vals)) / bin_size) * bin_size
        y_hi = np.ceil(float(np.max(y_vals)) / bin_size) * bin_size
        if x_hi <= x_lo:
            x_hi = x_lo + bin_size
        if y_hi <= y_lo:
            y_hi = y_lo + bin_size
        n_x = int(round((x_hi - x_lo) / bin_size))
        n_y = int(round((y_hi - y_lo) / bin_size))
    x_edges = np.linspace(x_lo, x_hi, n_x + 1)
    y_edges = np.linspace(y_lo, y_hi, n_y + 1)
    col_idx = np.clip(np.digitize(x_vals, x_edges) - 1, 0, n_x - 1)
    row_idx = np.clip(np.digitize(y_vals, y_edges) - 1, 0, n_y - 1)
    bin_ids = col_idx * n_y + row_idx
    return col_idx, row_idx, bin_ids, x_edges, y_edges, n_x, n_y


# ── Core pipeline ───────────────────────────────────────────────────────────

def generate_c2s_grid(
    dataset: _TaskInputDataset,
    step: dict,
    bin_size: float | None = DEFAULT_BIN_SIZE,
    n_bins: int | None = DEFAULT_N_BINS,
    x_thresholds: list[float] | None = None,
    y_thresholds: list[float] | None = None,
    top_hvp_n: int | None = DEFAULT_TOP_HVP_N,
    base_seed: int = DEFAULT_BASE_SEED,
    modality: str = 'cytof',
    cofactor: float | None = None,
    output_dir: str | None = None,
    scatter_filename: str = 'c2s_grid_scatter.png',
    mapping_path: str | None = None,
    cohort: str | None = None,
) -> dict | None:
    """Run the grid c2s pipeline on one gating step."""
    parent_mask = dataset.get_parent_mask(step)
    parent_idx = np.where(parent_mask)[0]
    if len(parent_idx) == 0:
        return None

    x_all, y_all = dataset.get_xy_vals(step)
    x_vals = x_all[parent_idx]
    y_vals = y_all[parent_idx]

    protein_names = dataset.protein_names
    if dataset._protein_idx:
        parent_protein = dataset.X[np.ix_(parent_idx, dataset._protein_idx)]
    else:
        parent_protein = np.empty((len(parent_idx), 0), dtype=np.float32)

    xm_disp = dataset.get_display_name(step['x_marker'])
    ym_disp = dataset.get_display_name(step['y_marker'])

    top_hvp = compute_top_hvp(
        parent_protein, protein_names, top_n=top_hvp_n,
        exclude=(xm_disp, ym_disp),
    )

    n_parent = int(len(parent_idx))
    gt_labels = dataset.get_gt_labels(step)
    orig = dataset._ti.original_indices
    parquet_rows_all = orig[parent_idx] if orig is not None else parent_idx

    if x_thresholds is not None or y_thresholds is not None:
        col_idx, row_idx, bin_ids, x_edges, y_edges, n_x, n_y = _assign_grid_from_thresholds(
            x_vals, y_vals,
            x_thresholds=list(x_thresholds or []),
            y_thresholds=list(y_thresholds or []),
        )
        bin_size_used = None
    elif n_bins is not None:
        col_idx, row_idx, bin_ids, x_edges, y_edges, n_x, n_y = _assign_uniform_grid(
            x_vals, y_vals, n_bins=n_bins,
        )
        bin_size_used = float((x_edges[-1] - x_edges[0]) / max(n_x, 1))
    else:
        col_idx, row_idx, bin_ids, x_edges, y_edges, n_x, n_y = _assign_uniform_grid(
            x_vals, y_vals, bin_size=bin_size,
        )
        bin_size_used = float(bin_size)

    # Compact bin_id → 0..K-1 over non-empty bins, in canonical col-major order.
    nonempty = np.unique(bin_ids)
    k_bins = int(len(nonempty))
    if k_bins < 1:
        return None
    bin_to_slot = np.full(int(nonempty.max()) + 1, -1, dtype=np.int32)
    for slot_i, bid in enumerate(nonempty):
        bin_to_slot[int(bid)] = slot_i
    slot_labels = bin_to_slot[bin_ids]  # (n_parent,), 0..k_bins-1

    # HVP column lookup hoisted out of the per-bin loop.
    name_to_idx = {m: i for i, m in enumerate(protein_names)}
    hvp_cols_list = [(m, name_to_idx[m]) for m in top_hvp
                     if m in name_to_idx]
    hvp_names = [m for m, _ in hvp_cols_list]
    hvp_cols = np.fromiter(
        (i for _, i in hvp_cols_list),
        dtype=np.int64, count=len(hvp_cols_list),
    )

    cells: dict[str, dict] = {}
    gt_answers: dict[str, str] = {}
    rep_parent_local = np.full(k_bins, -1, dtype=np.int32)

    for slot_i in range(k_bins):
        mask = slot_labels == slot_i
        locals_in_mask = np.where(mask)[0]
        n_in_bin = int(locals_in_mask.size)
        if n_in_bin == 0:
            continue

        # Bin means: (x, y) and protein.
        bin_x = float(x_vals[mask].mean())
        bin_y = float(y_vals[mask].mean())
        if parent_protein.size:
            prot_in = parent_protein[mask]
            prot_mean = prot_in.mean(axis=0)
            d2 = np.einsum(
                'ij,ij->i', prot_in - prot_mean, prot_in - prot_mean,
            )
            rep_local = int(locals_in_mask[int(np.argmin(d2))])
        else:
            prot_mean = np.empty(0, dtype=np.float32)
            rep_local = int(locals_in_mask[0])
        rep_parent_local[slot_i] = rep_local

        # Cell sentence from the bin's protein mean (high → low over HVP).
        if hvp_cols.size and prot_mean.size:
            mean_vals = prot_mean[hvp_cols]
            order = np.argsort(-mean_vals, kind='stable')
            sentence = ' > '.join(
                f'{hvp_names[j]}({float(mean_vals[j]):.2f})' for j in order
            )
        else:
            sentence = ''

        rep_sample_i = int(parent_idx[rep_local])
        rep_parquet = int(parquet_rows_all[rep_local])

        if gt_labels is not None:
            rep_gt = gt_labels[parent_idx[rep_local]]
            s = str(rep_gt) if rep_gt is not None else ''
            gt_answer = 'Unassigned' if (not s or s.lower() == 'nan') else s
        else:
            gt_answer = 'Unassigned'

        bid_int = int(nonempty[slot_i])
        cells[str(slot_i + 1)] = dict(
            cluster_id=int(slot_i),
            bin_id=bid_int,
            col=int(bid_int // n_y),
            row=int(bid_int % n_y),
            n_cells=n_in_bin,
            x=bin_x,
            y=bin_y,
            cell_index=rep_sample_i,
            parquet_row=rep_parquet,
            cell_sentence=sentence,
        )
        gt_answers[str(rep_sample_i)] = gt_answer

    # Canonical slot-id reordering: top-left (low x, high y) → bottom-right.
    n_final = len(cells)
    if n_final > 1:
        rep_xy = np.array(
            [[cells[str(i + 1)]['x'], cells[str(i + 1)]['y']]
             for i in range(n_final)],
            dtype=np.float64,
        )
        mn = rep_xy.min(axis=0)
        mx = rep_xy.max(axis=0)
        span = np.where(mx > mn, mx - mn, 1.0)
        xyn = (rep_xy - mn) / span
        score = xyn[:, 0] - xyn[:, 1]
        new_order = np.lexsort((-xyn[:, 1], xyn[:, 0], score))
        old_to_new = np.empty(n_final, dtype=np.int32)
        old_to_new[new_order] = np.arange(n_final, dtype=np.int32)

        remap_bin = np.full(int(nonempty.max()) + 1, -1, dtype=np.int32)
        for old_slot_i, bid in enumerate(nonempty):
            remap_bin[int(bid)] = old_to_new[old_slot_i]
        slot_labels = remap_bin[bin_ids]
        rep_parent_local = rep_parent_local[new_order]

        new_cells: dict[str, dict] = {}
        for new_id, old_slot in enumerate(new_order):
            entry = dict(cells[str(int(old_slot) + 1)])
            entry['cluster_id'] = int(new_id)
            new_cells[str(new_id + 1)] = entry
        cells = new_cells

    x64 = np.asarray(x_vals, dtype=np.float64)
    y64 = np.asarray(y_vals, dtype=np.float64)
    joint = detect_joint_peaks_valleys(x64, y64)

    result = dict(
        step=step['step'],
        cohort=cohort,
        x_marker=xm_disp,
        y_marker=ym_disp,
        parent=step.get('parent', 'ALL'),
        note=step.get('note', ''),
        tips=step.get('tips', ''),
        options=_add_unassigned(step.get('annotation categories', [])),
        n_parent_cells=n_parent,
        axis_distribution={
            'x': compute_axis_bins(x64),
            'y': compute_axis_bins(y64),
        },
        joint_distribution=joint,
        top_hvp_markers=top_hvp,
        n_clusters=int(len(cells)),
        method=('grid_thresholds' if x_thresholds is not None or y_thresholds is not None else 'grid'),
        bin_size=bin_size_used,
        n_bins=(int(n_bins) if n_bins is not None else None),
        x_thresholds=(list(x_thresholds) if x_thresholds is not None else None),
        y_thresholds=(list(y_thresholds) if y_thresholds is not None else None),
        n_x=int(n_x),
        n_y=int(n_y),
        x_edges=x_edges.tolist(),
        y_edges=y_edges.tolist(),
        base_seed=int(base_seed),
        modality=modality,
        cofactor=cofactor,
        cells=cells,
        gt_answers=gt_answers,
    )

    if mapping_path:
        np.savez(
            mapping_path,
            parent_indices=parquet_rows_all.astype(np.int32),
            cluster_labels=slot_labels.astype(np.int32),
            rep_parent_local=rep_parent_local.astype(np.int32),
            bin_ids=bin_ids.astype(np.int32),
        )

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            gt_parent = (gt_labels[parent_idx] if gt_labels is not None
                         else np.array(['Unassigned'] * n_parent))
            n_final = len(cells)
            bin_rep_xy = np.array(
                [[cells[str(i + 1)]['x'], cells[str(i + 1)]['y']]
                 for i in range(n_final)],
                dtype=np.float64,
            ) if n_final else None
            method_label = (f'grid {n_x}×{n_y} '
                            f'({"thresholds" if x_thresholds is not None or y_thresholds is not None else "uniform"})')
            from src.llm.utils.plot_helpers import plot_grid_cells
            plot_grid_cells(
                x_vals, y_vals, gt_parent,
                categories=result['options'],
                bin_slot_labels=slot_labels,
                rep_parent_local=rep_parent_local,
                x_edges=x_edges, y_edges=y_edges,
                x_marker=xm_disp, y_marker=ym_disp,
                step_num=step['step'],
                method_label=method_label,
                bin_rep_xy=bin_rep_xy,
                output_path=str(out_dir / scatter_filename),
            )
        except Exception as e:  # noqa: BLE001
            print(f'  [WARN] scatter plot failed: {e}')

    return result


# ── CLI / orchestration ─────────────────────────────────────────────────────

def _output_filenames(perturb_name: str | None) -> dict[str, str]:
    if not perturb_name:
        return {
            'json': 'c2s_grid.json',
            'scatter': 'c2s_grid_scatter.png',
            'mapping': 'cell2grid.npz',
        }
    return {
        'json': f'c2s_grid__{perturb_name}.json',
        'scatter': f'c2s_grid__{perturb_name}_scatter.png',
        'mapping': f'cell2grid__{perturb_name}.npz',
    }


@dataclass(frozen=True)
class _WorkerConfig:
    bin_size: float | None
    n_bins: int | None
    thresholds_dir: str | None
    top_hvp_n: int | None
    base_seed: int
    modality: str
    no_plots: bool
    perturbation_name: str | None
    filenames: dict[str, str]
    skip_existing: bool = False


def _process_one_task(ti: TaskInput, cfg: _WorkerConfig) -> tuple[str, int | None]:
    dataset = _TaskInputDataset(ti)
    step = _step_from_task(ti)
    step_dir = ti._step_dir
    label = f'{ti.dataset}/{ti.sample}/{ti.step_dir_name}'

    if cfg.skip_existing and (step_dir / cfg.filenames['json']).exists():
        return f'{label} [skip-existing]', -1

    x_thr = y_thr = None
    if cfg.thresholds_dir:
        thr_path = (Path(cfg.thresholds_dir) / ti.dataset / ti.sample
                    / ti.step_dir_name / 'thresholds.json')
        if not thr_path.exists():
            return f'{label} [skip-no-thresholds]', -1
        thr = json.load(open(thr_path))
        x_thr = list(thr.get('x_thresholds') or [])
        y_thr = list(thr.get('y_thresholds') or [])

    modality = (_detect_modality(ti.dataset, step)
                if cfg.modality == 'auto' else cfg.modality)

    result = generate_c2s_grid(
        dataset, step,
        bin_size=cfg.bin_size,
        n_bins=cfg.n_bins,
        x_thresholds=x_thr,
        y_thresholds=y_thr,
        top_hvp_n=cfg.top_hvp_n,
        base_seed=cfg.base_seed,
        modality=modality,
        cofactor=ti.cofactor,
        output_dir=None if cfg.no_plots else str(step_dir),
        scatter_filename=cfg.filenames['scatter'],
        mapping_path=str(step_dir / cfg.filenames['mapping']),
        cohort=ti.dataset,
    )
    if not result:
        return label, None

    if cfg.perturbation_name:
        result['perturbation'] = cfg.perturbation_name

    with open(step_dir / cfg.filenames['json'], 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return label, int(result['n_clusters'])


def run_bench(args: argparse.Namespace) -> None:
    loader = BenchmarkLoader.from_cli(args)
    cfg = _WorkerConfig(
        bin_size=(None if (args.n_bins is not None or args.thresholds_dir)
                  else args.bin_size),
        n_bins=(None if args.thresholds_dir else args.n_bins),
        thresholds_dir=args.thresholds_dir,
        top_hvp_n=args.top_hvp_n,
        base_seed=args.base_seed,
        modality=args.modality,
        no_plots=args.no_plots,
        perturbation_name=loader.perturbation_name,
        filenames=_output_filenames(loader.perturbation_name),
        skip_existing=args.skip_existing,
    )

    n_workers = max(1, args.n_workers)
    tasks = loader.iter_tasks_from_cli(args)
    worker = partial(_process_one_task, cfg=cfg)

    total, skipped = 0, 0

    def _report(label: str, n_clusters: int | None) -> None:
        nonlocal total, skipped
        if n_clusters is None:
            skipped += 1
            print(f'  [skip] {label} (empty parent)')
        elif n_clusters < 0:
            skipped += 1
            print(f'  [skip] {label}')
        else:
            total += 1
            print(f'  [ok]   {label}: {n_clusters} bins')

    if n_workers == 1:
        for ti in tasks:
            _report(*worker(ti))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for label, n_clusters in ex.map(worker, tasks, chunksize=1):
                _report(label, n_clusters)

    suffix = (f' (perturbation: {loader.perturbation_name})'
              if loader.perturbation_name else '')
    print(f'\n[DONE] {total} c2s_grid items written{suffix}; '
          f'{skipped} skipped; n_workers={n_workers}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate Cell Sentences via uniform (x, y) grid '
                    'binning + per-bin protein mean.',
    )
    BenchmarkLoader.add_cli_args(parser)
    parser.add_argument('--bin-size', type=float, default=DEFAULT_BIN_SIZE,
                        help='Uniform grid bin size on both x and y axes '
                             '(integer-aligned edges). Ignored if --n-bins '
                             f'is set. Default: {DEFAULT_BIN_SIZE}.')
    parser.add_argument('--n-bins', type=int, default=DEFAULT_N_BINS,
                        help='Fixed n_bins × n_bins grid spanning the data '
                             'range exactly (overrides --bin-size). e.g. '
                             '--n-bins 5 → 5×5 grid.')
    parser.add_argument('--thresholds-dir', type=str, default=None,
                        help='Directory of per-step threshold JSONs '
                             '({cohort}/{sample}/step_NN/thresholds.json '
                             'with x_thresholds/y_thresholds). Overrides '
                             '--bin-size and --n-bins. Outer edges clamp '
                             'to data min/max.')
    parser.add_argument('--top-hvp-n', type=int, default=DEFAULT_TOP_HVP_N,
                        help='Truncate HVP ranking / cell sentence to first N. '
                             'Default: None (all proteins, high→low).')
    parser.add_argument('--base-seed', type=int, default=DEFAULT_BASE_SEED,
                        help='Seed (kept for parity with c2s.py; grid path is '
                             'deterministic).')
    parser.add_argument('--modality', choices=['cytof', 'flow', 'auto'],
                        default='auto',
                        help='Recorded in c2s_grid.json. "auto" detects '
                             'per-cohort (Acute2020/Acute2021/Bjornson/Vaccine '
                             '→ cytof, others → flow unless task note says '
                             'CyTOF).')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip c2s_grid_scatter.png generation.')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Parallel worker processes. Default: 4')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip steps whose c2s_grid.json already exists.')
    args = parser.parse_args()
    run_bench(args)


if __name__ == '__main__':
    main()
