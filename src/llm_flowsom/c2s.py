#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cell-to-Sentence — single-stage clustering on the gating (x, y) plane.

Pipeline per gating step:

  1. Load parent cells (via bench loader).
  2. Compute top-HVP markers over parent (axis markers excluded).
  3. Cluster parent cells in the step's (x, y) plane. A uniform subsample
     (``--subsample-n``, default 10k) is used to FIT the clusterer; labels
     propagate to every parent cell by 1-NN in the (x, y) space.
  4. Emit one slot per cluster: pick a representative parent cell (1-NN to
     the cluster's protein centroid, Euclidean) and use its own values for
     everything — (x, y) position, HVP cell sentence (sorted high→low),
     and GT label.
  5. Persist ``c2s.json``, ``cell2cluster.npz``, and ``c2s_scatter.png``.

Supported clusterers (``--cluster-method``):

  - ``mbkm``    — MiniBatchKMeans (target K = ``--k``)
  - ``agglo``   — Ward Agglomerative (target K = ``--k``)
  - ``gmm``     — GaussianMixture (target K = ``--k``)
  - ``flowsom`` — SOM on the (x, y) plane (``--flowsom-grid``² neurons)
                  followed by hierarchical consensus metaclustering of
                  the SOM codebook into ``--flowsom-k`` final clusters
                  (canonical FlowSOM, average linkage). ``--k`` ignored.

Perturbation runs suffix all three outputs with the perturbation slug.

    python -m src.llm.c2s --benchmark benchmark/ --data-dir data/
    python -m src.llm.c2s --benchmark benchmark/ --data-dir data/ \\
                         --datasets Acute2020 --n-workers 8
    python -m src.llm.c2s --cluster-method flowsom --flowsom-grid 10
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from src.bench import BenchmarkLoader, TaskInput
from src.llm.utils.task_input_adapter import (
    _TaskInputDataset,
    _add_unassigned,
    _step_from_task,
)
from src.llm.utils.features import compute_top_hvp
from src.llm.utils.plot_helpers import plot_cluster_cells

warnings.filterwarnings('ignore', category=UserWarning)

CLUSTER_METHODS = ('mbkm', 'agglo', 'gmm', 'flowsom')

DEFAULT_CLUSTER_METHOD = 'agglo'
DEFAULT_K = 16
DEFAULT_SUBSAMPLE_N = 10_000
DEFAULT_TOP_HVP_N: int | None = None
DEFAULT_BASE_SEED = 42
DEFAULT_FLOWSOM_GRID = 10
DEFAULT_FLOWSOM_K = 20


# ── Clusterer fit ───────────────────────────────────────────────────────────

def _fit_clusterer(
    X: np.ndarray,
    method: str,
    k: int,
    seed: int,
    *,
    flowsom_grid: int = DEFAULT_FLOWSOM_GRID,
    flowsom_k: int = DEFAULT_FLOWSOM_K,
) -> np.ndarray:
    """Cluster ``X`` (n, d) → int32 label array (n,).

    For ``mbkm`` / ``agglo`` / ``gmm``, ``k`` is the target cluster count.
    For ``flowsom``, a SOM (``flowsom_grid``²) is trained and its codebook
    is hierarchically metaclustered into ``flowsom_k`` final clusters
    (average linkage — canonical FlowSOM). ``k`` is ignored.
    """
    if method == 'mbkm':
        return MiniBatchKMeans(
            n_clusters=k, random_state=seed, batch_size=2048,
            n_init=3, max_iter=100,
        ).fit_predict(X).astype(np.int32)
    if method == 'agglo':
        return AgglomerativeClustering(
            n_clusters=k, linkage='ward',
        ).fit_predict(X).astype(np.int32)
    if method == 'gmm':
        return GaussianMixture(
            n_components=k, random_state=seed,
            covariance_type='diag', max_iter=100, reg_covar=1e-2,
            init_params='k-means++',
        ).fit(X).predict(X).astype(np.int32)
    if method == 'flowsom':
        try:
            from minisom import MiniSom
        except ImportError as e:
            raise ImportError(
                "minisom is not installed. Run: uv add minisom"
            ) from e
        grid = int(flowsom_grid)
        som = MiniSom(
            grid, grid, X.shape[1],
            sigma=1.0, learning_rate=0.5, random_seed=seed,
        )
        som.random_weights_init(X)
        som.train(X, num_iteration=10_000, verbose=False)

        # Consensus metacluster the SOM codebook → final K clusters
        # (canonical FlowSOM step; without it we'd just have a SOM).
        codebook = som.get_weights().reshape(-1, X.shape[1])
        n_nodes = codebook.shape[0]
        k_eff = max(1, min(int(flowsom_k), n_nodes))
        if k_eff >= n_nodes:
            meta_of_node = np.arange(n_nodes, dtype=np.int32)
        else:
            meta_of_node = AgglomerativeClustering(
                n_clusters=k_eff, linkage='average',
            ).fit_predict(codebook).astype(np.int32)

        # Vectorised cell→neuron assignment (canonical FlowSOM: 1-NN against
        # the SOM codebook). NearestNeighbors over codebook avoids the Python
        # loop in MiniSom.winner; ordering matches reshape(-1, d) C-order, so
        # node_id == i*grid + j as in MiniSom.winner.
        nn = NearestNeighbors(n_neighbors=1).fit(codebook)
        node_ids = nn.kneighbors(X, return_distance=False)[:, 0]
        return meta_of_node[node_ids].astype(np.int32)
    raise ValueError(f'unknown cluster method: {method!r}')


def _cluster_xy(
    xy_full: np.ndarray,
    fit_idx: np.ndarray,
    method: str,
    k: int,
    seed: int,
    *,
    flowsom_grid: int = DEFAULT_FLOWSOM_GRID,
    flowsom_k: int = DEFAULT_FLOWSOM_K,
) -> np.ndarray:
    """Fit on ``xy_full[fit_idx]``, 1-NN propagate labels to every row."""
    X_fit = xy_full[fit_idx]
    fit_labels = _fit_clusterer(
        X_fit, method, k, seed,
        flowsom_grid=flowsom_grid, flowsom_k=flowsom_k,
    )
    if len(X_fit) == len(xy_full):
        return fit_labels
    nn = NearestNeighbors(n_neighbors=1).fit(X_fit)
    _, idx = nn.kneighbors(xy_full)
    return fit_labels[idx[:, 0]].astype(np.int32)


# ── Core pipeline ───────────────────────────────────────────────────────────

def generate_c2s(
    dataset: _TaskInputDataset,
    step: dict,
    cluster_method: str = DEFAULT_CLUSTER_METHOD,
    k: int = DEFAULT_K,
    subsample_n: int = DEFAULT_SUBSAMPLE_N,
    top_hvp_n: int | None = DEFAULT_TOP_HVP_N,
    base_seed: int = DEFAULT_BASE_SEED,
    flowsom_grid: int = DEFAULT_FLOWSOM_GRID,
    flowsom_k: int = DEFAULT_FLOWSOM_K,
    modality: str = 'cytof',
    cofactor: float | None = None,
    output_dir: str | None = None,
    scatter_filename: str = 'c2s_scatter.png',
    mapping_path: str | None = None,
) -> dict | None:
    """Run the single-stage clustering c2s pipeline on one gating step."""
    parent_mask = dataset.get_parent_mask(step)
    parent_idx = np.where(parent_mask)[0]
    if len(parent_idx) == 0:
        return None

    x_all, y_all = dataset.get_xy_vals(step)
    x_vals = x_all[parent_idx]
    y_vals = y_all[parent_idx]

    # Slice parent protein without materialising the full (n_full, n_prot)
    # matrix first.
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

    # Cluster (x, y) plane: fit on subsample, 1-NN propagate to full parent.
    # Exception: flowsom skips subsampling — SOM training cost is fixed in
    # num_iteration regardless of N, and using the SOM codebook as the
    # reference set (canonical) is more accurate than 1-NN-against-subsample.
    xy_full = np.column_stack([x_vals, y_vals]).astype(np.float32)
    rng = np.random.default_rng(base_seed)
    if cluster_method == 'flowsom':
        n_fit = n_parent
        fit_idx = np.arange(n_parent)
    else:
        n_fit = int(min(subsample_n, n_parent))
        fit_idx = (rng.choice(n_parent, size=n_fit, replace=False)
                   if n_parent > n_fit else np.arange(n_parent))
    k_eff = int(min(k, n_fit))
    if k_eff < 1:
        return None
    cluster_labels = _cluster_xy(
        xy_full, fit_idx, cluster_method, k_eff, base_seed,
        flowsom_grid=flowsom_grid, flowsom_k=flowsom_k,
    )

    # HVP column lookup hoisted out of the per-cluster loop.
    name_to_idx = {m: i for i, m in enumerate(protein_names)}
    hvp_cols_list = [(m, name_to_idx[m]) for m in top_hvp
                     if m in name_to_idx]
    hvp_names = [m for m, _ in hvp_cols_list]
    hvp_cols = np.fromiter(
        (i for _, i in hvp_cols_list),
        dtype=np.int64, count=len(hvp_cols_list),
    )

    # Remap raw cluster ids → contiguous slot ids (0..K-1), parent-order
    # preserved. Slot 1 (= cluster_to_slot[cluster_labels[...]] == 0)
    # corresponds to the first unique cluster id seen.
    unique_clusters = np.unique(cluster_labels)
    n_clusters_actual = int(len(unique_clusters))
    cluster_to_slot = np.full(int(unique_clusters.max()) + 1, -1, dtype=np.int32)
    for slot_i, cid in enumerate(unique_clusters):
        cluster_to_slot[int(cid)] = slot_i
    slot_labels = cluster_to_slot[cluster_labels]  # (n_parent,), 0..K-1

    cells: dict[str, dict] = {}
    gt_answers: dict[str, str] = {}
    rep_parent_local = np.full(n_clusters_actual, -1, dtype=np.int32)

    for slot_i in range(n_clusters_actual):
        mask = slot_labels == slot_i
        locals_in_mask = np.where(mask)[0]
        cluster_size = int(locals_in_mask.size)
        if cluster_size == 0:
            continue

        # Representative parent cell: closest to the cluster's protein
        # centroid (Euclidean, full protein). Falls back to closest to
        # (x, y) centroid if there are no protein columns.
        if parent_protein.size:
            prot_in = parent_protein[mask]
            prot_centroid = prot_in.mean(axis=0)
            d2 = np.einsum(
                'ij,ij->i', prot_in - prot_centroid, prot_in - prot_centroid,
            )
            rep_local = int(locals_in_mask[int(np.argmin(d2))])
        else:
            xy_in = xy_full[mask]
            xy_c = xy_in.mean(axis=0)
            d2 = np.einsum('ij,ij->i', xy_in - xy_c, xy_in - xy_c)
            rep_local = int(locals_in_mask[int(np.argmin(d2))])
        rep_parent_local[slot_i] = rep_local

        rep_x = float(x_vals[rep_local])
        rep_y = float(y_vals[rep_local])

        # Cell sentence from the rep cell's HVP profile (high → low).
        if hvp_cols.size and parent_protein.size:
            rep_vals = parent_protein[rep_local, hvp_cols]
            order = np.argsort(-rep_vals, kind='stable')
            sentence = ' > '.join(
                f'{hvp_names[j]}({float(rep_vals[j]):.2f})' for j in order
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

        cells[str(slot_i + 1)] = dict(
            cluster_id=int(slot_i),
            n_cells=cluster_size,
            x=rep_x,
            y=rep_y,
            cell_index=rep_sample_i,
            parquet_row=rep_parquet,
            cell_sentence=sentence,
        )
        gt_answers[str(rep_sample_i)] = gt_answer

    # Canonical slot-id reordering on the (x, y) gating plane:
    # top-left (low x, high y) → bottom-right (high x, low y). Makes IDs
    # invariant to BLAS/thread-driven permutations of the label space.
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
        score = xyn[:, 0] - xyn[:, 1]  # low = top-left, high = bottom-right
        # lexsort: primary = score, tie-break = x_norm, then -y_norm
        new_order = np.lexsort((-xyn[:, 1], xyn[:, 0], score))
        old_to_new = np.empty(n_final, dtype=np.int32)
        old_to_new[new_order] = np.arange(n_final, dtype=np.int32)

        # Remap parent-level slot labels (via original cluster id → new id).
        remap = np.full(int(unique_clusters.max()) + 1, -1, dtype=np.int32)
        for old_slot_i, cid in enumerate(unique_clusters):
            remap[int(cid)] = old_to_new[old_slot_i]
        slot_labels = remap[cluster_labels]
        rep_parent_local = rep_parent_local[new_order]

        new_cells: dict[str, dict] = {}
        for new_id, old_slot in enumerate(new_order):
            entry = dict(cells[str(int(old_slot) + 1)])
            entry['cluster_id'] = int(new_id)
            new_cells[str(new_id + 1)] = entry
        cells = new_cells

    result = dict(
        step=step['step'],
        x_marker=xm_disp,
        y_marker=ym_disp,
        parent=step.get('parent', 'ALL'),
        note=step.get('note', ''),
        tips=step.get('tips', ''),
        options=_add_unassigned(step.get('annotation categories', [])),
        n_parent_cells=n_parent,
        top_hvp_markers=top_hvp,
        n_clusters=int(len(cells)),
        cluster_method=cluster_method,
        k=int(k),
        k_fit=int(k_eff),
        subsample_n=int(n_fit),
        base_seed=int(base_seed),
        modality=modality,
        cofactor=cofactor,
        cells=cells,
        gt_answers=gt_answers,
    )
    if cluster_method == 'flowsom':
        result['flowsom_grid'] = int(flowsom_grid)
        result['flowsom_k'] = int(flowsom_k)

    # Persist mapping. ``cluster_labels`` is the load-bearing field:
    # for every parent cell (in parquet-row order aligned with
    # ``parent_indices``) it records the cluster slot (0..K-1).
    # Downstream propagation maps slot → LLM annotation directly,
    # without a 1-NN search.
    if mapping_path:
        np.savez(
            mapping_path,
            parent_indices=parquet_rows_all.astype(np.int32),
            cluster_labels=slot_labels.astype(np.int32),
            rep_parent_local=rep_parent_local.astype(np.int32),
        )

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            gt_parent = (gt_labels[parent_idx] if gt_labels is not None
                         else np.array(['Unassigned'] * n_parent))
            plot_cluster_cells(
                x_vals, y_vals, gt_parent,
                categories=result['options'],
                cluster_labels=slot_labels,
                rep_parent_local=rep_parent_local,
                x_marker=xm_disp, y_marker=ym_disp,
                step_num=step['step'],
                cluster_method=cluster_method,
                n_clusters=n_clusters_actual,
                output_path=str(out_dir / scatter_filename),
            )
        except Exception as e:  # noqa: BLE001
            print(f'  [WARN] scatter plot failed: {e}')

    return result


# ── CLI / orchestration ─────────────────────────────────────────────────────

def _output_filenames(perturb_name: str | None) -> dict[str, str]:
    if not perturb_name:
        return {
            'json': 'c2s.json',
            'scatter': 'c2s_scatter.png',
            'mapping': 'cell2cluster.npz',
        }
    return {
        'json': f'c2s__{perturb_name}.json',
        'scatter': f'c2s__{perturb_name}_scatter.png',
        'mapping': f'cell2cluster__{perturb_name}.npz',
    }


@dataclass(frozen=True)
class _WorkerConfig:
    """Serializable config passed to every worker process."""
    cluster_method: str
    k: int
    subsample_n: int
    top_hvp_n: int | None
    base_seed: int
    flowsom_grid: int
    flowsom_k: int
    modality: str
    no_plots: bool
    perturbation_name: str | None
    filenames: dict[str, str]
    skip_existing: bool = False


def _process_one_task(ti: TaskInput, cfg: _WorkerConfig) -> tuple[str, int | None]:
    """Worker: process one step, write artifacts, return (label, n_clusters|None)."""
    dataset = _TaskInputDataset(ti)
    step = _step_from_task(ti)
    step_dir = ti._step_dir
    label = f'{ti.dataset}/{ti.sample}/{ti.step_dir_name}'

    if cfg.skip_existing and (step_dir / cfg.filenames['json']).exists():
        return f'{label} [skip-existing]', -1

    result = generate_c2s(
        dataset, step,
        cluster_method=cfg.cluster_method,
        k=cfg.k,
        subsample_n=cfg.subsample_n,
        top_hvp_n=cfg.top_hvp_n,
        base_seed=cfg.base_seed,
        flowsom_grid=cfg.flowsom_grid,
        flowsom_k=cfg.flowsom_k,
        modality=cfg.modality,
        cofactor=ti.cofactor,
        output_dir=None if cfg.no_plots else str(step_dir),
        scatter_filename=cfg.filenames['scatter'],
        mapping_path=str(step_dir / cfg.filenames['mapping']),
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
        cluster_method=args.cluster_method,
        k=args.k,
        subsample_n=args.subsample_n,
        top_hvp_n=args.top_hvp_n,
        base_seed=args.base_seed,
        flowsom_grid=args.flowsom_grid,
        flowsom_k=args.flowsom_k,
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
            print(f'  [ok]   {label}: {n_clusters} clusters')

    if n_workers == 1:
        for ti in tasks:
            _report(*worker(ti))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for label, n_clusters in ex.map(worker, tasks, chunksize=1):
                _report(label, n_clusters)

    suffix = (f' (perturbation: {loader.perturbation_name})'
              if loader.perturbation_name else '')
    print(f'\n[DONE] {total} c2s items written{suffix}; '
          f'{skipped} skipped; n_workers={n_workers}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate Cell Sentences via single-stage clustering '
                    'on the gating (x, y) plane.',
    )
    BenchmarkLoader.add_cli_args(parser)
    parser.add_argument('--cluster-method', choices=CLUSTER_METHODS,
                        default=DEFAULT_CLUSTER_METHOD,
                        help='Clusterer on the (x, y) plane. '
                             f'Default: {DEFAULT_CLUSTER_METHOD}.')
    parser.add_argument('--k', type=int, default=DEFAULT_K,
                        help='Target cluster count for mbkm/agglo/gmm. '
                             'Ignored for flowsom (use --flowsom-k). '
                             f'Default: {DEFAULT_K}.')
    parser.add_argument('--subsample-n', type=int, default=DEFAULT_SUBSAMPLE_N,
                        help='Fit subsample size. The clusterer fits on this '
                             'many cells; labels propagate to the full parent '
                             f'via 1-NN. Default: {DEFAULT_SUBSAMPLE_N}.')
    parser.add_argument('--top-hvp-n', type=int, default=DEFAULT_TOP_HVP_N,
                        help='Truncate HVP ranking / cell sentence to first N. '
                             'Default: None (all proteins, high→low).')
    parser.add_argument('--base-seed', type=int, default=DEFAULT_BASE_SEED,
                        help='Seed for the fit subsample + random '
                             'initialisations.')
    parser.add_argument('--flowsom-grid', type=int, default=DEFAULT_FLOWSOM_GRID,
                        help='SOM grid size for flowsom — total SOM neurons '
                             f'= grid². Default: {DEFAULT_FLOWSOM_GRID}.')
    parser.add_argument('--flowsom-k', type=int, default=DEFAULT_FLOWSOM_K,
                        help='FlowSOM consensus metacluster count (final K). '
                             f'Default: {DEFAULT_FLOWSOM_K}.')
    parser.add_argument('--modality', choices=['cytof', 'flow'], default='cytof',
                        help='Recorded in c2s.json (downstream variants may key off it).')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip c2s_scatter.png generation.')
    parser.add_argument('--n-workers', type=int,
                        default=4,
                        help='Parallel worker processes. Default: 4')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip steps whose c2s.json already exists.')
    args = parser.parse_args()
    run_bench(args)


if __name__ == '__main__':
    main()
