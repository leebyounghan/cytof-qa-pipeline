"""flowsom_c2s clustering: per-sample FlowSom + cell-sentence emission.

Reads ``benchmark_flat/{ds}/{sample}/task.json`` (created by
``flowsom_c2s.preprocess``) plus the parquet sample, runs FlowSom
(SOM grid + consensus hierarchical metaclustering) over the full sample on
all protein channels, and writes:

  - ``c2s.json``         — per-cluster centroid + cell-sentence + diagnostics
  - ``cell2cluster.npz`` — parquet_indices (parquet row order),
                           cluster_labels (0..K_meta-1),
                           rep_parquet_row (K_meta,)
  - ``c2s_scatter.png``  — diagnostic: cells colored by meta-cluster on the
                           Step 1 axes (best-effort; skipped if axes missing)

Usage:
    python -m src.flowsom_c2s.c2s --benchmark benchmark_flat/ --data-dir data/ \
        --datasets Lyoplate_tcell --n-meta-clusters 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import load_config, load_sample, resolve_channel
from src.flowsom_c2s.utils import compute_top_hvp


# ── FlowSom backend (MiniSom + sklearn) ─────────────────────────────────────
#
# Ported from ``src/llm/c2s.py`` (the step-level C2S pipeline) so the whole
# codebase shares one canonical FlowSom implementation:
#   1. Train a SOM grid on the protein expression (random init, fixed iters).
#   2. Metacluster the SOM codebook into K clusters with average-linkage
#      agglomerative clustering — canonical FlowSom uses average-linkage
#      consensus hierarchical clustering (not Ward).
#   3. Assign each cell to its nearest codebook node (1-NN via sklearn
#      NearestNeighbors), then to that node's meta-cluster.
#
# This replaces the previous Ward-linkage + manual-argmin backend, which
# also had a memory bug — it materialised the
# full ``(N, M, F)`` BMU-distance tensor before the chunking guard.
#
# We use ``minisom`` instead of the saeyslab ``flowsom`` package because the
# latter pulls in ``numba``/``llvmlite``, which fails to build from source on
# common macOS + Python 3.11 setups.

def _flowsom_cluster(
    X: np.ndarray,
    cols: list[str],
    n_meta_clusters: int,
    xdim: int = 10,
    ydim: int = 10,
    seed: int = 42,
    som_iters: int = 30_000,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run FlowSom-style clustering on (n_cells, n_markers) ``X``.

    Backend ported from ``src/llm/c2s.py``: SOM grid → average-linkage
    codebook metaclustering → 1-NN cell assignment against the codebook.

    Returns:
        cluster_labels: (n_cells,) int — meta-cluster id 0..K-1
        centroids:      (K, n_markers) — meta-cluster centroids in marker space
    """
    try:
        from minisom import MiniSom
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise ImportError(
            "flowsom_c2s requires `minisom` and `scikit-learn`. "
            "Install via `uv pip install minisom scikit-learn`.\n"
            f"Original error: {e}"
        )

    n_features = X.shape[1]
    Xf = np.ascontiguousarray(X, dtype=np.float32)

    # 1) SOM grid on the full sample (random init). We use ``train_random``
    #    (vs src/llm/c2s.py's ``som.train``) so the SOM samples cells uniformly
    #    across the whole sample — minisom's ``train`` would only visit the
    #    first ``som_iters`` rows when ``som_iters << n_cells``, biasing the
    #    whole-cell SOM by parquet row order. 30k iters (vs llm's 10k) lets the
    #    higher-dim whole-cell SOM converge — empirically recovers cluster
    #    purity on Acute2020.
    som = MiniSom(
        x=xdim, y=ydim, input_len=n_features,
        sigma=sigma, learning_rate=learning_rate,
        random_seed=seed,
    )
    som.random_weights_init(Xf)
    som.train_random(Xf, som_iters)
    codebook = som.get_weights().reshape(xdim * ydim, n_features)  # (M, F)

    # 2) Metacluster the codebook into K clusters (average linkage — canonical
    #    FlowSom). If K >= n_nodes, every node is its own meta-cluster.
    n_nodes = codebook.shape[0]
    K = max(1, min(int(n_meta_clusters), n_nodes))
    if K >= n_nodes:
        meta_of_node = np.arange(n_nodes, dtype=np.int64)
    else:
        meta_of_node = AgglomerativeClustering(
            n_clusters=K, linkage="average",
        ).fit_predict(codebook).astype(np.int64)

    # 3) Each cell -> nearest codebook node (1-NN) -> that node's meta-cluster.
    nn = NearestNeighbors(n_neighbors=1).fit(codebook)
    node_ids = nn.kneighbors(Xf, return_distance=False)[:, 0]
    meta = meta_of_node[node_ids].astype(int)

    # Re-pack to dense 0..K_actual-1.
    used = np.unique(meta)
    if used.size != used.max() + 1:
        remap = -np.ones(used.max() + 1, dtype=int)
        remap[used] = np.arange(used.size)
        meta = remap[meta]
    K_actual = int(meta.max()) + 1

    centroids = np.zeros((K_actual, n_features), dtype=np.float32)
    for k in range(K_actual):
        sel = meta == k
        if sel.any():
            centroids[k] = X[sel].mean(axis=0)
    return meta, centroids


# ── Per-sample driver ───────────────────────────────────────────────────────

def _select_protein_channels(meta_channels: dict, expr_cols: list[str]) -> list[str]:
    """Filter expr_cols to channels marked ``protein`` in meta.json (if kind
    info is present); otherwise keep all expr_cols.
    """
    out: list[str] = []
    for ch in expr_cols:
        info = meta_channels.get(ch, [])
        kind = info[1] if len(info) > 1 else None
        if kind is None or kind == "protein":
            out.append(ch)
    return out or list(expr_cols)


def _render_cell_sentence(
    centroid: np.ndarray,
    markers: list[str],
    hvp_order: list[str],
    top_n: int | None,
) -> str:
    """Render a >-separated cell sentence: HVP order, '<marker>(<val>)'."""
    name_to_val = {m: float(centroid[i]) for i, m in enumerate(markers)}
    ordered = [m for m in hvp_order if m in name_to_val]
    if top_n is not None:
        ordered = ordered[:top_n]
    return " > ".join(f"{m}({name_to_val[m]:.2f})" for m in ordered)


def _majority_label(labels: np.ndarray) -> str | None:
    """Return the most frequent non-null label, or None if all null."""
    if len(labels) == 0:
        return None
    s = pd.Series(labels).dropna()
    if s.empty:
        return None
    return str(s.mode().iloc[0])


def _find_any_task_json(sample_dir: Path) -> Path | None:
    """Return any ``<sample>/<plan_slug>/task.json`` found under sample_dir.

    The c2s clustering itself is plan-independent (depends only on the
    sample's protein channels) — we just need a task.json to read modality /
    cofactor / step columns for the diagnostic. Any plan_slug works.
    """
    for sub in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
        cand = sub / "task.json"
        if cand.exists():
            return cand
    return None


def process_sample(
    sample_path: Path,
    sample_dir: Path,
    config: dict,
    meta_channels: dict,
    n_meta_clusters: int,
    seed: int,
    xdim: int,
    ydim: int,
    plot: bool = True,
) -> None:
    """Cluster one sample, write c2s.json + cell2cluster.npz (+ scatter)."""
    task_path = _find_any_task_json(sample_dir)
    if task_path is None:
        print(f"[SKIP] no task.json under any plan_slug in {sample_dir} (run preprocess first)")
        return
    with open(task_path) as f:
        task = json.load(f)

    df, expr_cols, _ = load_sample(sample_path, meta_channels, config.get("category_map", {}))
    protein_cols = _select_protein_channels(meta_channels, expr_cols)
    if not protein_cols:
        print(f"[SKIP] no protein channels for {sample_dir.name}")
        return

    channel_marker_map = config.get("channel_marker_map", {})
    display_names = [channel_marker_map.get(c, c) for c in protein_cols]
    if len(set(display_names)) != len(display_names):
        # Channel-marker collision — keep channel keys for unambiguous lookup.
        display_names = list(protein_cols)

    X = df[protein_cols].to_numpy(dtype=np.float32, copy=False)

    print(f"  {sample_dir.name}: n_cells={X.shape[0]:,}, n_proteins={X.shape[1]}, "
          f"running FlowSom (xdim={xdim}, ydim={ydim}, K_meta={n_meta_clusters})...")
    meta_labels, centroids = _flowsom_cluster(
        X, protein_cols, n_meta_clusters=n_meta_clusters,
        xdim=xdim, ydim=ydim, seed=seed,
    )
    K = centroids.shape[0]

    # Pick representative cell = 1-NN to centroid in protein space (within cluster)
    rep_parquet_rows = np.full(K, -1, dtype=np.int64)
    for k in range(K):
        sel = np.where(meta_labels == k)[0]
        if sel.size == 0:
            continue
        d = np.linalg.norm(X[sel] - centroids[k], axis=1)
        rep_parquet_rows[k] = int(sel[int(np.argmin(d))])

    # Global HVP ranking on the full sample (over chosen markers, in display-name space)
    hvp_order = compute_top_hvp(X, display_names)

    # Diagnostic: cluster-majority GT per step
    step_cols = [s["annotation_column"] for s in task["steps"]]
    gt_per_step: dict[str, dict[str, str | None]] = {}
    for col in step_cols:
        if col not in df.columns:
            continue
        gt_col = df[col].values
        per_cluster: dict[str, str | None] = {}
        for k in range(K):
            sel = meta_labels == k
            per_cluster[str(k + 1)] = _majority_label(gt_col[sel]) if sel.any() else None
        gt_per_step[col] = per_cluster

    # Build c2s.json
    n_cells = int(X.shape[0])
    clusters_payload: dict[str, dict] = {}
    for k in range(K):
        sel = meta_labels == k
        n_k = int(sel.sum())
        if n_k == 0:
            continue
        rep_row = int(rep_parquet_rows[k])
        centroid_dict = {display_names[i]: round(float(centroids[k][i]), 4) for i in range(len(display_names))}
        sentence = _render_cell_sentence(centroids[k], display_names, hvp_order, top_n=None)
        clusters_payload[str(k + 1)] = {
            "cluster_id": k,
            "n_cells": n_k,
            "fraction": round(n_k / n_cells, 6),
            "parquet_row": rep_row,
            "centroid": centroid_dict,
            "cell_sentence": sentence,
        }

    c2s = {
        "n_cells": n_cells,
        "n_clusters": int(K),
        "modality": task.get("modality"),
        "cofactor": task.get("cofactor"),
        "all_protein_markers": display_names,
        "top_hvp_markers": hvp_order,
        "clusters": clusters_payload,
        "gt_per_step": gt_per_step,
    }
    with open(sample_dir / "c2s.json", "w") as f:
        json.dump(c2s, f, indent=2, ensure_ascii=False)

    np.savez(
        sample_dir / "cell2cluster.npz",
        parquet_indices=np.arange(n_cells, dtype=np.int64),
        cluster_labels=meta_labels.astype(np.int32),
        rep_parquet_row=rep_parquet_rows.astype(np.int64),
    )

    if plot:
        _maybe_render_scatter(df, task, meta_labels, sample_dir / "c2s_scatter.png", expr_cols, channel_marker_map)

    print(f"    -> K={K}, rep cells written, c2s.json saved")


def _maybe_render_scatter(
    df: pd.DataFrame,
    task: dict,
    meta_labels: np.ndarray,
    out_path: Path,
    expr_cols: list[str],
    channel_marker_map: dict,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps = task.get("steps") or []
    if not steps:
        return
    s0 = steps[0]
    x_ch = resolve_channel(s0["x_marker"], expr_cols, channel_marker_map)
    y_ch = resolve_channel(s0["y_marker"], expr_cols, channel_marker_map)
    if x_ch is None or y_ch is None:
        return
    x_vals = df[x_ch].values
    y_vals = df[y_ch].values
    K = int(meta_labels.max()) + 1
    rng = np.random.default_rng(0)
    colors = rng.random((K, 3))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_vals, y_vals, s=0.3, c=colors[meta_labels], alpha=0.4, linewidths=0)
    ax.set_xlabel(s0["x_marker"]); ax.set_ylabel(s0["y_marker"])
    ax.set_title(f"FlowSom meta-clusters (K={K}) on Step 1 axes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def process_dataset(
    dataset_dir: Path,
    benchmark_flat: Path,
    n_meta_clusters: int,
    seed: int,
    xdim: int,
    ydim: int,
    samples: list[str] | None = None,
    plot: bool = True,
) -> None:
    parquet_dir = dataset_dir / "parquet"
    if not (parquet_dir / "meta.json").exists():
        print(f"[SKIP] No meta.json in {dataset_dir}")
        return
    config, _, meta_channels = load_config(dataset_dir)
    dataset_name = config["name"]
    bench_dir = benchmark_flat / dataset_name
    if not bench_dir.is_dir():
        print(f"[SKIP] No benchmark_flat dir for {dataset_name} — run preprocess first")
        return

    sample_dirs = sorted([d for d in bench_dir.iterdir() if d.is_dir()])
    if samples:
        wanted = set(samples)
        sample_dirs = [d for d in sample_dirs if d.name in wanted]

    print(f"\n{'='*60}\nDataset: {dataset_name} ({len(sample_dirs)} samples)\n{'='*60}")
    for sd in sample_dirs:
        sample_path = parquet_dir / f"{sd.name}.parquet"
        if not sample_path.exists():
            print(f"[SKIP] no parquet for {sd.name}")
            continue
        process_sample(
            sample_path=sample_path,
            sample_dir=sd,
            config=config,
            meta_channels=meta_channels,
            n_meta_clusters=n_meta_clusters,
            seed=seed,
            xdim=xdim,
            ydim=ydim,
            plot=plot,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="flowsom_c2s clustering")
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark_flat/"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/"))
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--samples", nargs="*", help="Restrict to given sample IDs")
    parser.add_argument("--n-meta-clusters", type=int, default=40)
    parser.add_argument("--xdim", type=int, default=10)
    parser.add_argument("--ydim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    dataset_dirs = sorted(args.data_dir.iterdir())
    dataset_dirs = [d for d in dataset_dirs if d.is_dir() and (d / "parquet" / "meta.json").exists()]
    if args.datasets:
        dataset_dirs = [d for d in dataset_dirs if d.name in args.datasets]

    for dd in dataset_dirs:
        process_dataset(
            dd,
            args.benchmark,
            n_meta_clusters=args.n_meta_clusters,
            seed=args.seed,
            xdim=args.xdim,
            ydim=args.ydim,
            samples=args.samples,
            plot=not args.no_plots,
        )


if __name__ == "__main__":
    main()
