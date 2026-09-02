"""PCA visualization of flat-annotation predictions.

Renders a side-by-side PCA scatter:

  Panel A — cells colored by GT primary leaf
  Panel B — cells colored by predicted primary leaf
  Panel C (optional) — cells colored by FlowSom cluster id
                       (when ``--cell2cluster`` is provided)

Method-agnostic: any ``predicted_steps.parquet`` (from
``src/flowsom_c2s/`` or ``src/llm_gate_flat/``) works as input.

PCA is the embedding for now — UMAP / t-SNE would be more informative but
their PyPI deps (numba/llvmlite) fail to build in many environments. PCA
keeps this self-contained on top of the existing scipy / sklearn install.

Usage:
    python -m src.flowsom_c2s.plot \\
        --predicted-steps results_flat/<method>/<plan_slug>/<ds>/<sample>/predicted_steps.parquet \\
        --benchmark-flat  benchmark_flat/ \\
        --gating-plan     benchmark_flat/<ds>/gating_plan*.json \\
        --data-dir        data/ \\
        --output          <method>_<plan_slug>_<sample>_pca.png \\
        [--cell2cluster   benchmark_flat/<ds>/<sample>/cell2cluster.npz]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from src.preprocess import load_config, load_sample  # noqa: E402
from src.flat_eval import derive_leaf_set, DISCARD_LABEL  # noqa: E402


def _primary_leaf(df: pd.DataFrame, step_cols: list[str], leaf_set: set[str]) -> np.ndarray:
    """Per-cell deepest-leaf-in-path or DISCARD_LABEL when no leaf hit."""
    n = len(df)
    primary = np.array([DISCARD_LABEL] * n, dtype=object)
    for col in step_cols:
        if col not in df.columns:
            continue
        vals = df[col].astype(object).values
        for i, v in enumerate(vals):
            if isinstance(v, str) and v in leaf_set:
                primary[i] = v
    return primary


def _select_protein_channels(meta_channels: dict, expr_cols: list[str]) -> list[str]:
    out: list[str] = []
    for ch in expr_cols:
        info = meta_channels.get(ch, [])
        kind = info[1] if len(info) > 1 else None
        if kind is None or kind == "protein":
            out.append(ch)
    return out or list(expr_cols)


def _stable_palette(labels: list[str], seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Deterministic color per label. ``Discard`` -> mid-gray."""
    rng = np.random.default_rng(seed)
    palette: dict[str, tuple[float, float, float]] = {DISCARD_LABEL: (0.6, 0.6, 0.6)}
    rest = [l for l in labels if l != DISCARD_LABEL]
    base_cm = plt.colormaps.get("tab20")
    for i, l in enumerate(rest):
        if i < 20:
            palette[l] = base_cm(i % 20)[:3]
        else:
            palette[l] = tuple(rng.uniform(0.15, 0.95, size=3).tolist())
    return palette


def _scatter_panel(ax, xy: np.ndarray, labels: np.ndarray, palette: dict, title: str) -> None:
    # Plot Discard first (background), then named leaves on top.
    discard_mask = labels == DISCARD_LABEL
    if discard_mask.any():
        ax.scatter(
            xy[discard_mask, 0], xy[discard_mask, 1],
            s=1.4, c=[palette[DISCARD_LABEL]], alpha=0.18, linewidths=0,
        )
    classes = sorted(set(labels.tolist()) - {DISCARD_LABEL})
    for cls in classes:
        m = labels == cls
        if m.any():
            ax.scatter(
                xy[m, 0], xy[m, 1],
                s=1.6, c=[palette[cls]], alpha=0.55, linewidths=0,
                label=cls,
            )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if len(classes) <= 24:
        ax.legend(
            loc="center left", bbox_to_anchor=(1.0, 0.5),
            fontsize=7, markerscale=2.0, frameon=False,
        )


def _scatter_clusters(ax, xy: np.ndarray, cluster_ids: np.ndarray, title: str) -> None:
    K = int(cluster_ids.max()) + 1
    rng = np.random.default_rng(0)
    colors = rng.uniform(0.15, 0.95, size=(K, 3))
    ax.scatter(xy[:, 0], xy[:, 1], s=1.4, c=colors[cluster_ids], alpha=0.55, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")


def render(
    predicted_steps_path: Path,
    sample_dir: Path,
    gating_plan_path: Path,
    data_dir: Path,
    output: Path,
    cell2cluster_path: Path | None = None,
    n_subsample: int | None = 50_000,
    seed: int = 42,
    title_suffix: str = "",
) -> None:
    with open(gating_plan_path) as f:
        gating_plan = json.load(f)
    leaf_set = derive_leaf_set(gating_plan)
    step_cols = [s["annotation column name"] for s in gating_plan["steps"]]

    # task.json -> figure out dataset/sample for raw parquet
    task_path = next((sample_dir.glob("*/task.json")), None)
    if task_path is None:
        raise SystemExit(f"[ERROR] no task.json found under {sample_dir}")
    with open(task_path) as f:
        task = json.load(f)
    dataset = task["dataset"]
    sample = task["sample"]

    config, _, meta_channels = load_config(data_dir / dataset)
    df, expr_cols, _ = load_sample(
        data_dir / dataset / "parquet" / f"{sample}.parquet",
        meta_channels, config.get("category_map", {}),
    )
    protein_cols = _select_protein_channels(meta_channels, expr_cols)

    pred_df = pd.read_parquet(predicted_steps_path)
    n = min(len(df), len(pred_df))
    df = df.iloc[:n].reset_index(drop=True)
    pred_df = pred_df.iloc[:n].reset_index(drop=True)

    # Subsample
    rng = np.random.default_rng(seed)
    if n_subsample is not None and n > n_subsample:
        idx = rng.choice(n, size=n_subsample, replace=False)
        idx.sort()
    else:
        idx = np.arange(n)

    X = df[protein_cols].to_numpy(dtype=np.float32, copy=False)[idx]
    gt_primary = _primary_leaf(df.iloc[idx].reset_index(drop=True), step_cols, leaf_set)
    pred_primary = _primary_leaf(pred_df.iloc[idx].reset_index(drop=True), step_cols, leaf_set)

    cluster_ids = None
    if cell2cluster_path is not None and cell2cluster_path.exists():
        npz = np.load(cell2cluster_path)
        cluster_ids = npz["cluster_labels"][idx].astype(int)

    print(f"PCA on {len(idx):,} cells × {len(protein_cols)} protein channels ...")
    pca = PCA(n_components=2, random_state=seed)
    XY = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print(f"  PC1 var={var[0]*100:.1f}%, PC2 var={var[1]*100:.1f}%")

    palette = _stable_palette(sorted(leaf_set), seed=seed)

    n_panels = 3 if cluster_ids is not None else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7.5 * n_panels, 7.0), dpi=120,
    )
    if n_panels == 2:
        ax_gt, ax_pred = axes
    else:
        ax_gt, ax_pred, ax_cluster = axes

    suffix = f" — {title_suffix}" if title_suffix else ""
    _scatter_panel(ax_gt, XY, gt_primary, palette, title=f"GT primary leaf{suffix}")
    _scatter_panel(ax_pred, XY, pred_primary, palette, title=f"Pred primary leaf{suffix}")
    if cluster_ids is not None:
        _scatter_clusters(ax_cluster, XY, cluster_ids,
                          title=f"FlowSom cluster id (K={int(cluster_ids.max()) + 1})")

    fig.suptitle(
        f"{dataset} / {sample}  |  n_subsample={len(idx):,} / total={n:,}  "
        f"|  PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA scatter (GT vs Pred primary leaf).")
    parser.add_argument("--predicted-steps", type=Path, required=True,
                        help="Path to predicted_steps.parquet")
    parser.add_argument("--benchmark-flat", type=Path, required=True,
                        help="Path to benchmark_flat/<dataset>/<sample>/ "
                             "(provides task.json + optional cell2cluster.npz)")
    parser.add_argument("--gating-plan", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell2cluster", type=Path, default=None,
                        help="(optional) cell2cluster.npz for the 3rd cluster panel; "
                             "auto-found at <benchmark-flat>/cell2cluster.npz if omitted")
    parser.add_argument("--n-subsample", type=int, default=50_000,
                        help="Subsample to N cells for PCA + plotting (default 50k; "
                             "use 0 to disable)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()

    cell2cluster = args.cell2cluster
    if cell2cluster is None:
        cand = args.benchmark_flat / "cell2cluster.npz"
        if cand.exists():
            cell2cluster = cand

    render(
        predicted_steps_path=args.predicted_steps,
        sample_dir=args.benchmark_flat,
        gating_plan_path=args.gating_plan,
        data_dir=args.data_dir,
        output=args.output,
        cell2cluster_path=cell2cluster,
        n_subsample=args.n_subsample if args.n_subsample > 0 else None,
        seed=args.seed,
        title_suffix=args.title_suffix,
    )


if __name__ == "__main__":
    main()
