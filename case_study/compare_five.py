"""
case_study/compare_five.py — 5-way method comparison stacked across steps.

Per step row, left -> right:
  1. GT             — points colored by ground-truth category
  2. UNITO          — points colored by UNITO predicted labels
                      (results/unito/predictions_normal)
  3. cyMAE          — points colored by cyMAE predicted labels
  4. density-grid   — x/y density thresholds + points colored by
                      flowdensity-derived predicted labels (was flowdensity+gpt-5.4)
  5. clustering     — per-cluster convex hulls; cluster -> label from
                      results/c2s_gpt-5.4_hvp10 (was flowsom+gpt-5.4)
  6. gate gpt-5.4   — per-category rectangular gates from
                      results/gate_gpt-5.4_full

Each step renders as a 1x5 row; rows stack vertically via PIL.

Defaults (overridable via CLI):
  Acute2020/994570_Normalized   steps 09, 13, 18
  Bjornson/...Hu_M_Basal        steps 01, 04, 07

cyMAE and flowdensity+gpt-5.4 are user-supplied per step. Resolver looks at:
  case_study/cymae/<ds_lower>/<sample>/step_NN/prediction.json    (per-step)
  case_study/cymae/<ds_lower>/prediction.json                     (legacy)
  case_study/flowdensity-gpt5.4/<ds_lower>/<sample>/step_NN/prediction.json
  case_study/flowdensity-gpt5.4/<ds_lower>/prediction.json
Missing predictions render as placeholder text (figure still builds).

Run:
    # run from the repository root
    uv run python -m case_study.compare_five
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from src.bench import BenchmarkLoader
from src.baselines.flowsom_baseline import (
    DEFAULT_GRID, DEFAULT_K, DEFAULT_ITERATIONS,
    SUBSAMPLE_THRESHOLD, SUBSAMPLE_SIZE, SUBSAMPLE_SEED,
    _train_som, _consensus_metacluster, _assign_neurons,
)

REPO = Path(__file__).resolve().parent.parent
PAPER_FIGURES = Path(os.environ.get("PAPER_FIGURES", "case_study/figures"))

# ------ Default rows (cohort, sample, step) — overridable via CLI ----------
DEFAULT_ROWS: list[tuple[str, str, int]] = [
    ("Acute2020", "994570_Normalized", 18),
    ("Acute2020", "994570_Normalized", 29),
    ("Bjornson",  "R15W11_W070515101208_Hu_M_Basal", 6),
    ("Bjornson",  "R15W11_W070515101208_Hu_M_Basal", 7),
]

# ------ Palette ------------------------------------------------------------
PALETTE = {
    "NaiveB":           "#1f77b4",
    "IgDposMemB":       "#2ca02c",
    "IgDnegMemB":       "#d62728",
    "ClassicalMono":    "#1f77b4",
    "TransitionalMono": "#2ca02c",
    "NonclassicalMono": "#d62728",
    "Granulocyte":      "#ff7f0e",
    "Mononuclear":      "#1f77b4",
    "CD45hiCD66bpos":   "#9467bd",
    "CD66bnegCD45lo":   "#8c564b",
    "CD4":              "#1f77b4",
    "CD8":              "#d62728",
    "DPT":              "#9467bd",
    "DNT":              "#8c564b",  # brown (distinct from Unassigned grey)
    # Bjornson Basal categories
    "Bcells":           "#1f77b4",
    "Tcells":           "#d62728",
    "Monocytes":        "#ff7f0e",
    "CD11bnegCD16neg":  "#17becf",  # cyan
    "NKcells":          "#9467bd",
    "HLA-DRposCD7neg":  "#17becf",  # cyan
    "CD3negCD19neg":    "#17becf",  # cyan (override the earlier grey)
    "Ir":               "#1f77b4",
    # Unassigned is the visual background — pale grey, also dimmed in scatter
    "Unassigned":       "#dddddd",
}
FALLBACK = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def build_color_map(label_sources: list[np.ndarray]) -> dict[str, str]:
    """Single label -> color map across all panel sources for this row."""
    from collections import Counter
    counter: Counter[str] = Counter()
    for arr in label_sources:
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        uniq, counts = np.unique(arr, return_counts=True)
        for u, c in zip(uniq, counts):
            counter[str(u)] += int(c)
    cmap: dict[str, str] = {}
    used_colors: set[str] = set()
    for lab, _ in counter.most_common():
        if lab in PALETTE:
            col = PALETTE[lab]
            cmap[lab] = col
            used_colors.add(col)
    fallback_iter = iter(FALLBACK)
    for lab, _ in counter.most_common():
        if lab in cmap:
            continue
        col = next(fallback_iter, None)
        while col is not None and col in used_colors:
            col = next(fallback_iter, None)
        if col is None:
            col = "#333333"
        cmap[lab] = col
        used_colors.add(col)
    return cmap


# ------ Per-panel renderers -------------------------------------------------
def scatter_by_label(ax, x, y, labels, color_map, alpha=0.5, s=4):
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts)]
    # Draw Unassigned first (under-layer), real categories on top
    order = sorted(order, key=lambda c: (str(c) != "Unassigned", -int((labels == c).sum())))
    for cat in order:
        m = labels == cat
        col = color_map.get(str(cat), "#333333")
        if str(cat) == "Unassigned":
            ax.scatter(x[m], y[m], s=max(s - 2, 2), alpha=max(alpha - 0.25, 0.15),
                       color=col, edgecolors="none",
                       label=f"{cat} (n={int(m.sum())})")
        else:
            ax.scatter(x[m], y[m], s=s, alpha=alpha, color=col,
                       edgecolors="none", label=f"{cat} (n={int(m.sum())})")


def run_flowsom_2d(X_2d: np.ndarray) -> np.ndarray:
    n_total = len(X_2d)
    if n_total < 2:
        return np.zeros(n_total, dtype=np.int32)
    if n_total > SUBSAMPLE_THRESHOLD:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        fit_idx = rng.choice(n_total, SUBSAMPLE_SIZE, replace=False)
        X_fit = X_2d[fit_idx]
    else:
        X_fit = X_2d
    som = _train_som(X_fit, grid=DEFAULT_GRID, seed=SUBSAMPLE_SEED,
                     n_iterations=DEFAULT_ITERATIONS)
    codebook = som.get_weights().reshape(-1, X_fit.shape[1])
    meta_of_node = _consensus_metacluster(codebook, k=DEFAULT_K)
    neuron_ids = _assign_neurons(som, X_2d, grid=DEFAULT_GRID)
    return meta_of_node[neuron_ids].astype(np.int32)


def cluster_majority(cluster_ids: np.ndarray,
                     labels: np.ndarray) -> dict[int, str]:
    out: dict[int, str] = {}
    for cid in np.unique(cluster_ids):
        m = cluster_ids == cid
        if not m.any():
            continue
        uniq, counts = np.unique(labels[m], return_counts=True)
        out[int(cid)] = str(uniq[np.argmax(counts)])
    return out


def plot_gt(ax, ti, color_map):
    scatter_by_label(ax, ti.X_2d[:, 0], ti.X_2d[:, 1], ti.y, color_map)
    ax.set_title("GT")


def plot_predicted(ax, ti, predicted, title, color_map):
    scatter_by_label(ax, ti.X_2d[:, 0], ti.X_2d[:, 1], predicted, color_map)
    ax.set_title(title)


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def plot_unito_segmask(ax, ti, predicted, title, color_map,
                       grid_res: int = 220, k_neighbors: int = 7,
                       mask_alpha: float = 0.22):
    """UNITO panel: KNN-derived decision region on a 2D grid as faint
    background (proxy for UNITO's per-pixel segmentation), with predicted
    scatter overlaid."""
    from sklearn.neighbors import KNeighborsClassifier
    x, y = ti.X_2d[:, 0], ti.X_2d[:, 1]
    pred = np.asarray(predicted)
    # Train KNN on cell positions -> predicted label
    knn = KNeighborsClassifier(n_neighbors=min(k_neighbors, len(x)),
                               weights="distance", n_jobs=-1)
    knn.fit(np.column_stack([x, y]), pred)
    # Build grid (small margin)
    pad_x = 0.04 * (x.max() - x.min() + 1e-6)
    pad_y = 0.04 * (y.max() - y.min() + 1e-6)
    gx = np.linspace(x.min() - pad_x, x.max() + pad_x, grid_res)
    gy = np.linspace(y.min() - pad_y, y.max() + pad_y, grid_res)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    grid_labels = knn.predict(grid_pts).reshape(grid_res, grid_res)
    # Render mask as RGBA
    rgba = np.ones((grid_res, grid_res, 4), dtype=np.float32)
    rgba[..., 3] = 0.0  # transparent default
    uniq = np.unique(grid_labels)
    for lab in uniq:
        if str(lab) == "Unassigned":
            continue  # don't paint Unassigned region
        col = color_map.get(str(lab), "#cccccc")
        r, g, b = _hex_to_rgb01(col)
        m = grid_labels == lab
        rgba[m, 0] = r
        rgba[m, 1] = g
        rgba[m, 2] = b
        rgba[m, 3] = mask_alpha
    ax.imshow(rgba, origin="lower", aspect="auto",
              extent=(gx.min(), gx.max(), gy.min(), gy.max()),
              interpolation="nearest", zorder=0)
    # Scatter overlay
    scatter_by_label(ax, x, y, pred, color_map, alpha=0.55, s=4)
    ax.set_title(title)


def plot_flowsom_clusters(ax, ti, cluster_ids, predicted, title, color_map,
                          color_by="pred"):
    x, y = ti.X_2d[:, 0], ti.X_2d[:, 1]
    point_labels = ti.y if color_by == "gt" else np.asarray(predicted)
    scatter_by_label(ax, x, y, point_labels, color_map, alpha=0.30)
    pred_arr = np.asarray(predicted)
    cluster_to_label = cluster_majority(cluster_ids, pred_arr)
    seen: set[str] = set()
    for cid in np.unique(cluster_ids):
        m = cluster_ids == cid
        pts = ti.X_2d[m]
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        lab = cluster_to_label.get(int(cid), "Unassigned")
        col = color_map.get(lab, "#333333")
        hp = pts[hull.vertices]
        ax.fill(hp[:, 0], hp[:, 1], color=col, alpha=0.18,
                edgecolor=col, linewidth=1.4)
        if lab not in seen:
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ax.text(cx, cy, lab, fontsize=7, ha="center", color="black",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=col, alpha=0.85))
            seen.add(lab)
    ax.set_title(title)


def plot_flowdensity_gpt(ax, ti, thresholds_path: Path | None,
                         predicted, title, color_map):
    scatter_by_label(ax, ti.X_2d[:, 0], ti.X_2d[:, 1], predicted,
                     color_map, alpha=0.45)
    if thresholds_path is not None and thresholds_path.exists():
        with open(thresholds_path) as f:
            th = json.load(f)
        for tx in th.get("x_thresholds", []):
            ax.axvline(tx, color="black", linestyle="--",
                       linewidth=1.3, alpha=0.85)
        for ty in th.get("y_thresholds", []):
            ax.axhline(ty, color="black", linestyle="--",
                       linewidth=1.3, alpha=0.85)
    ax.set_title(title)


def plot_gates(ax, ti, gates: dict, predicted, color_map,
               title="LLM-Gate gpt-5.4"):
    scatter_by_label(ax, ti.X_2d[:, 0], ti.X_2d[:, 1], predicted,
                     color_map, alpha=0.45)
    for cat, g in gates.items():
        col = color_map.get(cat, "#333333")
        rect = mpatches.Rectangle(
            (g["x_min"], g["y_min"]),
            g["x_max"] - g["x_min"],
            g["y_max"] - g["y_min"],
            fill=False, edgecolor=col, linewidth=2.2,
        )
        ax.add_patch(rect)
        ax.text(g["x_min"], g["y_max"], cat, fontsize=7, color=col,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.16", fc="white",
                          ec=col, alpha=0.85))
    ax.set_title(title)


def plot_missing(ax, title, want_path: Path):
    ax.text(0.5, 0.55, "(missing prediction)", ha="center", va="center",
            fontsize=9, color="#999", transform=ax.transAxes)
    ax.text(0.5, 0.42, f"expected:\n{want_path}", ha="center", va="center",
            fontsize=6, color="#999", transform=ax.transAxes)
    ax.set_title(title)


# ------ Path resolvers -----------------------------------------------------
def _resolve_user_pred(root_dir: str, dataset: str, sample: str,
                       step: int) -> tuple[Path | None, Path]:
    """Find user-supplied prediction.json under case_study/<root_dir>.

    Returns (resolved_path_or_None, primary_candidate_path_for_messaging).
    """
    expected_id = f"{dataset}__{sample}__step_{step:02d}"
    primary = (REPO / "case_study" / root_dir / dataset.lower() / sample /
               f"step_{step:02d}" / "prediction.json")
    candidates = [
        primary,
        # per-step layout (with sample dir)
        REPO / "case_study" / root_dir / dataset / sample /
            f"step_{step:02d}" / "prediction.json",
        # per-step layout (no sample dir — flat <ds_lower>/step_NN/)
        REPO / "case_study" / root_dir / dataset.lower() /
            f"step_{step:02d}" / "prediction.json",
        REPO / "case_study" / root_dir / dataset /
            f"step_{step:02d}" / "prediction.json",
        # one-file legacy layout
        REPO / "case_study" / root_dir / dataset.lower() / "prediction.json",
        REPO / "case_study" / root_dir / dataset / "prediction.json",
        REPO / "case_study" / root_dir / "prediction.json",
    ]
    mismatched: list[tuple[Path, str]] = []
    for c in candidates:
        if not c.exists():
            continue
        try:
            tid = json.loads(c.read_text()).get("task_id")
        except Exception:
            continue
        if tid == expected_id:
            return c, primary
        mismatched.append((c, str(tid)))
    if mismatched:
        print(f"  [WARN] {root_dir}: found prediction(s) but task_id mismatch "
              f"for {expected_id}:")
        for path, tid in mismatched:
            print(f"         {path}  (task_id={tid})")
    return None, primary


def load_pred(path: Path | None) -> list[str] | None:
    if path is None or not path.exists():
        return None
    return list(json.loads(path.read_text())["predicted_labels"])


# ------ Per-row renderer ---------------------------------------------------
def render_row(loader, dataset: str, sample: str, step: int,
               dpi: int, show_titles: bool = True) -> Image.Image:
    step_dir = REPO / "benchmark" / dataset / sample / f"step_{step:02d}"
    ti = loader.load_task(step_dir / "task.json")
    if ti is None:
        raise SystemExit(f"Failed to load task: {step_dir}")
    expected_id = f"{dataset}__{sample}__step_{step:02d}"
    print(f"\n[{expected_id}] n={len(ti.expression)} "
          f"axes=({ti.x_marker_display}, {ti.y_marker_display}) "
          f"categories={ti.categories}")

    # UNITO — from results/unito/predictions_normal
    unito_path = (REPO / "results" / "unito" / "predictions_normal" /
                  dataset / sample / f"step_{step:02d}" / "prediction.json")
    unito_pred = load_pred(unito_path)

    # cyMAE — user supplied
    cymae_path, cymae_primary = _resolve_user_pred(
        "cymae", dataset, sample, step)
    cymae_pred = load_pred(cymae_path)

    # flowsom+gpt-5.4 — from results/c2s_gpt-5.4_hvp10
    fs_gpt_path = (REPO / "results" / "c2s_gpt-5.4_hvp10" / dataset /
                   sample / f"step_{step:02d}" / "prediction.json")
    fs_gpt_pred = load_pred(fs_gpt_path)

    # flowdensity+gpt-5.4 — user supplied (predicted labels)
    fd_gpt_path, fd_gpt_primary = _resolve_user_pred(
        "flowdensity-gpt5.4", dataset, sample, step)
    fd_gpt_pred = load_pred(fd_gpt_path)

    # flowdensity thresholds (geometry for drawing) — from results/flowdensity_result
    fd_thr_path = (REPO / "results" / "flowdensity_result" / dataset /
                   sample / f"step_{step:02d}" / "thresholds.json")

    # gate gpt-5.4 — from results/gate_gpt-5.4_full
    gate_path = (REPO / "results" / "gate_gpt-5.4_full" / dataset / sample /
                 f"step_{step:02d}" / "prediction.json")
    gate_pred = load_pred(gate_path)
    gate_pred_full = REPO / "results" / "gate_gpt-5.4_full" / dataset / "pred.json"
    gates = {}
    if gate_pred_full.exists():
        gf = json.loads(gate_pred_full.read_text())
        gates = (gf.get("samples", {}).get(sample, {})
                  .get("steps", {}).get(str(step), {}).get("gates", {})) or {}

    # Validate lengths for whatever we did load
    n = len(ti.expression)
    for name, pred in [("UNITO", unito_pred),
                       ("cyMAE", cymae_pred),
                       ("flowsom+gpt-5.4", fs_gpt_pred),
                       ("flowdensity+gpt-5.4", fd_gpt_pred),
                       ("LLM-Gate gpt-5.4", gate_pred)]:
        if pred is not None and len(pred) != n:
            raise SystemExit(
                f"[{expected_id}] length mismatch: {name}={len(pred)} vs n={n}")

    # Cluster boundaries for the clustering+gpt panel.
    # Prefer benchmark/<ds>/<sample>/step_NN/cell2cluster.npz which holds the
    # exact partition the c2s_hvp10 pipeline fed to the LLM (per-cell
    # cluster_id matches the LLM labels 1:1). Fall back to a fresh 2-D
    # FlowSOM re-run only when the file is missing.
    cluster_ids = None
    if fs_gpt_pred is not None:
        c2c_path = (REPO / "benchmark" / dataset / sample /
                    f"step_{step:02d}" / "cell2cluster.npz")
        if c2c_path.exists():
            try:
                npz = np.load(c2c_path, allow_pickle=True)
                cluster_ids = np.asarray(npz["cluster_labels"], dtype=np.int32)
                if len(cluster_ids) != n:
                    print(f"  [WARN] cell2cluster size {len(cluster_ids)} != "
                          f"n={n}; falling back to 2-D FlowSOM re-run")
                    cluster_ids = None
            except Exception as e:
                print(f"  [WARN] failed to read {c2c_path}: {e}")
                cluster_ids = None
        if cluster_ids is None:
            cluster_ids = run_flowsom_2d(ti.X_2d)

    # Color map shared across all panels in this row
    sources = [np.asarray(ti.y)]
    for p in [unito_pred, cymae_pred, fs_gpt_pred, fd_gpt_pred, gate_pred]:
        if p is not None:
            sources.append(np.asarray(p))
    if gates:
        sources.append(np.array(list(gates.keys()), dtype=object))
    color_map = build_color_map(sources)

    fig, axes = plt.subplots(1, 6, figsize=(21.6, 2.6),
                             sharex=True, sharey=True)
    # Methods reversed (LLM-side first, trained baselines next); GT
    # at the rightmost position as the reference column.
    if gate_pred is not None and gates:
        plot_gates(axes[0], ti, gates, gate_pred, color_map)
    else:
        plot_missing(axes[0], "LLM-Gate gpt-5.4", gate_path)
    if fs_gpt_pred is not None and cluster_ids is not None:
        plot_flowsom_clusters(axes[1], ti, cluster_ids, fs_gpt_pred,
                              "clustering + gpt-5.4", color_map, color_by="pred")
    else:
        plot_missing(axes[1], "clustering + gpt-5.4", fs_gpt_path)
    if fd_gpt_pred is not None:
        plot_flowdensity_gpt(axes[2], ti, fd_thr_path, fd_gpt_pred,
                             "density-grid + gpt-5.4", color_map)
    else:
        plot_missing(axes[2], "density-grid + gpt-5.4", fd_gpt_primary)
    if cymae_pred is not None:
        plot_predicted(axes[3], ti, cymae_pred, "cyMAE", color_map)
    else:
        plot_missing(axes[3], "cyMAE", cymae_primary)
    if unito_pred is not None:
        plot_unito_segmask(axes[4], ti, unito_pred, "UNITO", color_map)
    else:
        plot_missing(axes[4], "UNITO", unito_path)
    plot_gt(axes[5], ti, color_map)

    for ax in axes:
        ax.set_xlabel(ti.x_marker_display, fontsize=14, labelpad=2)
        ax.set_ylabel(ti.y_marker_display, fontsize=14, labelpad=2)
        ax.tick_params(axis="both", labelsize=8, pad=1)
        if show_titles:
            ax.set_title(ax.get_title(), fontsize=15,
                         fontweight="bold", pad=3)
        else:
            ax.set_title("")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.tight_layout(pad=0.2, h_pad=0.0, w_pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def vstack(images: list[Image.Image]) -> Image.Image:
    max_w = max(im.width for im in images)
    total_h = sum(im.height for im in images)
    out = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for im in images:
        out.paste(im, ((max_w - im.width) // 2, y))
        y += im.height
    return out


def grid_stack(images: list[Image.Image], n_cols: int) -> Image.Image:
    """Tile images into a grid with `n_cols` columns, row-major."""
    if not images:
        raise ValueError("no images to stack")
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    out = Image.new("RGB", (cell_w * n_cols, cell_h * n_rows), "white")
    for idx, im in enumerate(images):
        r, c = divmod(idx, n_cols)
        x = c * cell_w + (cell_w - im.width) // 2
        y = r * cell_h + (cell_h - im.height) // 2
        out.paste(im, (x, y))
    return out


def _parse_row(spec: str) -> tuple[str, str, int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"row spec must be 'dataset:sample:step' (got {spec!r})")
    return parts[0], parts[1], int(parts[2])


def main():
    p = argparse.ArgumentParser(description="5-way method comparison")
    p.add_argument("--row", action="append", type=_parse_row,
                   metavar="DATASET:SAMPLE:STEP",
                   help="Override row(s). Repeat for multiple rows.")
    p.add_argument("--out", type=Path,
                   default=PAPER_FIGURES / "fig_method_compare_5way.png")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--cols", type=int, default=1,
                   help="Number of step-block columns in the composite grid "
                        "(default 1 -> vstack; e.g. 4 rows x 6 panels each).")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = args.row if args.row else DEFAULT_ROWS
    # Row-major fill: with cols=2 and the default DEFAULT_ROWS order
    # (Acute,Acute,Acute,Bj,Bj,Bj), we'd get Acute beside Acute. Reorder
    # to interleave so column 1 = Acute, column 2 = Bjornson.
    if args.cols == 2 and rows == DEFAULT_ROWS:
        rows = [rows[0], rows[3], rows[1], rows[4], rows[2], rows[5]]
    print(f"Rendering {len(rows)} step blocks in a "
          f"{(len(rows) + args.cols - 1) // args.cols} x {args.cols} grid "
          f"-> {args.out}")
    for ds, sa, st in rows:
        print(f"  {ds} / {sa} / step {st:02d}")

    loader = BenchmarkLoader(
        benchmark_dir=REPO / "benchmark",
        data_dir=REPO / "data",
    )
    images = [render_row(loader, ds, sa, st, args.dpi,
                          show_titles=(i == 0))
              for i, (ds, sa, st) in enumerate(rows)]
    if len(images) == 1:
        out_image = images[0]
    elif args.cols <= 1:
        out_image = vstack(images)
    else:
        out_image = grid_stack(images, n_cols=args.cols)
    out_image.save(args.out, optimize=True)
    print(f"\nWrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
