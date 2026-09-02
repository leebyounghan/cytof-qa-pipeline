"""
case_study/compare_gates_three.py — 3-way gate-method comparison.

Per (cohort, sample, step) row, left -> right:
  1. LLM-Gate gpt-5.4   — text-only single-shot   (results/gate_gpt-5.4_full)
  2. VLM-Gate gpt-5.4   — text + scatter image    (results/vlm_gpt-5.4_full)
  3. Agent-Gate gpt-5.4 — render-revise tool loop (results/agent_gpt-5.4_full)

Each panel renders the parent-population scatter (colored by predicted label,
same style as compare_five.py:plot_gates) overlaid with the per-category
axis-aligned rectangle the method emitted. The total rectangle area in the
(x_v, y_v) arcsinh plane is printed in the panel title so the gate-tightening
trend (LLM > VLM > Agent) is visible at a glance — this mirrors fig3's
"LLM-Gate gpt-5.4" panel style from compare_five.py.

Multiple step rows stack vertically via PIL.

Run:
    # run from the repository root
    uv run python -m case_study.compare_gates_three
    uv run python -m case_study.compare_gates_three \
        --row Acute2020:994570_Normalized:18 \
        --row Bjornson:R15W11_W070515101208_Hu_M_Basal:7
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

from src.bench import BenchmarkLoader

# Reuse helpers from compare_five so the visual style stays in lock-step
# with fig3's LLM-Gate panel.
from case_study.compare_five import (
    PALETTE,
    FALLBACK,
    build_color_map,
    plot_missing,
    load_pred,
    vstack,
    grid_stack,
)

# Local Unassigned styling — compare_five dims it (pale grey, small, low alpha)
# which makes the background almost invisible in these gate panels. Bump it up
# so the parent population shape is readable behind the rectangles.
UNASSIGNED_COLOR = "#888888"
UNASSIGNED_ALPHA = 0.45
UNASSIGNED_SIZE_DELTA = 0   # same as foreground points (was -2 in compare_five)


def scatter_by_label(ax, x, y, labels, color_map, alpha=0.5, s=4):
    """fig3-style scatter with a darker/larger Unassigned underlay so the
    parent-population shape stays visible behind the gate rectangles."""
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    # Draw Unassigned first (under-layer), real categories on top.
    order = sorted(uniq, key=lambda c: (str(c) != "Unassigned",
                                        -int((labels == c).sum())))
    for cat in order:
        m = labels == cat
        if str(cat) == "Unassigned":
            ax.scatter(x[m], y[m],
                       s=max(s + UNASSIGNED_SIZE_DELTA, 2),
                       alpha=UNASSIGNED_ALPHA,
                       color=UNASSIGNED_COLOR, edgecolors="none",
                       label=f"{cat} (n={int(m.sum())})")
        else:
            col = color_map.get(str(cat), "#333333")
            ax.scatter(x[m], y[m], s=s, alpha=alpha, color=col,
                       edgecolors="none", label=f"{cat} (n={int(m.sum())})")

REPO = Path(__file__).resolve().parent.parent
PAPER_FIGURES = Path(os.environ.get("PAPER_FIGURES", "case_study/figures"))

# (cohort, sample, step) defaults — overridable via --row.
DEFAULT_ROWS: list[tuple[str, str, int]] = [
    ("Acute2020", "994570_Normalized", 18),
    ("Acute2020", "994570_Normalized", 29),
    ("Bjornson",  "R15W11_W070515101208_Hu_M_Basal", 6),
    ("Bjornson",  "R15W11_W070515101208_Hu_M_Basal", 7),
]

METHODS: list[tuple[str, str, str]] = [
    # (display_title_prefix, results_dir, short_tag)
    ("LLM-Gate",   "gate_gpt-5.4_full", "llm"),
    ("VLM-Gate",   "vlm_gpt-5.4_full",  "vlm"),
    ("Agent-Gate", "agent_gpt-5.4_full", "agent"),
]


def _load_gates(results_dir: str, dataset: str, sample: str,
                step: int) -> tuple[dict, Path]:
    """Return (gates_dict, pred.json path).  gates_dict empty if unavailable."""
    pred_full = REPO / "results" / results_dir / dataset / "pred.json"
    if not pred_full.exists():
        return {}, pred_full
    try:
        gf = json.loads(pred_full.read_text())
    except Exception:
        return {}, pred_full
    return (gf.get("samples", {}).get(sample, {}).get("steps", {})
            .get(str(step), {}).get("gates", {}) or {}), pred_full


def _gate_total_area(gates: dict) -> float:
    """Sum of (x_max-x_min)*(y_max-y_min) across all categories with a valid
    rectangle. NaN-safe."""
    tot = 0.0
    for g in gates.values():
        try:
            w = float(g["x_max"]) - float(g["x_min"])
            h = float(g["y_max"]) - float(g["y_min"])
            if w > 0 and h > 0:
                tot += w * h
        except (KeyError, TypeError, ValueError):
            continue
    return tot


def _clip_to_axes(g: dict, xlim: tuple[float, float],
                  ylim: tuple[float, float]) -> tuple[float, float, float, float]:
    """Clip a gate to the visible axes so rectangles drawn outside the data
    range don't blow up auto-limits (we set explicit xlim/ylim anyway, but
    clipping keeps labels/legends honest)."""
    x0 = max(float(g["x_min"]), xlim[0])
    x1 = min(float(g["x_max"]), xlim[1])
    y0 = max(float(g["y_min"]), ylim[0])
    y1 = min(float(g["y_max"]), ylim[1])
    return x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0)


def plot_gates_panel(ax, ti, gates: dict, predicted, color_map, title: str):
    """fig3-style: predicted-colored scatter underneath, per-category
    axis-aligned rectangle overlaid."""
    if predicted is not None:
        scatter_by_label(ax, ti.X_2d[:, 0], ti.X_2d[:, 1], predicted,
                         color_map, alpha=0.45)
    else:
        # No per-cell prediction loaded; still show the raw distribution so
        # the gate is visually anchored.
        ax.scatter(ti.X_2d[:, 0], ti.X_2d[:, 1], s=3, c="#cccccc",
                   alpha=0.4, edgecolors="none")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for cat, g in gates.items():
        col = color_map.get(cat, "#333333")
        x0, y0, w, h = _clip_to_axes(g, xlim, ylim)
        if w <= 0 or h <= 0:
            continue
        rect = mpatches.Rectangle(
            (x0, y0), w, h, fill=False, edgecolor=col, linewidth=2.2,
        )
        ax.add_patch(rect)
        ax.text(x0, y0 + h, cat, fontsize=7, color=col,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.16", fc="white",
                          ec=col, alpha=0.85))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)


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

    # Per-method gates + per-cell predictions (same propagated prediction.json
    # layout across all three methods).
    method_state: list[tuple[str, dict, list | None, Path]] = []
    for title_prefix, rdir, _tag in METHODS:
        gates, pred_full = _load_gates(rdir, dataset, sample, step)
        ppath = (REPO / "results" / rdir / dataset / sample /
                 f"step_{step:02d}" / "prediction.json")
        pred = load_pred(ppath)
        n = len(ti.expression)
        if pred is not None and len(pred) != n:
            print(f"  [WARN] {title_prefix}: prediction length "
                  f"{len(pred)} != n={n} — dropping")
            pred = None
        area = _gate_total_area(gates)
        title = title_prefix
        print(f"  {title_prefix}: gates={len(gates)} total_area={area:.3f} "
              f"pred={'ok' if pred is not None else 'missing'}")
        method_state.append((title, gates, pred, pred_full))

    # Shared color map across all panels (categories + per-cell labels).
    sources: list[np.ndarray] = [np.asarray(ti.y)]
    for _t, gates, pred, _p in method_state:
        if pred is not None:
            sources.append(np.asarray(pred))
        if gates:
            sources.append(np.array(list(gates.keys()), dtype=object))
    color_map = build_color_map(sources)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0),
                             sharex=True, sharey=True)
    for ax, (title, gates, pred, pred_full) in zip(axes, method_state):
        if not gates and pred is None:
            plot_missing(ax, title, pred_full)
            continue
        plot_gates_panel(ax, ti, gates, pred, color_map, title)

    for ax in axes:
        ax.set_xlabel(ti.x_marker_display, fontsize=10, labelpad=2)
        ax.set_ylabel(ti.y_marker_display, fontsize=10, labelpad=2)
        ax.tick_params(axis="both", labelsize=8, pad=1)
        if show_titles:
            ax.set_title(ax.get_title(), fontsize=13,
                         fontweight="bold", pad=3)
        else:
            ax.set_title("")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.tight_layout(pad=0.2, h_pad=0.0, w_pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _parse_row(spec: str) -> tuple[str, str, int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"row spec must be 'dataset:sample:step' (got {spec!r})")
    return parts[0], parts[1], int(parts[2])


def main():
    p = argparse.ArgumentParser(
        description="3-way gate-method comparison (LLM / VLM / Agent)")
    p.add_argument("--row", action="append", type=_parse_row,
                   metavar="DATASET:SAMPLE:STEP",
                   help="Override row(s). Repeat for multiple rows.")
    p.add_argument("--out", type=Path,
                   default=PAPER_FIGURES / "fig_gate_compare_three.png")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--cols", type=int, default=1,
                   help="Number of step-block columns (default 1 = vstack).")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = args.row if args.row else DEFAULT_ROWS
    print(f"Rendering {len(rows)} step rows -> {args.out}")
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
