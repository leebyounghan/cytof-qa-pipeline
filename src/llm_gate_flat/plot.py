"""Plot helpers for the cascading LLM-Gate pipeline.

Each step gets a 2-D scatter on its (x_marker, y_marker) channels showing
the predicted-parent cells coloured by predicted category, with the LLM's
gate rectangles / polygons overlaid.

Also produces a per-sample summary grid covering all completed steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.llm_gate.inference.output_schema import is_absent_gate

_UNASSIGNED_COLOR = (0.82, 0.82, 0.82)


def _palette(categories: Sequence[str]) -> dict[str, tuple]:
    base = (
        list(plt.get_cmap("tab10").colors)
        + list(plt.get_cmap("tab20").colors)
        + list(plt.get_cmap("tab20b").colors)
    )
    targets = sorted(c for c in categories if c != "Unassigned")
    return {c: base[i % len(base)] for i, c in enumerate(targets)}


def _iter_real_gates(gates: dict) -> Iterable[tuple[str, dict]]:
    for cat, g in gates.items():
        if g is None:
            continue
        if isinstance(g, dict) and is_absent_gate(g):
            continue
        if isinstance(g, list):
            for sub in g:
                if isinstance(sub, dict) and not is_absent_gate(sub):
                    yield cat, sub
        elif isinstance(g, dict):
            yield cat, g


def _draw_step(
    ax,
    *,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    predicted_labels: Sequence[str],
    gates: dict,
    options: Sequence[str],
    palette: dict[str, tuple],
    title: str,
    x_label: str,
    y_label: str,
    legend: bool,
    point_size: float,
) -> None:
    pred = np.asarray(predicted_labels)
    # Background: Unassigned first
    una_mask = pred == "Unassigned"
    if una_mask.any():
        ax.scatter(
            x_arr[una_mask], y_arr[una_mask],
            s=point_size, c=[_UNASSIGNED_COLOR], alpha=0.45,
            linewidths=0, label=f"Unassigned ({int(una_mask.sum())})",
        )
    for cat in [c for c in options if c != "Unassigned"]:
        m = pred == cat
        if not m.any():
            continue
        ax.scatter(
            x_arr[m], y_arr[m],
            s=point_size, c=[palette[cat]], alpha=0.7,
            linewidths=0, label=f"{cat} ({int(m.sum())})",
        )

    for cat, g in _iter_real_gates(gates):
        color = palette.get(cat, "black")
        if "vertices" in g:
            verts = np.asarray(g["vertices"], dtype=float)
            ax.add_patch(mpatches.Polygon(
                verts, closed=True, fill=False,
                edgecolor=color, linewidth=1.6,
            ))
        else:
            ax.add_patch(mpatches.Rectangle(
                (g["x_min"], g["y_min"]),
                g["x_max"] - g["x_min"],
                g["y_max"] - g["y_min"],
                fill=False, edgecolor=color, linewidth=1.6,
            ))

    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    if legend:
        ax.legend(loc="best", fontsize=6, markerscale=3, frameon=True)


def plot_step(
    out_path: Path,
    *,
    step_num: int,
    parent_expr: str,
    x_marker_display: str,
    y_marker_display: str,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    predicted_labels: Sequence[str],
    gates: dict,
    options: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=110)
    palette = _palette(options)
    title = (
        f"step_{step_num:02d}  |  parent: {parent_expr}\n"
        f"({x_marker_display} × {y_marker_display})  n={len(x_arr):,}"
    )
    _draw_step(
        ax,
        x_arr=x_arr, y_arr=y_arr,
        predicted_labels=predicted_labels, gates=gates, options=options,
        palette=palette, title=title,
        x_label=x_marker_display, y_label=y_marker_display,
        legend=True, point_size=4.0,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_sample_summary(
    out_path: Path,
    step_records: list[dict],
    *,
    sample: str,
    dataset: str,
) -> None:
    """Grid of per-step scatters for one sample.

    ``step_records`` is a list of dicts each containing the same kwargs
    accepted by :func:`plot_step` (minus ``out_path``).
    """
    n = len(step_records)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows), dpi=110,
    )
    axes = np.atleast_2d(axes)
    for i, rec in enumerate(step_records):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        palette = _palette(rec["options"])
        title = (
            f"step_{rec['step_num']:02d}: "
            f"{rec['x_marker_display']}×{rec['y_marker_display']} "
            f"n={len(rec['x_arr']):,}"
        )
        _draw_step(
            ax,
            x_arr=rec["x_arr"], y_arr=rec["y_arr"],
            predicted_labels=rec["predicted_labels"],
            gates=rec["gates"], options=rec["options"],
            palette=palette, title=title,
            x_label=rec["x_marker_display"],
            y_label=rec["y_marker_display"],
            legend=False, point_size=2.5,
        )
    # blank out unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle(
        f"{dataset} / {sample} — cascade ({n} steps)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
