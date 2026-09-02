"""Matplotlib helpers used by the c2s generator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


UNASSIGNED_COLOR = '#cccccc'


def _build_category_colors(categories: Sequence[str]) -> dict[str, tuple]:
    """Stable category → RGBA color map. ``Unassigned`` always → light gray."""
    import matplotlib.colors as mcolors
    from matplotlib import cm

    named = [c for c in categories if c != 'Unassigned']
    cmap_name = 'tab10' if len(named) <= 10 else 'tab20'
    cmap = cm.get_cmap(cmap_name)
    colors: dict[str, tuple] = {}
    for i, c in enumerate(named):
        colors[c] = cmap(i % cmap.N)
    colors['Unassigned'] = mcolors.to_rgba(UNASSIGNED_COLOR)
    return colors


def _normalize_gt(labels: np.ndarray) -> np.ndarray:
    """Map NaN / empty / 'nan' strings to 'Unassigned'."""
    out = np.asarray(labels, dtype=object).copy()
    for i, v in enumerate(out):
        if v is None:
            out[i] = 'Unassigned'
            continue
        s = str(v)
        if not s or s.lower() == 'nan':
            out[i] = 'Unassigned'
        else:
            out[i] = s
    return out.astype(str)


def _build_cluster_palette(n: int, seed: int = 0) -> np.ndarray:
    """Cluster-id colour palette. Falls back to HSV for very large K."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    bases = [
        plt.get_cmap('tab20')(np.arange(20)),
        plt.get_cmap('tab20b')(np.arange(20)),
        plt.get_cmap('tab20c')(np.arange(20)),
    ]
    base = np.concatenate(bases)[:, :3]
    rng = np.random.default_rng(seed)
    if n <= len(base):
        idx = np.arange(n)
        rng.shuffle(idx)
        return base[idx[:n]]
    out = np.zeros((n, 3), dtype=np.float32)
    out[: len(base)] = base
    for i in range(len(base), n):
        out[i] = mpl.colors.hsv_to_rgb([
            rng.uniform(0, 1),
            rng.uniform(0.55, 0.95),
            rng.uniform(0.55, 0.95),
        ])
    return out


def plot_cluster_cells(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    gt_labels_parent: np.ndarray,
    categories: Sequence[str],
    cluster_labels: np.ndarray,
    rep_parent_local: np.ndarray,
    x_marker: str,
    y_marker: str,
    step_num: int,
    cluster_method: str,
    n_clusters: int,
    output_path: str,
    max_scatter: int = 20000,
    point_size: float = 10.0,
) -> None:
    """2-panel scatter: GT | clusters (with rep-cell stars overlaid)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(x_vals)
    if n == 0:
        return
    if n > max_scatter:
        bg_idx = np.random.default_rng(0).choice(n, max_scatter, replace=False)
    else:
        bg_idx = np.arange(n)
    bg_x = x_vals[bg_idx]
    bg_y = y_vals[bg_idx]
    bg_gt = _normalize_gt(np.asarray(gt_labels_parent)[bg_idx])
    bg_clusters = np.asarray(cluster_labels)[bg_idx]

    cat_list = list(categories) if categories else []
    if 'Unassigned' not in cat_list:
        cat_list.append('Unassigned')
    extras = [c for c in sorted(set(bg_gt.tolist())) if c not in cat_list]
    cat_list += extras
    gt_color_map = _build_category_colors(cat_list)
    gt_colors = np.array([gt_color_map[c] for c in bg_gt])

    n_pal = max(int(cluster_labels.max()) + 1 if cluster_labels.size else 1, 1)
    cluster_pal = _build_cluster_palette(n_pal, seed=3)
    cluster_colors = cluster_pal[bg_clusters % n_pal]

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 5.8), sharex=True, sharey=True,
    )
    xmin, xmax = float(bg_x.min()), float(bg_x.max())
    ymin, ymax = float(bg_y.min()), float(bg_y.max())

    def _scatter(ax, colors, title):
        ax.scatter(bg_x, bg_y, s=point_size, c=colors,
                   alpha=0.55, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x_marker)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    _scatter(axes[0], gt_colors, 'GT')
    _scatter(axes[1], cluster_colors,
             f'{cluster_method}  (K = {n_clusters})')

    # Overlay rep cells on the cluster panel.
    rep = np.asarray(rep_parent_local)
    rep = rep[rep >= 0]
    if rep.size:
        axes[1].scatter(
            x_vals[rep], y_vals[rep],
            marker='*', s=140,
            color='white', edgecolors='black', linewidths=0.8,
            zorder=5,
        )

    axes[0].set_ylabel(y_marker)

    present = [c for c in cat_list if c in set(bg_gt.tolist())]
    handles = [
        Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=gt_color_map[c], markeredgecolor='none',
               markersize=7, label=c)
        for c in present
    ]
    fig.legend(
        handles=handles, loc='lower center',
        ncol=min(len(handles), 8), frameon=False,
        bbox_to_anchor=(0.5, -0.02), fontsize=9,
    )

    fig.suptitle(
        f'Step {step_num}: {x_marker} vs {y_marker}  —  '
        f'clustering ({cluster_method}, K = {n_clusters})',
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
