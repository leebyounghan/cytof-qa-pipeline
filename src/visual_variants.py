"""GT-free parent-population plot variants for the visual gating experiment."""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

from src.vlm_gate.task import _render_background_scatter


MAX_POINTS = 60_000
FIGSIZE = 6.4
DPI = 100


def describe_plot(text: str, style: str) -> str:
    """Keep visual instructions truthful while leaving other wording fixed."""
    if style not in {'scatter', 'contour', 'density'}:
        raise ValueError(style)
    if style == 'scatter':
        return text
    return re.sub(r'\bscatter\b',
                  lambda match: style.upper() if match.group().isupper() else style,
                  text, flags=re.IGNORECASE)


def extent_for(x: np.ndarray, y: np.ndarray) -> list[float]:
    xmn, xmx = float(np.min(x)), float(np.max(x))
    ymn, ymx = float(np.min(y)), float(np.max(y))
    if xmx <= xmn:
        xmn, xmx = xmn - 0.5, xmx + 0.5
    if ymx <= ymn:
        ymn, ymx = ymn - 0.5, ymx + 0.5
    return [xmn, xmx, ymn, ymx]


def _sample(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) > MAX_POINTS:
        idx = np.random.default_rng(42).choice(len(x), MAX_POINTS, replace=False)
        x, y = x[idx], y[idx]
    return x, y


def render_background(
    x: np.ndarray, y: np.ndarray, out_path: Path,
    extent: list[float], style: str,
) -> None:
    """Save a 640px undecorated image aligned exactly to ``extent``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if style == 'scatter':
        _render_background_scatter(x, y, out_path, extent)
        return
    if style not in {'density', 'contour'}:
        raise ValueError(f'Unknown plot style: {style}')

    x, y = _sample(x, y)
    bins = int(np.clip(1.5 * np.sqrt(len(x)), 65, 180))
    hist, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    field = hist.T
    fig = plt.figure(figsize=(FIGSIZE, FIGSIZE), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    if style == 'density':
        image = np.ma.masked_where(field <= 0, field)
        cmap = plt.colormaps['viridis'].copy()
        cmap.set_bad('white')
        ax.imshow(image, extent=extent, origin='lower', aspect='auto',
                  interpolation='nearest', cmap=cmap,
                  norm=LogNorm(vmin=1, vmax=max(2, float(field.max()))))
    else:
        smooth = gaussian_filter(field, sigma=1.8, mode='nearest')
        positive = smooth[smooth > 0]
        if positive.size:
            lo = max(0.15, float(np.quantile(positive, 0.20)))
            hi = max(lo * 1.01, float(smooth.max()))
            levels = np.geomspace(lo, hi, 10)
            xc = (xedges[:-1] + xedges[1:]) / 2
            yc = (yedges[:-1] + yedges[1:]) / 2
            ax.contourf(xc, yc, smooth, levels=levels, cmap='viridis',
                        norm=LogNorm(vmin=lo, vmax=hi), extend='max')
            ax.contour(xc, yc, smooth, levels=levels[::2],
                       colors='#303030', linewidths=0.40, alpha=0.75)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()
    fig.savefig(out_path, dpi=DPI, facecolor='white')
    plt.close(fig)


def load_visual_groups(
    dataset_path: str,
    input_root: str | Path,
    style: str,
    sample_filter: set[str] | None = None,
) -> list[dict]:
    """Load one immutable prompt and one selected plot per test step."""
    if style not in {'scatter', 'contour', 'density'}:
        raise ValueError(f'Unknown plot style: {style}')
    ds = Path(dataset_path).name
    ds_root = Path(input_root) / ds
    if not ds_root.is_dir():
        raise FileNotFoundError(ds_root)
    groups = []
    for sample_dir in sorted(ds_root.iterdir()):
        if not sample_dir.is_dir() or (sample_filter and sample_dir.name not in sample_filter):
            continue
        for step_dir in sorted(sample_dir.iterdir()):
            if not step_dir.is_dir():
                continue
            meta_path = step_dir / 'meta.json'
            prompt_path = step_dir / 'cells.md'
            plot_path = step_dir / f'{style}.png'
            for path in (meta_path, prompt_path, plot_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            meta = json.loads(meta_path.read_text())
            groups.append({
                'prompt_text': prompt_path.read_text(),
                'options': meta['options'],
                'sample': sample_dir.name,
                'step': meta['step'],
                'ablation': 'full',
                'x_marker': meta['x_marker'],
                'y_marker': meta['y_marker'],
                'scatter_bg_path': str(plot_path.resolve()),
                'scatter_extent': meta['scatter_extent'],
            })
    if not groups:
        raise RuntimeError(f'No visual inputs under {ds_root}')
    return groups
