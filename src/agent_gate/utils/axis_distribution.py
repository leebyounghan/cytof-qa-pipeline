"""1-D axis binning for the LLM-gate axis-distribution prompt block.

Bins each axis into a fixed number of bins (default 48) for ASCII bar
rendering, **but** detects 1-D peaks / valleys on a flowDensity-style
KDE rather than the binned histogram. The renderer turns this into an
ASCII density bar with ``P{i}`` / ``V{i}`` markers + numeric legend —
no raw count / center arrays in the prompt.

Why KDE for detection (not histogram) — see ``_detect_kde_peaks_valleys``.
The short version: in cytometry it is normal for a dominant cluster
(say CD3-CD19- residuals at CD19~0) to outweigh a rare-but-biologically-
meaningful cluster (say Bcells at CD19~3.7) by ~25×. Detection on raw
density (flowDensity default) loses the rare cluster — its peak is
~4% of max. Detection on ``log1p(density × N)`` (this module) keeps the
rare cluster — its peak is ~68% of max in log space, well above the
``tinypeak_removal`` filter. The histogram is retained only because the
ASCII bar still needs per-bin counts.

Public API:

- ``compute_axis_bins(values, n_bins=48) -> dict``
- ``render_axis_summary(marker, info, role='x') -> str``

ASCII rendering helpers (``_density_bar``, ``_bin_index_for_value``,
``_annotation_rows``, ``_fmt``) are re-used by ``density_clouds.py``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


_N_BINS = 48
_BLOCKS = '▁▂▃▄▅▆▇█'

_KDE_ADJUST = 1.5                # bandwidth multiplier (flowDensity default)
_KDE_GRID = 512                  # KDE evaluation grid resolution
_TINYPEAK_REMOVAL = 0.5          # drop peaks whose log-height < 0.5 × max
                                 # log-height (in log1p(density×N) space).
                                 # 0.5 admits a 4–5%-of-parent rare cluster
                                 # alongside a dominant cluster, while still
                                 # cutting tail/shoulder ghost peaks (which
                                 # sit at <0.3 × max in log space).


def compute_axis_bins(
    values: np.ndarray, *, n_bins: int = _N_BINS,
) -> dict:
    """Bin a 1-D array into ``n_bins`` linear bins over the data extent.

    Returns a dict with bin centers and per-bin cell counts. Empty /
    degenerate inputs yield empty lists with ``n=0``.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)

    if n == 0:
        return {
            'n':           0,
            'min':         0.0,
            'max':         0.0,
            'n_bins':      int(n_bins),
            'bin_centers': [],
            'counts':      [],
        }

    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6

    edges = np.linspace(vmin, vmax, n_bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    peaks, valleys = _detect_kde_peaks_valleys(arr, edges)

    return {
        'n':           n,
        'min':         vmin,
        'max':         float(arr.max()),
        'n_bins':      int(n_bins),
        'bin_centers': [float(c) for c in centers],
        'counts':      [int(c) for c in counts],
        'peaks':       peaks,
        'valleys':     valleys,
    }


def _x_to_bin(edges: np.ndarray, x: float) -> int:
    """Map a continuous x-position to its histogram bin index, clamped."""
    n_bins = len(edges) - 1
    if n_bins <= 0:
        return 0
    idx = int(np.searchsorted(edges, x, side='right') - 1)
    return int(max(0, min(n_bins - 1, idx)))


def _detect_kde_peaks_valleys(
    values: np.ndarray,
    edges: np.ndarray,
    *,
    adjust: float = _KDE_ADJUST,
    n_grid: int = _KDE_GRID,
    tinypeak_removal: float = _TINYPEAK_REMOVAL,
) -> tuple[list[dict], list[dict]]:
    """Detect 1-D peaks / valleys on a flowDensity-style KDE.

    Pipeline (close to flowDensity::getPeaks, with one critical change):

      1. KDE on the raw values (Gaussian kernel, Silverman bandwidth ×
         ``adjust``). flowDensity uses ``adjust=1.5`` by default.
      2. Convert to **log-counts**: ``log_d = log1p(density × N)``. This
         is the critical departure from flowDensity. On raw density a
         rare-but-real cluster (say 4 % of parent) sits at ~4 % of the
         dominant peak's height — any reasonable ``tinypeak_removal``
         kills it. In log space the same cluster sits at ~70 % of the
         dominant peak's height and survives the filter, while shoulder
         / tail "ghost" peaks (with no underlying mass) stay below ~30 %
         and are still discarded.
      3. ``find_peaks`` over ``log_d``; relative-height filter at
         ``tinypeak_removal × max(log_d_at_peaks)``.
      4. Valleys: ``argmin(log_d)`` in the open interval between adjacent
         kept peaks (left-to-right).

    Returns the same shape as the legacy histogram-based detector so
    ``render_axis_summary`` and downstream consumers don't need changes:
    ``(peaks, valleys)`` where each peak carries ``{x, bin, height,
    fraction_near}`` and each valley carries ``{x, bin, depth,
    left_peak_idx, right_peak_idx}``. ``fraction_near`` is preserved as
    a numeric field for backward compat — we now compute it as the share
    of mass within ±5 % of the peak's x-position (window-free).
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n < 2:
        return [], []
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax <= vmin:
        return [], []

    try:
        kde = gaussian_kde(arr, bw_method='silverman')
        kde.set_bandwidth(kde.factor * float(adjust))
    except (np.linalg.LinAlgError, ValueError):
        # Singular covariance → degenerate axis (e.g. all values identical).
        return [], []

    x_grid = np.linspace(vmin, vmax, int(n_grid))
    density = kde(x_grid)
    log_d = np.log1p(density * float(n))
    if not np.isfinite(log_d).any():
        return [], []

    raw = find_peaks(log_d)[0]
    if len(raw) == 0:
        return [], []

    heights = log_d[raw]
    max_h = float(heights.max())
    if max_h <= 0.0:
        return [], []
    keep_mask = heights >= float(tinypeak_removal) * max_h
    keep = raw[keep_mask]
    keep = np.sort(keep)

    n_bins = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    span = vmax - vmin
    near_radius = 0.05 * span if span > 0 else 1e-9

    peaks: list[dict] = []
    for p_idx in keep:
        xpk = float(x_grid[p_idx])
        b = _x_to_bin(edges, xpk)
        # fraction_near: share of cells within ±5 % of x-range around peak
        frac_near = float(np.mean(np.abs(arr - xpk) <= near_radius))
        peaks.append({
            'x':              xpk,
            'bin':            b,
            'height':         float(log_d[p_idx]),
            'fraction_near':  frac_near,
        })

    valleys: list[dict] = []
    for i in range(len(keep) - 1):
        lo, hi = int(keep[i]) + 1, int(keep[i + 1])
        if hi <= lo:
            continue
        v_local = int(np.argmin(log_d[lo:hi]))
        v_idx = lo + v_local
        xv = float(x_grid[v_idx])
        valleys.append({
            'x':              xv,
            'bin':            _x_to_bin(edges, xv),
            'depth':          float(log_d[v_idx]),
            'left_peak_idx':  int(i),
            'right_peak_idx': int(i + 1),
        })

    return peaks, valleys


def render_axis_summary(
    marker: str, info: dict, role: str = 'x',
) -> str:
    """Render an axis-bins dict into a text block for the LLM prompt.

    Layout::

        ### {marker} ({role}-axis), range [{vmin}, {vmax}], n=N parent cells
        ▁▁▂▃▅▇█▇▅▃▂▁ ... ▁     (log-scale density bar, n_bins chars)
                P1     V1     P2   (peak / valley markers, may stack)
        {vmin}                          {vmax}
        P1: x=1.20, frac=0.18
        P2: x=3.40, frac=0.42
        V1: x=2.30 (between P1↔P2)

    Numeric ``counts`` and ``bin_centers`` arrays are deliberately
    omitted from the prompt — the bar conveys shape, the markers anchor
    mode locations, and the legend carries exact x-values plus the
    fraction of mass in a 5-bin window around each peak.
    """
    n = int(info.get('n', 0))
    vmin = float(info.get('min', 0.0))
    vmax = float(info.get('max', 0.0))
    decimals = 3 if abs(vmax - vmin) < 1.0 else 2

    def _fmt2(v: float) -> str:
        return f'{v:.{decimals}f}'

    header = (
        f'### {marker} ({role}-axis), '
        f'range [{_fmt2(vmin)}, {_fmt2(vmax)}], '
        f'n={n} parent cells'
    )

    counts = list(info.get('counts', []) or [])
    if not counts:
        return header + '\n(no data)'

    n_bins = len(counts)
    bar = _density_bar(counts, scale='log')

    peaks = list(info.get('peaks') or [])
    valleys = list(info.get('valleys') or [])
    peak_bins = [(int(p['bin']), f'P{i + 1}') for i, p in enumerate(peaks)]
    valley_bins = [(int(v['bin']), f'V{i + 1}') for i, v in enumerate(valleys)]
    ann_rows = _annotation_rows(n_bins, peak_bins, valley_bins)

    vmin_str, vmax_str = _fmt2(vmin), _fmt2(vmax)
    tick_row = [' '] * n_bins
    for k, ch in enumerate(vmin_str):
        if k < n_bins:
            tick_row[k] = ch
    end_start = max(0, n_bins - len(vmax_str))
    for k, ch in enumerate(vmax_str):
        if end_start + k < n_bins:
            tick_row[end_start + k] = ch

    lines: list[str] = [header, bar]
    lines.extend(ann_rows)
    lines.append(''.join(tick_row))

    if peaks:
        for i, p in enumerate(peaks):
            lines.append(
                f'P{i + 1}: x={_fmt2(float(p["x"]))}, '
                f'frac={float(p["fraction_near"]):.2f}'
            )
    else:
        lines.append('peaks: none')

    if valleys:
        for i, v in enumerate(valleys):
            li = int(v['left_peak_idx'])
            ri = int(v['right_peak_idx'])
            lines.append(
                f'V{i + 1}: x={_fmt2(float(v["x"]))} '
                f'(between P{li + 1}↔P{ri + 1})'
            )
    elif len(peaks) >= 2:
        lines.append('valleys: none')

    return '\n'.join(lines)


# ── ASCII rendering helpers (shared with density_clouds.py) ────────────────

def _fmt(v: float, decimals: int = 2) -> str:
    return f'{v:.{decimals}f}'


def _bin_index_for_value(edges: list[float], value: float) -> int:
    """Return bin index containing ``value``; clamped to ``[0, n_bins-1]``."""
    n_bins = len(edges) - 1
    if n_bins <= 0:
        return 0
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return n_bins - 1
    for i in range(n_bins):
        if edges[i] <= value < edges[i + 1]:
            return i
    return n_bins - 1


def _density_bar(counts: list[int], scale: str = 'linear') -> str:
    """Render counts as a unicode-block density bar.

    ``scale='linear'`` shows true mass; ``scale='log'`` uses ``log1p``
    so minor modes remain visible alongside a dominant peak.
    """
    if not counts:
        return ''
    if scale == 'log':
        vals = [math.log1p(max(0, c)) for c in counts]
    else:
        vals = [float(max(0, c)) for c in counts]
    mx = max(vals) if vals else 0.0
    if mx <= 0:
        return _BLOCKS[0] * len(counts)
    out = []
    for v, c in zip(vals, counts):
        if c <= 0:
            out.append(' ')
            continue
        level = int(round((v / mx) * (len(_BLOCKS) - 1)))
        level = max(0, min(len(_BLOCKS) - 1, level))
        out.append(_BLOCKS[level])
    return ''.join(out)


def _annotation_rows(
    n_bins: int,
    peak_bins: list[tuple[int, str]],
    thr_bins: list[tuple[int, str]],
    extra_bins: list[tuple[int, str]] | None = None,
) -> list[str]:
    """Build annotation rows where each marker hangs under its bin.

    Markers that share / overlap a bin span stack onto consecutive rows
    so labels never collide. Returns one or more strings of length
    ``n_bins``.
    """
    items: list[tuple[int, str]] = []
    items.extend(peak_bins)
    items.extend(thr_bins)
    if extra_bins:
        items.extend(extra_bins)
    if not items:
        return []
    items.sort(key=lambda kv: (kv[0], kv[1]))
    rows: list[list[str]] = []
    occupied: list[set[int]] = []

    for bin_idx, label in items:
        span = range(bin_idx, min(n_bins, bin_idx + len(label)))
        placed = False
        for row, taken in zip(rows, occupied):
            if not any(b in taken for b in span):
                for b in span:
                    row[b] = label[b - bin_idx]
                taken.update(span)
                placed = True
                break
        if not placed:
            row = [' '] * n_bins
            taken: set[int] = set()
            for b in span:
                row[b] = label[b - bin_idx]
                taken.add(b)
            rows.append(row)
            occupied.append(taken)

    return [''.join(r) for r in rows]
