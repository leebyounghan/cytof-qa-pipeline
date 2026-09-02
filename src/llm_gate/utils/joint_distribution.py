"""2-D joint-distribution peak / valley detection for the LLM-gate
prompt block.

Pipeline:

  1. ``np.histogram2d(x, y, bins=(grid_w, grid_h))`` — same orientation
     as ``with_ascii_2d_digits`` variant.
  2. ``log1p(counts)`` then 2-D Gaussian smoothing — peak detection on
     the log scale prevents one dominant mode from swamping minor ones.
  3. ``skimage.feature.peak_local_max`` — 2-D local maxima with a
     prominence-style absolute height threshold and a minimum-distance
     separator.
  4. **MST cross-section valleys** — for n peaks, build a minimum
     spanning tree on bin-space Euclidean distances; for each MST edge
     trace a Bresenham-like line on the smoothed density grid and pick
     the lowest density point on the line as the valley between that
     pair. Yields ``n - 1`` valleys for ``n`` peaks.

Modality- and semantics-agnostic — peaks/valleys are reported as
geometric features, not labelled "negative" / "positive". The LLM
decides which mode is which.

Public API:

- ``detect_joint_peaks_valleys(x, y, **kwargs) -> dict``
- ``render_joint_summary(x_marker, y_marker, info) -> str``
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.feature import peak_local_max


# ── Tuning ─────────────────────────────────────────────────────────────────

_GRID_W = 48
_GRID_H = 24
_SMOOTHING_SIGMA = 1.5
_MIN_PEAK_DISTANCE_BINS = 3
_TINYPEAK_RATIO = 1 / 25
_FRACTION_NEAR_WINDOW = 5            # 5×5 bin window around peak
_DEGENERATE_MIN_N = 20
_DEGENERATE_MIN_UNIQUE = 3

# Lower bound on the rendered axis max — anchors the visible scale to a
# biologically meaningful range (~arcsinh-cofactor saturation) so the LLM
# reads "high" / "low" against an absolute reference instead of a per-step
# data extent. Bins beyond the data tail render as empty cells.
_RENDER_MAX_FLOOR = 8.0


# ── Helpers ────────────────────────────────────────────────────────────────

def _bin_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _line_points(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    """Bresenham-like sampling of integer grid points along the line
    segment from p0 to p1 (inclusive).

    For our purposes we only need a "dense enough" set; a simple
    `linspace + round` over `max(|dw|, |dh|) + 1` points suffices.
    """
    (w0, h0), (w1, h1) = p0, p1
    n = max(abs(w1 - w0), abs(h1 - h0)) + 1
    if n <= 1:
        return [(int(w0), int(h0))]
    ws = np.round(np.linspace(w0, w1, n)).astype(int)
    hs = np.round(np.linspace(h0, h1, n)).astype(int)
    pts = list({(int(w), int(h)) for w, h in zip(ws, hs)})
    pts.sort(key=lambda wh: (wh[0], wh[1]))
    return pts


def _fraction_near(
    counts: np.ndarray, x_bin: int, y_bin: int, window: int,
) -> float:
    """Mass in (window × window) bin window around (x_bin, y_bin) /
    total mass. ``window`` should be odd; we use the symmetric half-width.
    """
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    half = window // 2
    W, H = counts.shape
    w0 = max(0, x_bin - half)
    w1 = min(W, x_bin + half + 1)
    h0 = max(0, y_bin - half)
    h1 = min(H, y_bin + half + 1)
    return float(counts[w0:w1, h0:h1].sum()) / total


def _degenerate_payload(
    n: int, params: dict, grid_w: int, grid_h: int,
    counts: np.ndarray | None,
    x_edges: np.ndarray | None, y_edges: np.ndarray | None,
) -> dict:
    return {
        'ok': False,
        'n': int(n),
        'grid_w': int(grid_w), 'grid_h': int(grid_h),
        'x_edges': (x_edges.tolist() if x_edges is not None else []),
        'y_edges': (y_edges.tolist() if y_edges is not None else []),
        'counts':  (counts.astype(int).tolist() if counts is not None else []),
        'peaks': [], 'valleys': [],
        'params': params,
    }


# ── Public: detection ──────────────────────────────────────────────────────

def detect_joint_peaks_valleys(
    x: np.ndarray, y: np.ndarray,
    *,
    grid_w: int = _GRID_W,
    grid_h: int = _GRID_H,
    smoothing_sigma: float = _SMOOTHING_SIGMA,
    min_peak_distance_bins: int = _MIN_PEAK_DISTANCE_BINS,
    tinypeak_ratio: float = _TINYPEAK_RATIO,
) -> dict:
    """Detect 2-D peaks + MST cross-section valleys on the joint
    distribution of ``(x, y)``. See module docstring for the algorithm.
    """
    params = {
        'grid_w': int(grid_w), 'grid_h': int(grid_h),
        'smoothing_sigma': float(smoothing_sigma),
        'min_peak_distance_bins': int(min_peak_distance_bins),
        'tinypeak_ratio': float(tinypeak_ratio),
    }
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    mask = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[mask], ya[mask]
    if xa.size != ya.size:
        raise ValueError('x and y must share length after NaN removal')
    n = int(xa.size)

    if n < _DEGENERATE_MIN_N:
        return _degenerate_payload(n, params, grid_w, grid_h, None, None, None)

    if (np.unique(xa).size < _DEGENERATE_MIN_UNIQUE
            or np.unique(ya).size < _DEGENERATE_MIN_UNIQUE):
        return _degenerate_payload(n, params, grid_w, grid_h, None, None, None)
    if xa.max() <= xa.min() or ya.max() <= ya.min():
        return _degenerate_payload(n, params, grid_w, grid_h, None, None, None)

    # Rendered axis max is floored at _RENDER_MAX_FLOOR so the LLM sees a
    # consistent absolute scale across steps; bins beyond the data tail are
    # zero-count and render as empty cells. Data extent is preserved
    # separately so the LLM can see where the data actually ends.
    x_lo, x_hi_data = float(xa.min()), float(xa.max())
    y_lo, y_hi_data = float(ya.min()), float(ya.max())
    x_hi = max(_RENDER_MAX_FLOOR, x_hi_data)
    y_hi = max(_RENDER_MAX_FLOOR, y_hi_data)

    counts, x_edges, y_edges = np.histogram2d(
        xa, ya, bins=(grid_w, grid_h),
        range=[[x_lo, x_hi], [y_lo, y_hi]],
    )
    counts = counts.astype(np.float64)              # shape (W, H)

    log_counts = np.log1p(counts)
    density = gaussian_filter(log_counts, sigma=smoothing_sigma)
    dmax = float(density.max())

    if dmax <= 0:
        # All bins empty after smoothing — degenerate.
        return _degenerate_payload(
            n, params, grid_w, grid_h, counts, x_edges, y_edges,
        )

    # ── 2-D peaks ──
    coords = peak_local_max(
        density,
        min_distance=int(min_peak_distance_bins),
        threshold_abs=float(tinypeak_ratio) * dmax,
        exclude_border=False,
    )
    # peak_local_max returns array of shape (n_peaks, 2) with (row, col).
    # Our density is indexed [W][H] so row=W=x_bin, col=H=y_bin.
    peaks_raw: list[dict] = []
    for r, c in coords:
        peaks_raw.append({
            'x_bin':  int(r),
            'y_bin':  int(c),
            'height': float(density[r, c]),
        })
    # Sort by height desc — overlay letters A B C ... follow this order.
    peaks_raw.sort(key=lambda p: -p['height'])

    xc = _bin_centers(x_edges)
    yc = _bin_centers(y_edges)
    peaks: list[dict] = []
    for p in peaks_raw:
        peaks.append({
            'x':              float(xc[p['x_bin']]),
            'y':              float(yc[p['y_bin']]),
            'x_bin':          p['x_bin'],
            'y_bin':          p['y_bin'],
            'height':         p['height'],
            'fraction_near':  _fraction_near(
                counts, p['x_bin'], p['y_bin'], _FRACTION_NEAR_WINDOW,
            ),
        })

    # ── MST → valleys ──
    valleys: list[dict] = []
    if len(peaks) >= 2:
        n_pk = len(peaks)
        # Pairwise Euclidean distance in bin space.
        coords_arr = np.array([[p['x_bin'], p['y_bin']] for p in peaks],
                              dtype=np.float64)
        diff = coords_arr[:, None, :] - coords_arr[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        # MST on upper-triangular cost; csgraph returns sparse mst.
        mst = minimum_spanning_tree(dist).toarray()
        edges = []
        for i in range(n_pk):
            for j in range(i + 1, n_pk):
                if mst[i, j] > 0 or mst[j, i] > 0:
                    edges.append((i, j))
        edges.sort()  # deterministic ordering for valley letter assignment.

        for li, ri in edges:
            p0 = (peaks[li]['x_bin'], peaks[li]['y_bin'])
            p1 = (peaks[ri]['x_bin'], peaks[ri]['y_bin'])
            pts = _line_points(p0, p1)
            # Exclude the two endpoints — we want a *between* point.
            interior = [pt for pt in pts if pt != p0 and pt != p1]
            if not interior:
                # Adjacent peaks (touching) — fall back to midpoint.
                mw, mh = (p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2
                interior = [(mw, mh)]
            # Min density point on the line.
            interior.sort(key=lambda wh: density[wh[0], wh[1]])
            vw, vh = interior[0]
            valleys.append({
                'x':               float(xc[vw]),
                'y':               float(yc[vh]),
                'x_bin':           int(vw),
                'y_bin':           int(vh),
                'depth':           float(density[vw, vh]),
                'left_peak_idx':   int(li),
                'right_peak_idx':  int(ri),
            })

    return {
        'ok':         True,
        'n':          n,
        'grid_w':     int(grid_w),
        'grid_h':     int(grid_h),
        'x_edges':    x_edges.tolist(),
        'y_edges':    y_edges.tolist(),
        'counts':     counts.astype(int).tolist(),
        'peaks':      peaks,
        'valleys':    valleys,
        # Data extent (renderer shows alongside the wider axis scale).
        'x_data_max': float(x_hi_data),
        'y_data_max': float(y_hi_data),
        'params':     params,
    }


# ── Public: rendering ──────────────────────────────────────────────────────

def _peak_label(i: int) -> str:
    return f'P{i + 1}'


def _valley_label(i: int) -> str:
    return f'V{i + 1}'


def _shade_grid(
    counts: np.ndarray, n_levels: int = 9,
) -> list[list[str]]:
    """Per-bin shading char (' ' or '1'..'9') using log-quantile binning.

    Mirrors the scheme in
    ``with_ascii_2d_digits/c2s_prompt.py::_render_ascii_heatmap``.
    """
    log_c = np.log1p(counts.astype(np.float64))
    nz = log_c[log_c > 0]
    W, H = counts.shape
    out = [[' '] * H for _ in range(W)]
    if nz.size == 0:
        return out
    qs = np.quantile(nz, np.arange(1, n_levels) / n_levels)
    for w in range(W):
        for h in range(H):
            v = log_c[w, h]
            if v <= 0:
                continue
            i = int(np.searchsorted(qs, v, side='right'))   # 0..n_levels-1
            out[w][h] = str(min(i + 1, n_levels))
    return out


def _stacked_x_axis_rows(
    n_bins: int, items: list[tuple[int, str]],
) -> list[str]:
    """Place each (bin, label) on a row such that no two markers overlap.

    A marker occupies ``len(label)`` cells starting at ``bin``. If a row
    is full at that range, push to a new row. Returns one or more
    strings of width ``n_bins``.
    """
    if not items:
        return []
    items = sorted(items, key=lambda kv: (kv[0], kv[1]))
    rows: list[list[str]] = []
    occupied: list[set[int]] = []
    for bin_idx, label in items:
        span = list(range(bin_idx, min(n_bins, bin_idx + len(label))))
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


def render_joint_summary(
    x_marker: str, y_marker: str, info: dict,
) -> str:
    """Render the joint-distribution dict into a heatmap text block.

    Conventions:
      - The grid is **pure density** (digits 1..9 + space). Peaks and
        valleys are NOT overwritten on the grid — their cells keep the
        density shading.
      - Y-axis labels show ``P{i}`` / ``V{i}`` markers (not numeric
        coordinates) for whichever peaks/valleys fall in that row.
      - X-axis tick rows show ``P{i}`` / ``V{i}`` markers at the column
        of each peak/valley. Multiple markers per column are stacked
        onto additional rows to avoid collision.
      - Numeric ``(x, y)`` coordinates live in the legend below the grid.
    """
    n = int(info.get('n', 0))
    header = (f'[Joint distribution: {x_marker} (x) × {y_marker} (y)]   '
              f'n={n:,}')

    if not info.get('ok', False) or not info.get('counts'):
        return f'{header}\n(degenerate input — 2D heatmap omitted)'

    counts = np.asarray(info['counts'], dtype=np.float64)
    x_edges = info.get('x_edges') or []
    y_edges = info.get('y_edges') or []
    W, H = counts.shape
    peaks = list(info.get('peaks') or [])
    valleys = list(info.get('valleys') or [])

    grid = _shade_grid(counts)

    # Axis ranges. x_edges/y_edges describe the rendered scale (extended
    # to the floor); ``x_data_max`` / ``y_data_max`` are the actual data
    # extents, surfaced separately so the LLM can see where the data tail
    # actually ends.
    x_min = float(x_edges[0]) if x_edges else 0.0
    x_max = float(x_edges[-1]) if x_edges else 0.0
    y_min = float(y_edges[0]) if y_edges else 0.0
    y_max = float(y_edges[-1]) if y_edges else 0.0
    x_data_max = float(info.get('x_data_max', x_max))
    y_data_max = float(info.get('y_data_max', y_max))
    x_min_str, x_max_str = f'{x_min:.2f}', f'{x_max:.2f}'
    y_min_str, y_max_str = f'{y_min:.2f}', f'{y_max:.2f}'

    def _edge_to_bin(edges, value, n_bins):
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

    show_xmx = x_edges and x_data_max < x_max
    show_ymx = y_edges and y_data_max < y_max
    x_mx_bin = _edge_to_bin(x_edges, x_data_max, W) if show_xmx else None
    y_mx_bin = _edge_to_bin(y_edges, y_data_max, H) if show_ymx else None

    # Per-row / per-column marker maps.
    y_labels: dict[int, list[str]] = {}
    x_labels: list[tuple[int, str]] = []
    for i, p in enumerate(peaks):
        lbl = _peak_label(i)
        y_labels.setdefault(int(p['y_bin']), []).append(lbl)
        x_labels.append((int(p['x_bin']), lbl))
    for i, v in enumerate(valleys):
        lbl = _valley_label(i)
        y_labels.setdefault(int(v['y_bin']), []).append(lbl)
        x_labels.append((int(v['x_bin']), lbl))
    if y_mx_bin is not None:
        y_labels.setdefault(int(y_mx_bin), []).append('My')
    if x_mx_bin is not None:
        x_labels.append((int(x_mx_bin), 'Mx'))

    # Y-axis label width = max of (widest joined marker, numeric tick).
    y_widths = [len('/'.join(labs)) for labs in y_labels.values()] or [0]
    y_label_w = max(len(y_min_str), len(y_max_str), 4, max(y_widths))

    legend_extra = (
        '   Mx=data x_max, My=data y_max'
        if (show_xmx or show_ymx) else ''
    )
    lines: list[str] = [
        header,
        f'data:  x ∈ [{x_min_str}, {x_data_max:.2f}]   '
        f'y ∈ [{y_min_str}, {y_data_max:.2f}]',
        f'scale: x ∈ [{x_min_str}, {x_max_str}]   '
        f'y ∈ [{y_min_str}, {y_max_str}]',
        "density (' '=empty, 1..9 = log-quantile)" + legend_extra,
        '',
        ' ' * (y_label_w + 1) + y_marker,
    ]
    # Y-axis: numeric ticks pinned to top (y_max) and bottom (y_min) rows
    # — combined with any P/V marker that falls there via '/' join.
    for yi in range(H - 1, -1, -1):
        labs = list(y_labels.get(yi, []))
        if yi == H - 1:
            labs = [y_max_str] + labs
        if yi == 0:
            labs = [y_min_str] + labs
        ylab = '/'.join(labs) if labs else ''
        row = ''.join(grid[xi][yi] for xi in range(W))
        lines.append(f' {ylab:>{y_label_w}} │{row}')
    lines.append(' ' * (y_label_w + 1) + '└' + '─' * W)

    indent = ' ' * (y_label_w + 2)
    # X-axis: numeric tick row (x_min at left, x_max at right) BEFORE
    # the P/V marker rows.
    x_tick_row = list(' ' * W)
    for k, ch in enumerate(x_min_str):
        if k < W:
            x_tick_row[k] = ch
    end_start = max(0, W - len(x_max_str))
    for k, ch in enumerate(x_max_str):
        if end_start + k < W:
            x_tick_row[end_start + k] = ch
    lines.append(indent + ''.join(x_tick_row))

    x_axis_rows = _stacked_x_axis_rows(W, x_labels)
    for r in x_axis_rows:
        lines.append(indent + r)
    lines.append(indent + ' ' * W + '  ' + x_marker)

    # Legends.
    if peaks:
        for i, p in enumerate(peaks):
            lines.append(
                f'{_peak_label(i)}: x={p["x"]:.2f}, y={p["y"]:.2f}, '
                f'frac={p["fraction_near"]:.2f}'
            )
    else:
        lines.append('peaks: none')

    if valleys:
        for i, v in enumerate(valleys):
            li, ri = int(v['left_peak_idx']), int(v['right_peak_idx'])
            lines.append(
                f'{_valley_label(i)}: x={v["x"]:.2f}, y={v["y"]:.2f} '
                f'(between {_peak_label(li)}↔{_peak_label(ri)})'
            )
    else:
        lines.append('valleys: none')

    return '\n'.join(lines)
