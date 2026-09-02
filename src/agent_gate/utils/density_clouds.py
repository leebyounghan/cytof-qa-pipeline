"""Density-level cloud detection for the LLM-gate prompt block.

Level-walk peak detection + cell-count merge + watershed:

  1. **Levels grid** = ``_compute_levels(counts)`` → 0 for empty bins,
     1..9 for log-quantile of ``log1p(counts)``. Mirrors
     ``joint_distribution._shade_grid``.
  2. **Walk ``k = 9..1``**. At each ``k`` we compute connected components
     on ``levels >= k`` and reconcile them with the previously tracked
     peaks. Three cases per CC:
       - **No live peak inside** → a new peak is born at the highest-
         density bin of the CC (``birth_level = k``). Sub-modes living
         inside an already-tracked CC do **not** spawn extra peaks —
         dense regions naturally yield one peak.
       - **One live peak inside** → continuation; update the peak's
         ``max_cells_alone`` (the largest cell count its CC has reached
         while it was the lone live peak).
       - **Multiple live peaks inside** → merger event. For each pair of
         live peaks first sharing a CC, decide by the *shorter* peak's
         ``max_cells_alone``:
           - ``< threshold`` → the shorter peak is a noise sub-mode of
             the taller one; drop it (alive=False).
           - ``≥ threshold`` → the shorter peak is a genuinely distinct
             cluster that happens to be topologically connected to the
             taller one (e.g. via a dense bridge column). Both stay
             alive; their ``death_level`` is recorded as ``k + 1`` for
             reporting.
  3. **Final filter**: drop any alive peak whose ``max_cells_alone`` is
     still below the threshold (covers noise peaks born at low ``k``
     that never participated in a merger).
  4. **Watershed** flood-fill on the smoothed density, seeded by the
     surviving peak bins. Every grid bin (including empty / out-of-mode
     bins) ends up labelled — the parent population is fully partitioned,
     no diffuse residue.

The threshold is ``max(min_cells_frac × n_parent, min_cells_floor)`` —
relative-to-sample with a small absolute floor so very small samples
still drop true noise.

``birth_level`` / ``death_level`` / ``persistence`` per cloud are kept
as **informational** fields for the LLM prompt — they describe how
isolated each mode is — but they are not used as filter criteria.

Output:
  - per-cloud ``peak``, ``birth_level``, ``death_level``, ``persistence``,
    cell stats (``n``, ``frac``, ``mean``, ``std``), bbox of the cloud's
    bins, ``x_histogram`` / ``y_histogram`` (per-axis 1-D bars), and a
    single-letter ``label`` (A, B, …) — sorted by peak height desc.
  - ``overlay``: a ``W × H`` grid of single-character cloud labels —
    same orientation as the global heatmap so the LLM can align them
    mentally. Every grid bin is labelled (no ``.``).

Public API:

- ``detect_density_clouds(x, y, joint_info, **kwargs) -> dict``
- ``render_density_clouds(x_marker, y_marker, info) -> str``
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from src.agent_gate.utils.axis_distribution import (
    _annotation_rows,
    _bin_index_for_value,
    _density_bar,
    _fmt,
)


# ── Tuning ─────────────────────────────────────────────────────────────────

_N_LEVELS = 9
# Mirrors joint_distribution._SMOOTHING_SIGMA — the watershed surface uses
# the same gaussian-smoothed log1p(counts) the joint detector ran peak
# detection on, so cluster basins agree with the visible peaks.
_SMOOTHING_SIGMA = 1.5
# A peak is dropped (as a noise sub-mode) when its ``max_cells_alone``
# (the largest cell count its CC reached while it was the only live peak
# inside) is below ``max(min_cells_frac × n_parent, min_cells_floor)``.
# - 0.005 (0.5 %) catches low-density local maxima that only ever covered
#   a few sparse bins, while preserving genuinely distinct clusters whose
#   alone-CC mass is comparable to the parent population.
# - The absolute floor of 50 keeps very small samples from over-segmenting
#   into 1-cell "clusters" when 0.5 % rounds below the noise level.
_MIN_CELLS_FRAC = 0.005
_MIN_CELLS_FLOOR = 50

# Per-cloud 1-D histogram resolution (rendered to ASCII bar). Mirrors
# ``axis_distribution._HIST_BINS_RENDER`` so the LLM sees a consistent
# bar width across global / per-cloud blocks.
_HIST_BINS_RENDER = 50

# 4-connectivity for component labelling on the (W, H) grid.
_CONNECTIVITY = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool,
)


# ── Histogram helper ───────────────────────────────────────────────────────

def _axis_histogram(
    vals: np.ndarray, vmin: float, vmax: float, n_bins: int,
) -> dict:
    """Per-axis 1-D histogram on the rendered scale ``[vmin, vmax]``.

    Bins beyond the cell mass are zero-filled — the bar visually aligns
    with the global axis distribution, regardless of how local the cloud
    is.
    """
    if vals.size == 0 or vmax <= vmin:
        edges = np.linspace(vmin, max(vmin + 1e-6, vmax), n_bins + 1)
        return {
            'edges':  edges.astype(float).tolist(),
            'counts': [0] * n_bins,
            'n_bins': int(n_bins),
        }
    edges = np.linspace(vmin, vmax, n_bins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    return {
        'edges':  edges.astype(float).tolist(),
        'counts': counts.astype(int).tolist(),
        'n_bins': int(n_bins),
    }


# ── Level computation ──────────────────────────────────────────────────────

def _compute_levels(counts: np.ndarray, n_levels: int = _N_LEVELS) -> np.ndarray:
    """Per-bin level: 0 for empty, 1..n_levels by log-quantile of nonzero
    bins. Same scheme as ``joint_distribution._shade_grid``.
    """
    log_c = np.log1p(counts.astype(np.float64))
    levels = np.zeros_like(log_c, dtype=np.int32)
    nonzero = log_c > 0
    nz = log_c[nonzero]
    if nz.size == 0:
        return levels
    qs = np.quantile(nz, np.arange(1, n_levels) / n_levels)
    idx = np.searchsorted(qs, log_c[nonzero], side='right')   # 0..n_levels-1
    levels[nonzero] = np.clip(idx + 1, 1, n_levels)
    return levels


# ── Detection ──────────────────────────────────────────────────────────────

def detect_density_clouds(
    x: np.ndarray, y: np.ndarray, joint_info: dict,
    *,
    n_levels: int = _N_LEVELS,
    smoothing_sigma: float = _SMOOTHING_SIGMA,
    min_cells_frac: float = _MIN_CELLS_FRAC,
    min_cells_floor: int = _MIN_CELLS_FLOOR,
) -> dict:
    """Level-walk peak detection + cell-count merge + watershed flood-fill.

    Every grid bin ends up labelled — the parent population is fully
    partitioned, no diffuse residue. See the module docstring for a
    description of the walk and merge rules.
    """
    params = {
        'n_levels':         int(n_levels),
        'smoothing_sigma':  float(smoothing_sigma),
        'min_cells_frac':   float(min_cells_frac),
        'min_cells_floor':  int(min_cells_floor),
    }
    empty = {
        'ok':        False,
        'n_clouds':  0,
        'clouds':    [],
        'overlay':   [],
        'grid_w':    0,
        'grid_h':    0,
        'x_edges':   [],
        'y_edges':   [],
        'n_outside': 0,
        'frac_outside': 0.0,
        'diffuse':   None,
        'params':    params,
    }

    if not joint_info or not joint_info.get('ok'):
        return empty
    counts_raw = joint_info.get('counts') or []
    counts = np.array(counts_raw, dtype=np.int64)
    if counts.size == 0 or counts.ndim != 2:
        return empty

    W, H = counts.shape
    levels_grid = _compute_levels(counts, n_levels)
    # Smoothed surface for watershed — log1p compresses the dominant mode
    # so secondary modes still flood from their own basins.
    density = ndimage.gaussian_filter(
        np.log1p(counts.astype(np.float64)), smoothing_sigma,
    )

    n_total_raw = int(counts.sum())
    threshold = max(int(min_cells_frac * n_total_raw), int(min_cells_floor))

    # ── Step 1: level walk — discover peaks + decide mergers ─────────────
    # tracked_peaks[pi] = {x_bin, y_bin, birth_level, death_level, alive,
    #                      max_cells_alone}
    tracked_peaks: list[dict] = []
    handled_pairs: set[frozenset] = set()

    for k in range(n_levels, 0, -1):
        mask = levels_grid >= k
        if not mask.any():
            continue
        cc_labels, n_ccs = ndimage.label(mask, structure=_CONNECTIVITY)
        for cid in range(1, n_ccs + 1):
            cc_mask = (cc_labels == cid)
            cells_in_cc = int(counts[cc_mask].sum())
            contained = [
                pi for pi, p in enumerate(tracked_peaks)
                if p['alive'] and cc_mask[p['x_bin'], p['y_bin']]
            ]
            if not contained:
                # New peak born at this CC's highest-density bin.
                ws_, hs_ = np.where(cc_mask)
                best = int(density[ws_, hs_].argmax())
                wb, hb = int(ws_[best]), int(hs_[best])
                tracked_peaks.append({
                    'x_bin':           wb,
                    'y_bin':           hb,
                    'birth_level':     k,
                    'death_level':     None,
                    'alive':           True,
                    'max_cells_alone': cells_in_cc,
                })
            elif len(contained) == 1:
                # Continuation — peak is the lone live tenant of the CC.
                pi = contained[0]
                if cells_in_cc > tracked_peaks[pi]['max_cells_alone']:
                    tracked_peaks[pi]['max_cells_alone'] = cells_in_cc
            else:
                # Multiple live peaks → merger. Process each new pair.
                heights = {
                    pi: density[
                        tracked_peaks[pi]['x_bin'],
                        tracked_peaks[pi]['y_bin'],
                    ] for pi in contained
                }
                ranked = sorted(contained, key=lambda p: -heights[p])
                taller = ranked[0]
                for shorter in ranked[1:]:
                    pair = frozenset((taller, shorter))
                    if pair in handled_pairs:
                        continue
                    handled_pairs.add(pair)
                    if (tracked_peaks[shorter]['max_cells_alone']
                            < threshold):
                        # Noise sub-mode of the taller peak — drop.
                        tracked_peaks[shorter]['alive'] = False
                        if tracked_peaks[shorter]['death_level'] is None:
                            tracked_peaks[shorter]['death_level'] = k + 1
                    else:
                        # Distinct cluster — both stay alive; record the
                        # death-of-isolation level for reporting.
                        if tracked_peaks[shorter]['death_level'] is None:
                            tracked_peaks[shorter]['death_level'] = k + 1
                        if tracked_peaks[taller]['death_level'] is None:
                            tracked_peaks[taller]['death_level'] = k + 1

    # ── Step 2: final filter — drop alive peaks below threshold ─────────
    # Catches noise peaks born late (low k, tiny isolated CC) that never
    # participated in a merger but stayed alive.
    survivor_indices = [
        pi for pi, p in enumerate(tracked_peaks)
        if p['alive'] and p['max_cells_alone'] >= threshold
    ]
    if not survivor_indices:
        return {**empty, 'ok': True}
    # Sort by peak height desc — A is the tallest mode.
    survivor_indices.sort(
        key=lambda pi: -float(density[tracked_peaks[pi]['x_bin'],
                                       tracked_peaks[pi]['y_bin']]),
    )

    # ── Step 3: watershed seeded by survivors ────────────────────────────
    markers = np.zeros((W, H), dtype=np.int32)
    for new_id, pi in enumerate(survivor_indices, start=1):
        markers[tracked_peaks[pi]['x_bin'],
                tracked_peaks[pi]['y_bin']] = new_id
    cluster_grid = watershed(-density, markers=markers, connectivity=1)

    # ── Step 4: cell assignment + per-cluster stats ──────────────────────
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    keep = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[keep], ya[keep]
    n_total = int(xa.size)
    if n_total == 0:
        return {**empty, 'ok': True}

    x_edges = np.array(joint_info['x_edges'], dtype=np.float64)
    y_edges = np.array(joint_info['y_edges'], dtype=np.float64)
    x_bin_idx = np.clip(
        np.searchsorted(x_edges, xa, side='right') - 1, 0, W - 1,
    )
    y_bin_idx = np.clip(
        np.searchsorted(y_edges, ya, side='right') - 1, 0, H - 1,
    )
    cell_cluster = cluster_grid[x_bin_idx, y_bin_idx] - 1   # 0-based

    x_render_min = float(x_edges[0])
    x_render_max = float(x_edges[-1])
    y_render_min = float(y_edges[0])
    y_render_max = float(y_edges[-1])

    clouds: list[dict] = []
    for new_id, pi in enumerate(survivor_indices):
        peak_rec = tracked_peaks[pi]
        wb, hb = peak_rec['x_bin'], peak_rec['y_bin']
        peak_x = float(0.5 * (x_edges[wb] + x_edges[wb + 1]))
        peak_y = float(0.5 * (y_edges[hb] + y_edges[hb + 1]))

        cell_mask = cell_cluster == new_id
        n_in = int(cell_mask.sum())
        frac = (n_in / float(n_total)) if n_total > 0 else 0.0
        if n_in > 0:
            x_vals = xa[cell_mask]
            y_vals = ya[cell_mask]
            x_mean = float(x_vals.mean())
            x_std = float(x_vals.std())
            y_mean = float(y_vals.mean())
            y_std = float(y_vals.std())
        else:
            x_vals = np.empty(0, dtype=np.float64)
            y_vals = np.empty(0, dtype=np.float64)
            x_mean, y_mean = peak_x, peak_y
            x_std = y_std = 0.0

        ws_, hs_ = np.where(cluster_grid == new_id + 1)
        if ws_.size > 0:
            bbox = [
                float(x_edges[int(ws_.min())]),
                float(x_edges[int(ws_.max()) + 1]),
                float(y_edges[int(hs_.min())]),
                float(y_edges[int(hs_.max()) + 1]),
            ]
        else:
            bbox = [x_render_min, x_render_min,
                    y_render_min, y_render_min]

        birth_level = int(peak_rec['birth_level'])
        # death_level None ⇒ the peak stayed isolated until k=1; report
        # 1 (= bottom of the walk) so persistence is well-defined.
        death_level = int(peak_rec['death_level']
                          if peak_rec['death_level'] is not None else 1)
        persistence = birth_level - death_level + 1
        clouds.append({
            'idx':              new_id,
            'label':            (chr(ord('A') + new_id) if new_id < 26
                                 else f'C{new_id}'),
            'peak':             {'x':     peak_x,
                                 'y':     peak_y,
                                 'x_bin': wb,
                                 'y_bin': hb},
            'birth_level':      birth_level,
            'death_level':      death_level,
            'persistence':      persistence,
            'max_cells_alone':  int(peak_rec['max_cells_alone']),
            'n':                n_in,
            'frac':             frac,
            'mean':             [x_mean, y_mean],
            'std':              [x_std, y_std],
            'bbox':             bbox,
            'x_histogram':      _axis_histogram(x_vals, x_render_min,
                                                x_render_max, _HIST_BINS_RENDER),
            'y_histogram':      _axis_histogram(y_vals, y_render_min,
                                                y_render_max, _HIST_BINS_RENDER),
        })

    overlay: list[list[str]] = [['.' for _ in range(H)] for _ in range(W)]
    for w in range(W):
        for h in range(H):
            cid = int(cluster_grid[w, h]) - 1
            if 0 <= cid < len(clouds):
                overlay[w][h] = clouds[cid]['label']

    return {
        'ok':           True,
        'n_clouds':     len(clouds),
        'clouds':       clouds,
        'overlay':      overlay,
        'grid_w':       int(W),
        'grid_h':       int(H),
        'x_edges':      x_edges.tolist(),
        'y_edges':      y_edges.tolist(),
        'n_outside':    0,
        'frac_outside': 0.0,
        'diffuse':      None,
        'params':       params,
    }


# ── Rendering ──────────────────────────────────────────────────────────────

def render_density_clouds(
    x_marker: str, y_marker: str, info: dict,
) -> str:
    """Compact text block: cloud-overlay heatmap + per-cloud stats.

    Empty string when ``n_clouds <= 1`` (single-cloud or none ⇒ the
    global axis / joint distribution blocks already cover that case).
    """
    if not info or not info.get('ok'):
        return ''
    clouds = list(info.get('clouds') or [])
    K = len(clouds)
    if K <= 1:
        return ''

    overlay = info.get('overlay') or []
    if not overlay:
        return ''

    W = int(info.get('grid_w') or len(overlay))
    H = int(info.get('grid_h') or (len(overlay[0]) if overlay else 0))
    x_edges = info.get('x_edges') or []
    y_edges = info.get('y_edges') or []
    if W == 0 or H == 0 or not x_edges or not y_edges:
        return ''

    x_min, x_max = float(x_edges[0]), float(x_edges[-1])
    y_min, y_max = float(y_edges[0]), float(y_edges[-1])
    x_min_str, x_max_str = f'{x_min:.2f}', f'{x_max:.2f}'
    y_min_str, y_max_str = f'{y_min:.2f}', f'{y_max:.2f}'
    y_label_w = max(len(y_min_str), len(y_max_str), 4)

    lines: list[str] = [
        f'[Density-level cloud detection]   K={K} clouds',
        '  every bin is labelled with its cloud (watershed flood-fill '
        'from each peak; the parent population is fully partitioned).',
        '',
        ' ' * (y_label_w + 1) + y_marker,
    ]
    for hi in range(H - 1, -1, -1):
        ylab = ''
        if hi == H - 1:
            ylab = y_max_str
        elif hi == 0:
            ylab = y_min_str
        row = ''.join(overlay[wi][hi] for wi in range(W))
        lines.append(f' {ylab:>{y_label_w}} │{row}')
    lines.append(' ' * (y_label_w + 1) + '└' + '─' * W)

    indent = ' ' * (y_label_w + 2)
    x_tick_row = list(' ' * W)
    for k_, ch in enumerate(x_min_str):
        if k_ < W:
            x_tick_row[k_] = ch
    end_start = max(0, W - len(x_max_str))
    for k_, ch in enumerate(x_max_str):
        if end_start + k_ < W:
            x_tick_row[end_start + k_] = ch
    lines.append(indent + ''.join(x_tick_row) + '  ' + x_marker)

    for c in clouds:
        peak = c['peak']
        bbox = c['bbox']
        lines.append('')
        lines.append(
            f"[Cloud {c['label']}]   peak=({peak['x']:.2f}, {peak['y']:.2f})   "
            f"levels {c['birth_level']}↓{c['death_level']} "
            f"(persistence {c['persistence']})   "
            f"n={c['n']:,} ({c['frac'] * 100:.0f}%)"
        )
        lines.append(
            f"  bbox: x ∈ [{bbox[0]:.2f}, {bbox[1]:.2f}], "
            f"y ∈ [{bbox[2]:.2f}, {bbox[3]:.2f}]"
        )
        lines.extend(_render_axis_block(
            x_marker, 'x', c['mean'][0], c['std'][0], c['x_histogram'],
        ))
        lines.extend(_render_axis_block(
            y_marker, 'y', c['mean'][1], c['std'][1], c['y_histogram'],
        ))

    return '\n'.join(lines)


# ── Per-axis block helper ──────────────────────────────────────────────────

def _render_axis_block(
    marker: str, role: str, mu: float, sigma: float, hist: dict,
) -> list[str]:
    """Single-cloud (or diffuse) per-axis block: stats line + bar +
    annotation rows for ``↑μ``, ``↑L = μ - 2σ``, ``↑U = μ + 2σ``.

    Markers that fall outside the rendered scale are dropped from the bar
    annotation, with a ``(L clipped)`` / ``(U clipped)`` note inline in
    the stats line so the LLM still sees the numeric value.
    """
    counts = list(hist.get('counts') or [])
    edges = list(hist.get('edges') or [])
    n_bins = int(hist.get('n_bins') or len(counts))
    if not counts or not edges:
        return []
    vmin = float(edges[0])
    vmax = float(edges[-1])
    decimals = 3 if abs(vmax - vmin) < 1.0 else 2

    lo = mu - 2.0 * sigma
    hi = mu + 2.0 * sigma
    lo_clip = lo < vmin
    hi_clip = hi > vmax
    note = ''
    if lo_clip and hi_clip:
        note = '   (L,U clipped)'
    elif lo_clip:
        note = '   (L clipped)'
    elif hi_clip:
        note = '   (U clipped)'

    role_label = 'x' if role == 'x' else 'y'
    stats_line = (
        f'  {marker} ({role_label}):  μ={_fmt(mu, decimals)}   '
        f'σ={_fmt(sigma, decimals)}   '
        f'2σ=[{_fmt(lo, decimals)}, {_fmt(hi, decimals)}]'
        + note
    )

    vmin_str = _fmt(vmin, decimals)
    vmax_str = _fmt(vmax, decimals)
    bar = _density_bar(counts, scale='linear')
    bar_line = f'    {vmin_str} {bar} {vmax_str}'
    ann_indent = '    ' + ' ' * len(vmin_str) + ' '

    marks: list[tuple[int, str]] = [
        (_bin_index_for_value(edges, mu), '↑μ'),
    ]
    if not lo_clip:
        marks.append((_bin_index_for_value(edges, lo), '↑L'))
    if not hi_clip:
        marks.append((_bin_index_for_value(edges, hi), '↑U'))

    ann_rows = _annotation_rows(n_bins, marks, [])
    return [stats_line, bar_line] + [ann_indent + r for r in ann_rows]
