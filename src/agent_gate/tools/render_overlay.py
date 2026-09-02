#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``render_gate_overlay`` — the agent's one tool.

The agent proposes per-category gates, calls this tool, and gets back a
picture of those boxes drawn on the parent population's scatter plus a
text read-back of how propagation will interpret them. It then *sees*
whether a box sits on the right cells and revises.

Why a pre-rendered background:
``src.agent_gate.task`` renders the (slow, 60k-point) parent scatter
ONCE as a decoration-free ``scatter_bg.png`` whose pixel bounds map
exactly to ``scatter_extent`` ``[xmin, xmax, ymin, ymax]``. This tool
just loads that PNG with ``imshow(extent=...)`` and overlays the gate
boxes in **data coordinates** — matplotlib does the data→pixel mapping,
so the overlay lands on the right cells as long as the same extent is
reused. Cheap enough to run every agent turn.

The image is returned as a base64 ``data:`` URI; the runner forwards it
to the model in a follow-up ``user`` message (the OpenAI ``tool`` role
cannot carry images).
"""

from __future__ import annotations

import base64
import io

# Bypass pyplot entirely: ``run_agent`` dispatches this in
# ``asyncio.to_thread`` with up to ``--concurrency`` threads in flight,
# and pyplot's global figure manager (``_pylab_helpers.Gcf``) is not
# thread-safe even with the Agg backend. Construct ``Figure`` objects
# directly and rasterise via ``FigureCanvasAgg`` — both are pure-Python
# objects with no shared state.
import matplotlib  # noqa: E402
matplotlib.use('Agg')  # belt-and-braces — also pins the rasteriser
import matplotlib as mpl  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.agent_gate.inference.output_schema import (  # noqa: E402
    _parse_single_gate, is_absent_gate,
)


_OVERLAY_FIGSIZE = (7.0, 6.0)   # inches
_OVERLAY_DPI = 100              # → 700×600 px — modest, image-tile budget aware
_PAD_FRAC = 0.06                # axis padding beyond data extent so a gate
                                # that overshoots the data is still visible


# ── gate geometry helpers ───────────────────────────────────────────────────

def _gate_bbox(gate: dict) -> tuple[float, float, float, float] | None:
    """Axis-aligned bounding box ``(x_min, x_max, y_min, y_max)`` of a
    parsed rectangle or polygon gate; ``None`` for an absent sentinel."""
    if is_absent_gate(gate):
        return None
    if 'vertices' in gate:
        v = np.asarray(gate['vertices'], dtype=np.float64)
        return float(v[:, 0].min()), float(v[:, 0].max()), \
            float(v[:, 1].min()), float(v[:, 1].max())
    return (gate['x_min'], gate['x_max'], gate['y_min'], gate['y_max'])


def _rect_area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[1] - b[0]) * max(0.0, b[3] - b[2])


def _rect_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ox = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return ox * oy


def _draw_gate(ax, gate: dict, color, label: str | None) -> None:
    """Draw one parsed gate (rectangle or polygon) as a dashed outline."""
    if 'vertices' in gate:
        verts = np.asarray(gate['vertices'] + [gate['vertices'][0]],
                           dtype=np.float64)
        ax.plot(verts[:, 0], verts[:, 1], color=color, lw=2.0, ls='--')
        lx, ly = verts[0]
    else:
        ax.add_patch(Rectangle(
            (gate['x_min'], gate['y_min']),
            gate['x_max'] - gate['x_min'],
            gate['y_max'] - gate['y_min'],
            fill=False, edgecolor=color, lw=2.0, ls='--',
        ))
        lx, ly = gate['x_min'], gate['y_max']
    if label:
        ax.text(lx, ly, f' {label}', color=color, fontsize=8,
                va='bottom', ha='left', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec=color, lw=0.6, alpha=0.85))


# ── diagnostics ─────────────────────────────────────────────────────────────

def _fmt_gate(gate: dict) -> str:
    if 'vertices' in gate:
        n = len(gate['vertices'])
        b = _gate_bbox(gate)
        return (f'polygon ({n} verts) bbox x∈[{b[0]:.2f}, {b[1]:.2f}] '
                f'y∈[{b[2]:.2f}, {b[3]:.2f}]')
    return (f'rectangle x∈[{gate["x_min"]:.2f}, {gate["x_max"]:.2f}] '
            f'y∈[{gate["y_min"]:.2f}, {gate["y_max"]:.2f}]  '
            f'area={_rect_area(_gate_bbox(gate)):.2f}')


def _build_diagnostics(
    parsed: dict[str, list[dict] | dict | None],
    raw_gates: dict,
    targets: list[str],
    extent: list[float],
) -> str:
    """Read-back of how propagation will see each category + warnings.

    ``parsed`` maps category → list of real gates, a single absent
    sentinel, or ``None`` (entry present but unparseable / degenerate).
    A category missing from ``parsed`` had no entry at all.
    """
    xmn, xmx, ymn, ymx = extent
    lines: list[str] = ['Parsed gates (exactly as propagation will see them):']
    warnings: list[str] = []
    no_entry: list[str] = []
    bboxes: dict[str, list[tuple]] = {}

    for cat in targets:
        if cat not in raw_gates:
            no_entry.append(cat)
            continue
        val = parsed.get(cat)
        if val is None:
            lines.append(f'  {cat}: INVALID — entry present but not a valid '
                         f'gate; dropped at propagation, its cells → Unassigned')
            continue
        if isinstance(val, dict) and is_absent_gate(val):
            lines.append(f'  {cat}: absent — declared not present, no box drawn')
            continue
        gate_list = val if isinstance(val, list) else [val]
        descs = [_fmt_gate(g) for g in gate_list]
        joined = descs[0] if len(descs) == 1 else (
            f'{len(descs)} rectangles (union): ' + '; '.join(descs))
        lines.append(f'  {cat}: {joined}')
        bb = []
        for g in gate_list:
            box = _gate_bbox(g)
            if box is None:
                continue
            bb.append(box)
            # out-of-data-range check
            if box[0] < xmn - 1e-9 or box[1] > xmx + 1e-9:
                warnings.append(
                    f'{cat} extends beyond the x data range '
                    f'[{xmn:.2f}, {xmx:.2f}] — x∈[{box[0]:.2f}, {box[1]:.2f}]')
            if box[2] < ymn - 1e-9 or box[3] > ymx + 1e-9:
                warnings.append(
                    f'{cat} extends beyond the y data range '
                    f'[{ymn:.2f}, {ymx:.2f}] — y∈[{box[2]:.2f}, {box[3]:.2f}]')
        bboxes[cat] = bb

    # pairwise overlap (rectangles only)
    cats = list(bboxes)
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            ov = 0.0
            for ba in bboxes[cats[i]]:
                for bb in bboxes[cats[j]]:
                    ov += _rect_overlap_area(ba, bb)
            if ov > 1e-9:
                warnings.append(
                    f'{cats[i]} ∩ {cats[j]} overlap (area≈{ov:.2f}) — '
                    f'cells in the intersection go to the smallest-area gate')

    if no_entry:
        lines.append(f'No entry for: {", ".join(no_entry)} '
                     f'(those cells → Unassigned)')
    if warnings:
        lines.append('Warnings:')
        lines.extend(f'  - {w}' for w in warnings)
    else:
        lines.append('Warnings: none — gates are within range and non-overlapping.')
    return '\n'.join(lines)


# ── main entry point ────────────────────────────────────────────────────────

def render_gate_overlay(
    gates: dict,
    *,
    scatter_bg_path: str | None,
    scatter_extent: list[float],
    x_marker: str,
    y_marker: str,
    options: list[str],
    step_label: str = '',
) -> tuple[str, str]:
    """Draw ``gates`` over the parent scatter and read them back.

    Args:
        gates: the model's raw tool argument — ``{category: gate | [gate,
            ...] | {"absent": true, ...}}``. Parsed through the same
            validator propagation uses, so the read-back is honest.
        scatter_bg_path: path to ``scatter_bg.png`` from
            ``src.agent_gate.task`` (may be missing → blank background).
        scatter_extent: ``[xmin, xmax, ymin, ymax]`` the background PNG
            was rendered with — reused verbatim for ``imshow``.
        x_marker / y_marker: axis labels.
        options: category list incl. ``Unassigned`` — fixes per-category
            colours and the set of categories validated.
        step_label: optional title annotation.

    Returns:
        ``(data_uri, diagnostics_text)`` — a base64 ``data:image/png``
        URI and the text read-back. The runner puts the text in the
        ``tool`` message and the image in a follow-up ``user`` message.
    """
    xmn, xmx, ymn, ymx = scatter_extent
    targets = [c for c in options if c != 'Unassigned']

    # Parse every entry through the propagation validator so what we draw
    # and read back is exactly what propagate.py would act on.
    parsed: dict[str, list[dict] | dict | None] = {}
    for cat in targets:
        if cat not in gates:
            continue
        entry = gates[cat]
        if isinstance(entry, list):
            sub = [_parse_single_gate(e) for e in entry]
            real = [g for g in sub if g is not None and not is_absent_gate(g)]
            if real:
                parsed[cat] = real
            else:
                absent = [g for g in sub if g is not None and is_absent_gate(g)]
                parsed[cat] = absent[0] if absent else None
        elif isinstance(entry, dict):
            parsed[cat] = _parse_single_gate(entry)
        else:
            parsed[cat] = None

    # ── figure ──────────────────────────────────────────────────────────
    fig = Figure(figsize=_OVERLAY_FIGSIZE, dpi=_OVERLAY_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    bg_ok = False
    if scatter_bg_path:
        try:
            bg = mpimg.imread(scatter_bg_path)
            # Background row 0 = high y (saved with y increasing upward,
            # axes filling the figure). origin='upper' + extent maps row 0
            # → ymax, so the scatter lands at the right y. aspect='auto'
            # matches the background's independent x/y scaling.
            ax.imshow(bg, extent=[xmn, xmx, ymn, ymx],
                      origin='upper', aspect='auto', interpolation='nearest')
            bg_ok = True
        except (FileNotFoundError, OSError, ValueError):
            bg_ok = False

    cmap = mpl.colormaps['tab10']
    color_map = {c: cmap(i % 10) for i, c in enumerate(targets)}

    drawn_any = False
    for cat in targets:
        val = parsed.get(cat)
        if val is None or (isinstance(val, dict) and is_absent_gate(val)):
            continue
        gate_list = val if isinstance(val, list) else [val]
        for k, g in enumerate(gate_list):
            _draw_gate(ax, g, color_map[cat],
                       label=cat if k == 0 else None)
            drawn_any = True

    # padded view so overshooting gates remain visible
    px = (xmx - xmn) * _PAD_FRAC or 0.5
    py = (ymx - ymn) * _PAD_FRAC or 0.5
    ax.set_xlim(xmn - px, xmx + px)
    ax.set_ylim(ymn - py, ymx + py)
    ax.set_xlabel(x_marker, fontsize=10)
    ax.set_ylabel(y_marker, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    title = 'Proposed gates on parent population'
    if step_label:
        title = f'{step_label} — {title.lower()}'
    if not bg_ok:
        title += '  [background scatter unavailable]'
    if not drawn_any:
        title += '  [no valid gate to draw]'
    ax.set_title(title, fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    canvas.print_png(buf)  # no plt.close — fig is dropped on scope exit
    buf.seek(0)
    data_uri = ('data:image/png;base64,'
                + base64.b64encode(buf.read()).decode('ascii'))

    diagnostics = _build_diagnostics(parsed, gates, targets, scatter_extent)
    if not bg_ok:
        diagnostics = ('[warning] background scatter PNG was unavailable — '
                       'boxes are drawn on a blank plane; positions are still '
                       'to scale.\n' + diagnostics)
    return data_uri, diagnostics


# ── bare scatter (initial user-message attachment + empty-gates response) ──

def render_bare_scatter(
    *,
    scatter_bg_path: str | None,
    scatter_extent: list[float],
    x_marker: str,
    y_marker: str,
    step_label: str = '',
) -> str:
    """Decorated scatter with no gate overlay. Returns a base64 ``data:`` URI.

    The raw ``scatter_bg.png`` is decoration-free by design — it exists
    only as the substrate for ``imshow(extent=…)`` so overlay box
    coordinates stay honest. Anything the model SEES, on the other
    hand, must carry axis ticks + labels so it can read coordinates;
    every scatter shown to the LLM goes through this wrapper.

    Used by :func:`run_agent` for the initial user-message attachment
    and as the response when the model calls the tool with empty /
    missing ``gates`` (graceful peek instead of an error).
    """
    xmn, xmx, ymn, ymx = scatter_extent

    fig = Figure(figsize=_OVERLAY_FIGSIZE, dpi=_OVERLAY_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    bg_ok = False
    if scatter_bg_path:
        try:
            bg = mpimg.imread(scatter_bg_path)
            ax.imshow(bg, extent=[xmn, xmx, ymn, ymx],
                      origin='upper', aspect='auto', interpolation='nearest')
            bg_ok = True
        except (FileNotFoundError, OSError, ValueError):
            bg_ok = False

    px = (xmx - xmn) * _PAD_FRAC or 0.5
    py = (ymx - ymn) * _PAD_FRAC or 0.5
    ax.set_xlim(xmn - px, xmx + px)
    ax.set_ylim(ymn - py, ymx + py)
    ax.set_xlabel(x_marker, fontsize=10)
    ax.set_ylabel(y_marker, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    title = 'Parent population — propose your gates from this view'
    if step_label:
        title = f'{step_label} — {title.lower()}'
    if not bg_ok:
        title += '  [bg scatter unavailable]'
    ax.set_title(title, fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    return ('data:image/png;base64,'
            + base64.b64encode(buf.read()).decode('ascii'))


# ── trajectory composite (post-loop, human inspection) ──────────────────────

def _style_for_call(call_idx: int, n_calls: int) -> tuple[float, float, object]:
    """``(alpha, lw, linestyle)`` for the call at ``call_idx`` of ``n_calls``.

    Final call always solid + bold so the committed proposal pops; earlier
    calls fade toward dotted+thin so the trajectory direction is obvious.
    """
    if call_idx == n_calls - 1:
        return 1.0, 2.4, '-'  # final: solid bold
    progress = call_idx / max(n_calls - 1, 1)  # in [0, 1)
    alpha = 0.30 + 0.40 * progress
    lw    = 1.2 + 0.8 * progress
    ls    = (0, (1, 2)) if progress < 0.5 else (0, (5, 3))
    return alpha, lw, ls


def _draw_traj_gate(ax, gate: dict, color, alpha, lw, dash) -> None:
    if 'vertices' in gate:
        verts = np.asarray(gate['vertices'] + [gate['vertices'][0]],
                           dtype=np.float64)
        ax.plot(verts[:, 0], verts[:, 1], color=color, lw=lw,
                ls=dash, alpha=alpha)
    else:
        ax.add_patch(Rectangle(
            (gate['x_min'], gate['y_min']),
            gate['x_max'] - gate['x_min'],
            gate['y_max'] - gate['y_min'],
            fill=False, edgecolor=color, lw=lw, ls=dash, alpha=alpha,
        ))


def render_gate_trajectory(
    per_call_gates: list,
    out_path: str,
    *,
    scatter_bg_path: str | None,
    scatter_extent: list[float],
    x_marker: str,
    y_marker: str,
    options: list[str],
    step_label: str = '',
    final_gates: dict | None = None,
) -> None:
    """Composite plot — every gate proposal the model actually made.

    Empty / null tool calls (the "peek" pattern — see
    :func:`src.agent_gate.tools.registry.dispatch_tool_call`) are
    filtered out: they were re-orientation moments, not proposals, and
    cluttering the trajectory with empty legend rows hides the actual
    progression.

    Args:
        per_call_gates: the raw ``gates`` arg from each tool call,
            ordered chronologically (earliest first). ``None`` /
            non-dict / empty entries are dropped.
        final_gates: the model's COMMITTED final answer (parsed via
            ``extract_gates``). When given, appended as the final entry
            in the trajectory — captures revisions that happened
            *between* the last tool call and the final text turn,
            which would otherwise be invisible.
        scatter_bg_path / scatter_extent / x_marker / y_marker /
            options / step_label: same as :func:`render_gate_overlay`.

    Style: faintest dotted = earliest proposal, solid bold = "FINAL"
    (the committed answer if ``final_gates`` was passed; otherwise the
    last real tool-call proposal). No return — writes ``out_path``.
    Goes through ``Figure`` + ``FigureCanvasAgg`` so the call is
    thread-safe under :func:`asyncio.to_thread`.
    """
    xmn, xmx, ymn, ymx = scatter_extent
    targets = [c for c in options if c != 'Unassigned']

    # Filter to chronological list of (label, gates_dict) — only real
    # proposals end up here. The committed final answer, if any, sits
    # at the end and always wears the solid-bold style.
    seq: list[tuple[str, dict]] = []
    real_idx = 0
    for raw in per_call_gates:
        if isinstance(raw, dict) and raw:
            real_idx += 1
            seq.append((f'Proposal {real_idx}', raw))
    if final_gates:
        # Skip if final is byte-equal to the last proposal — then
        # the committed answer adds no new geometry, and labelling the
        # last entry as "FINAL" is more informative than duplicating.
        if seq and seq[-1][1] == final_gates:
            seq[-1] = ('FINAL (= last proposal)', seq[-1][1])
        else:
            seq.append(('FINAL (committed)', final_gates))

    n_seq = len(seq)

    fig = Figure(figsize=_OVERLAY_FIGSIZE, dpi=_OVERLAY_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    bg_ok = False
    if scatter_bg_path:
        try:
            bg = mpimg.imread(scatter_bg_path)
            ax.imshow(bg, extent=[xmn, xmx, ymn, ymx],
                      origin='upper', aspect='auto', interpolation='nearest')
            bg_ok = True
        except (FileNotFoundError, OSError, ValueError):
            bg_ok = False

    cmap = mpl.colormaps['tab10']
    color_map = {c: cmap(i % 10) for i, c in enumerate(targets)}

    for idx, (_label, raw_gates) in enumerate(seq):
        alpha, lw, dash = _style_for_call(idx, n_seq)
        for cat in targets:
            entry = raw_gates.get(cat)
            if entry is None:
                continue
            sub_list = entry if isinstance(entry, list) else [entry]
            for sub in sub_list:
                parsed = _parse_single_gate(sub) if isinstance(sub, dict) \
                    else None
                if parsed is None or is_absent_gate(parsed):
                    continue
                _draw_traj_gate(ax, parsed, color_map[cat], alpha, lw, dash)

    # legend stack: categories on the left (color), trajectory on the right (style)
    from matplotlib.lines import Line2D
    cat_handles = [Line2D([], [], color=color_map[c], lw=2.0, label=c)
                   for c in targets]
    seq_handles = []
    for idx, (label, _g) in enumerate(seq):
        alpha, lw, dash = _style_for_call(idx, n_seq)
        seq_handles.append(Line2D(
            [], [], color='black', lw=lw, ls=dash, alpha=alpha, label=label,
        ))
    leg1 = ax.legend(handles=cat_handles, fontsize=7, loc='upper left',
                     title='Category', title_fontsize=7, framealpha=0.85)
    ax.add_artist(leg1)
    if seq_handles:
        ax.legend(handles=seq_handles, fontsize=7, loc='upper right',
                  title='Trajectory', title_fontsize=7, framealpha=0.85)

    px = (xmx - xmn) * _PAD_FRAC or 0.5
    py = (ymx - ymn) * _PAD_FRAC or 0.5
    ax.set_xlim(xmn - px, xmx + px)
    ax.set_ylim(ymn - py, ymx + py)
    ax.set_xlabel(x_marker, fontsize=10)
    ax.set_ylabel(y_marker, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    n_skipped = sum(1 for raw in per_call_gates
                    if not (isinstance(raw, dict) and raw))
    note = f'  ({n_skipped} empty peek call(s) skipped)' if n_skipped else ''
    title = f'Gate trajectory — {n_seq} proposal(s){note}'
    if step_label:
        title = f'{step_label} — {title.lower()}'
    if not bg_ok:
        title += '  [bg scatter unavailable]'
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    canvas.print_png(out_path)

