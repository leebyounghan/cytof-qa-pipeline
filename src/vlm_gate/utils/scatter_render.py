#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decorated scatter renderer for the vlm-gate user message.

The raw ``scatter_bg.png`` produced by ``src.vlm_gate.task`` is
decoration-free by design — it exists as the ``imshow`` substrate for
overlay tools (in ``src.agent_gate``). For a vision LM, the model
needs to READ COORDINATES off the picture, so any scatter shown to the
LLM must carry axis ticks + labels. This module wraps ``scatter_bg.png``
with the decoration the model sees, and returns a base64 ``data:`` URI
ready to drop into a Chat Completions ``image_url`` content block.

Independent copy of the ``render_bare_scatter`` helper that lives in
``src.agent_gate.tools.render_overlay``. Duplicated rather than
imported so ``src.vlm_gate`` carries no dependency on ``src.agent_gate``.
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use('Agg')  # headless, thread-safe rasteriser
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


_FIGSIZE = (7.0, 6.0)   # inches
_DPI = 100              # → 700×600 px — modest, image-tile budget aware
_PAD_FRAC = 0.06        # axis padding beyond data extent


def render_input_scatter(
    *,
    scatter_bg_path: str | None,
    scatter_extent: list[float],
    x_marker: str,
    y_marker: str,
    step_label: str = '',
) -> str:
    """Decorated scatter (axes + ticks + grid). Returns ``data:`` URI.

    Constructs ``Figure`` + ``FigureCanvasAgg`` directly (bypasses
    pyplot) so it stays thread-safe under ``asyncio.to_thread``.
    """
    xmn, xmx, ymn, ymx = scatter_extent

    fig = Figure(figsize=_FIGSIZE, dpi=_DPI)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    bg_ok = False
    if scatter_bg_path:
        try:
            bg = mpimg.imread(scatter_bg_path)
            # Background row 0 = high y (saved with y increasing upward,
            # axes filling the figure). origin='upper' + extent maps row
            # 0 → ymax. aspect='auto' matches the bg's independent x/y
            # scaling.
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
    title = 'Parent population — read coordinates off the axes to gate'
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
