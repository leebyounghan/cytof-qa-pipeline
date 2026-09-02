#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool registry for the agent-gate runner.

One tool so far — ``render_gate_overlay`` — but the registry is the
single place the runner imports: ``get_tool_schemas()`` for the OpenAI
``tools=[...]`` array and ``dispatch_tool_call()`` to execute a call the
model made. Adding a tool later means adding a schema here and a branch
in :func:`dispatch_tool_call`; the runner does not change.
"""

from __future__ import annotations

import json

from src.agent_gate.tools.render_overlay import (
    render_bare_scatter, render_gate_overlay,
)


OVERLAY_TOOL_NAME = 'render_gate_overlay'


_GATES_DESC = (
    'Per-category gate proposal — a JSON object keyed by category name '
    '(the named categories of this step, NOT including "Unassigned"). '
    'Each value is one of: '
    '(1) a rectangle {"x_min","x_max","y_min","y_max","rationale"}; '
    '(2) a list of rectangle dicts (union — only for a category split '
    'across spatially-disconnected sub-clusters); '
    '(3) a polygon {"vertices":[[x,y],...],"rationale"} with 3+ vertices; '
    '(4) an absent sentinel {"absent": true, "rationale": "..."} when the '
    'category is genuinely not present in this parent population. '
    'Coordinates are in the SAME numeric units as the [Axis distribution] '
    'block in the prompt. You may pass a partial object (only the '
    'categories you have decided so far) — categories you omit are simply '
    'not drawn.'
)


def get_tool_schemas() -> list[dict]:
    """OpenAI Chat Completions ``tools`` array for the agent-gate runner."""
    return [
        {
            'type': 'function',
            'function': {
                'name': OVERLAY_TOOL_NAME,
                'description': (
                    'Draw your proposed gates as dashed boxes on the parent '
                    "population's scatter plot and return the rendered image "
                    'plus a text read-back of how each gate will be '
                    'interpreted at propagation time (parsed coordinates, '
                    'out-of-range warnings, overlap warnings). Use this to '
                    'VISUALLY CHECK whether each box actually sits on the '
                    'cluster you intend before committing — call it as many '
                    'times as needed to refine the boxes.'
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'gates': {
                            'type': 'object',
                            'description': _GATES_DESC,
                            'additionalProperties': True,
                        },
                    },
                    'required': ['gates'],
                },
            },
        },
    ]


def dispatch_tool_call(tool_call, ctx: dict) -> tuple[str, str | None]:
    """Execute one model tool call.

    Args:
        tool_call: an OpenAI tool-call object (``.function.name``,
            ``.function.arguments`` JSON string, ``.id``).
        ctx: per-(sample, step) context the model does not pass —
            ``scatter_bg_path``, ``scatter_extent``, ``x_marker``,
            ``y_marker``, ``options``, ``step_label``.

    Returns:
        ``(text, image_data_uri | None)``. ``text`` goes into the
        ``tool`` role message; when an image is produced the runner
        forwards it in a follow-up ``user`` message (the ``tool`` role
        cannot carry images on the Chat Completions API).
    """
    name = tool_call.function.name
    if name != OVERLAY_TOOL_NAME:
        return (f'[error] unknown tool {name!r}. Available: '
                f'{OVERLAY_TOOL_NAME}.', None)

    try:
        args = json.loads(tool_call.function.arguments or '{}')
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return (f'[error] could not parse tool arguments as JSON: {e}. '
                f'Re-call {OVERLAY_TOOL_NAME} with a valid JSON object.',
                None)

    gates = args.get('gates')
    # Treat empty / null / missing ``gates`` as a "show me the canvas"
    # request rather than an error — the model already has a decorated
    # scatter from turn 1, but it may want to re-anchor before
    # proposing. Returns the same bare scatter the user message carries
    # plus a nudge to actually call again WITH gates.
    if not isinstance(gates, dict) or not gates:
        try:
            data_uri = render_bare_scatter(
                scatter_bg_path=ctx.get('scatter_bg_path'),
                scatter_extent=ctx['scatter_extent'],
                x_marker=ctx.get('x_marker', 'x'),
                y_marker=ctx.get('y_marker', 'y'),
                step_label=ctx.get('step_label', ''),
            )
        except Exception as e:  # noqa: BLE001
            return (f'[error] render_bare_scatter failed: '
                    f'{type(e).__name__}: {e}', None)
        text = ('You called the tool without any gate proposal — here is the '
                'parent scatter again with axes for re-orientation. Re-call '
                f'{OVERLAY_TOOL_NAME} with a real `gates` object to verify '
                'your boxes, or output the final JSON answer directly.')
        return text, data_uri

    try:
        data_uri, diagnostics = render_gate_overlay(
            gates,
            scatter_bg_path=ctx.get('scatter_bg_path'),
            scatter_extent=ctx['scatter_extent'],
            x_marker=ctx.get('x_marker', 'x'),
            y_marker=ctx.get('y_marker', 'y'),
            options=ctx['options'],
            step_label=ctx.get('step_label', ''),
        )
    except Exception as e:  # noqa: BLE001 — tool errors must not crash the loop
        return (f'[error] render_gate_overlay failed: {type(e).__name__}: {e}',
                None)

    text = (diagnostics
            + '\n\nThe overlay image follows in the next message. '
            'Inspect each box against the cell density, then either call '
            f'{OVERLAY_TOOL_NAME} again with revised gates or output your '
            'final JSON answer.')
    return text, data_uri
