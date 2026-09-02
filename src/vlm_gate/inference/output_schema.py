"""Schema and assemblers for the LLM-gate eval result JSON.

- ``extract_gates`` — parse LLM response text → ``{category: gate}`` dict.
- ``assemble_eval_output`` — turn (groups, raw_texts) into the
  hierarchical ``{meta, samples}`` JSON of *gates only* (no metric
  fields).
- ``iter_step_records`` / ``group_by_sample_step`` — flat readers over
  the hierarchical structure.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Iterator


_GATE_KEYS = ('x_min', 'x_max', 'y_min', 'y_max')


def _coerce_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_valid_gate(gate: dict) -> bool:
    """All four bounds present, finite, and forming a non-empty box."""
    try:
        xmn = float(gate['x_min'])
        xmx = float(gate['x_max'])
        ymn = float(gate['y_min'])
        ymx = float(gate['y_max'])
    except (KeyError, TypeError, ValueError):
        return False
    if any(v != v for v in (xmn, xmx, ymn, ymx)):  # NaN check
        return False
    return xmx > xmn and ymx > ymn


def _coerce_vertices(raw) -> list[list[float]] | None:
    """Coerce a vertex list to ``[[x,y], ...]`` of floats, or None."""
    if not isinstance(raw, list) or len(raw) < 3:
        return None
    out: list[list[float]] = []
    for pt in raw:
        if isinstance(pt, dict):
            x = _coerce_float(pt.get('x'))
            y = _coerce_float(pt.get('y'))
        elif isinstance(pt, (list, tuple)) and len(pt) == 2:
            x = _coerce_float(pt[0])
            y = _coerce_float(pt[1])
        else:
            return None
        if x is None or y is None or x != x or y != y:
            return None
        out.append([x, y])
    return out


def _polygon_area(vertices: list[list[float]]) -> float:
    """Absolute Shoelace area; 0 for degenerate polygons."""
    n = len(vertices)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _is_valid_quad_gate(gate: dict) -> bool:
    """Vertices present (3+ pts), finite, positive polygon area."""
    verts = gate.get('vertices')
    if not isinstance(verts, list) or len(verts) < 3:
        return False
    return _polygon_area(verts) > 0.0


def _is_absent_entry(entry: dict) -> bool:
    """True iff the dict declares the category absent (`absent: true`)."""
    if not isinstance(entry, dict):
        return False
    val = entry.get('absent')
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() == 'true'
    return False


def _parse_single_gate(entry: dict) -> dict | None:
    """Parse one gate dict → validated gate, absent sentinel, or None.

    Returns:
      - rectangle/polygon gate dict (valid box / valid polygon),
      - absent sentinel ``{"absent": True, "rationale": ...}`` when the
        LLM declares the category not present in the parent population,
      - ``None`` for parse failures / degenerate boxes.
    """
    if not isinstance(entry, dict):
        return None
    if _is_absent_entry(entry):
        return {'absent': True, 'rationale': entry.get('rationale')}
    if 'vertices' in entry:
        verts = _coerce_vertices(entry.get('vertices'))
        gate = {
            'vertices': verts,
            'rationale': entry.get('rationale'),
        }
        return gate if (verts is not None
                        and _is_valid_quad_gate(gate)) else None
    gate = {
        'x_min': _coerce_float(entry.get('x_min')),
        'x_max': _coerce_float(entry.get('x_max')),
        'y_min': _coerce_float(entry.get('y_min')),
        'y_max': _coerce_float(entry.get('y_max')),
        'rationale': entry.get('rationale'),
    }
    return gate if _is_valid_gate(gate) else None


def is_absent_gate(value) -> bool:
    """True iff a stored gate value is an absent sentinel (not a real gate)."""
    return (isinstance(value, dict)
            and bool(value.get('absent', False))
            and 'x_min' not in value
            and 'vertices' not in value)


def extract_gates(
    text: str,
    categories: list[str],
    *,
    strip_think: bool = False,
) -> dict:
    """Parse LLM response → ``{category: gate | list[gate] | None}``.

    Per category the LLM may emit:

    1. **A single gate dict** (axis-aligned rectangle or vertex polygon)::

           "Bcells": {"x_min": <f>, "x_max": <f>,
                      "y_min": <f>, "y_max": <f>, "rationale": "..."}

       Rectangle: ``{x_min, x_max, y_min, y_max}``.
       Polygon: ``{"vertices": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}``
       (3+ vertices; each vertex may also be ``{"x": ..., "y": ...}``).

    2. **A list of gate dicts** (multi-rectangle / multi-polygon)::

           "DCs": [
             {"x_min": 0.0, "x_max": 1.5, "y_min": 3.0, "y_max": 5.5},
             {"x_min": 2.5, "x_max": 4.7, "y_min": 0.0, "y_max": 3.0}
           ]

       The category's gate is the **union** of the listed rectangles —
       a cell is predicted as the category if it falls inside ANY of
       them. Used for diagonal-distributed categories (Q2+Q4 patterns)
       that a single rectangle cannot cleanly capture.

    3. **An absent sentinel** when the category is genuinely not present
       in this step's parent population::

           "RareSubset": {"absent": true, "rationale": "..."}

       Stored as ``{"absent": True, "rationale": ...}`` (use
       :func:`is_absent_gate` to check). Cells contribute no membership;
       they fall through to other gates or to ``Unassigned`` — same
       runtime effect as ``None``, but distinguishable from parse
       failure.

    JSON-extraction strategy depends on ``strip_think``:

    - ``strip_think=False`` (default — OpenAI-class single-shot models):
      single fenced ```json``` block, falling back to a bare ``{...}``.

    - ``strip_think=True`` (open-source thinking models — Qwen3 with
      reasoning_parser, DeepSeek-V4, …): a 4-step robust fallback that
      (1) slices past the last ``</think>`` close tag if present
      (defends against pseudo-JSON inside thinking sections that
      confuses the greedy match), (2) tries fenced blocks
      **last-to-first** (thinking models often emit drafts before the
      final answer), (3) falls back to bare ``{...}`` within the
      post-think slice, (4) retries on full text if the post-think
      slice yielded nothing.

    Categories without a valid gate (missing, unparsable, all entries
    degenerate) are recorded as ``None``. ``Unassigned`` is the residual
    and is filtered out.
    """
    targets = [c for c in categories if c != 'Unassigned']
    empty = {c: None for c in targets}

    if strip_think:
        think_end = text.rfind('</think>')
        search_text = (text[think_end + len('</think>'):]
                       if think_end >= 0 else text)
        raw = None
        matches = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```',
                             search_text, re.DOTALL)
        for json_str in reversed(matches):
            try:
                raw = json.loads(json_str)
                break
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        if raw is None:
            match = re.search(r'\{[\s\S]*\}', search_text)
            if match:
                try:
                    raw = json.loads(match.group(0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    raw = None
        if raw is None and think_end >= 0:
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    raw = json.loads(match.group(0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    raw = None
        if raw is None:
            return empty
    else:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r'\{[\s\S]*\}', text)
            if not match:
                return empty
            json_str = match.group(0)
        try:
            raw = json.loads(json_str)
        except (json.JSONDecodeError, TypeError, ValueError):
            return empty

    result: dict = {}
    for c in targets:
        entry = raw.get(c)
        if isinstance(entry, list):
            parsed_all = [_parse_single_gate(sub) for sub in entry]
            real_gates = [g for g in parsed_all
                          if g is not None and not is_absent_gate(g)]
            if real_gates:
                result[c] = real_gates
                continue
            absent_hits = [g for g in parsed_all if g is not None
                           and is_absent_gate(g)]
            if absent_hits:
                result[c] = absent_hits[0]
                continue
            result[c] = None
            continue
        if isinstance(entry, dict):
            result[c] = _parse_single_gate(entry)
            continue
        result[c] = None
    return result


def assemble_eval_output(
    groups: list[dict],
    raw_texts: list[str],
    model: str,
    ablation: str,
    dataset_path: str,
    perturbation: str | None = None,
    save_raw: bool = False,
    reasoning_texts: list[str] | None = None,
    strip_think: bool = False,
) -> dict:
    """Build the hierarchical pred JSON from groups + LLM responses.

    Schema::

        {
          "meta": {model, ablation, dataset_path, n_samples, n_steps},
          "samples": {
            <sample>: {
              "steps": {
                <step>: {
                  "x_marker": str, "y_marker": str, "options": [...],
                  "gates": {
                    <category>: {x_min, x_max, y_min, y_max, rationale} | None
                  }
                }
              }
            }
          }
        }

    ``strip_think`` is forwarded to :func:`extract_gates` — set True for
    open-source thinking models so the parser slices past `</think>`
    and prefers the last fenced JSON block over earlier drafts.

    When ``save_raw`` is true, each step also carries the raw model
    text (``raw``) and, if ``reasoning_texts`` is provided, the
    server-stripped thinking trace (``reasoning``).
    """
    samples: dict = {}
    reasoning_iter = (reasoning_texts if reasoning_texts is not None
                      else [''] * len(raw_texts))
    for group, text, reasoning in zip(groups, raw_texts, reasoning_iter):
        gates = extract_gates(text, group['options'], strip_think=strip_think)
        sample_entry = samples.setdefault(group['sample'], {'steps': {}})
        step_entry = {
            'x_marker': group.get('x_marker'),
            'y_marker': group.get('y_marker'),
            'options':  group['options'],
            'gates':    gates,
        }
        if save_raw:
            step_entry['raw'] = text
            if reasoning:
                step_entry['reasoning'] = reasoning
        sample_entry['steps'][str(group['step'])] = step_entry

    sample_list = sorted(samples.keys())
    step_set = {int(s) for sample in samples.values()
                for s in sample['steps'].keys()}
    meta = {
        'model':        model,
        'ablation':     ablation,
        'dataset_path': dataset_path,
        'n_samples':    len(sample_list),
        'n_steps':      len(step_set),
    }
    if perturbation is not None:
        meta['perturbation'] = perturbation
    return {'meta': meta, 'samples': samples}


def iter_step_records(eval_data) -> Iterator[dict]:
    """Yield flat per-step records from the hierarchical dict."""
    for sample_name, sample in eval_data.get('samples', {}).items():
        for step_str, step in sample.get('steps', {}).items():
            yield {
                'sample':   sample_name,
                'step':     int(step_str),
                'x_marker': step.get('x_marker'),
                'y_marker': step.get('y_marker'),
                'options':  step.get('options', []),
                'gates':    step.get('gates', {}),
            }


def group_by_sample_step(eval_data) -> dict[tuple[str, int], dict]:
    """(sample, step) → step record."""
    out: dict[tuple[str, int], dict] = {}
    for r in iter_step_records(eval_data):
        out[(r['sample'], int(r['step']))] = r
    return out
