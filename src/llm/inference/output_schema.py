"""Schema and assemblers for the LLM eval result JSON.

- ``extract_cluster_answer`` — parse LLM response text → per-cell dict.
- ``assemble_eval_output`` — turn (groups, raw_texts) into the hierarchical
  ``{meta, samples}`` JSON of *predictions only* (no metric fields).
- ``iter_anchor_records`` / ``group_by_sample_step`` — flat readers over
  the hierarchical structure.

Anchor-level metric computation lives in
``src.llm.postprocess.anchor_metric`` and writes to its own
``pred_anchor_metrics.json`` file — pred.json stays prediction-only.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Iterator


def _iter_balanced_json_objects(text: str):
    """Yield every balanced ``{...}`` substring that parses as JSON.

    Brace-balanced scanner that respects string literals (so braces inside
    quoted rationales don't break the count). Robust to thinking-tag noise
    around the answer and to multiple JSON-ish snippets in the text.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] != '{':
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            ch = text[j]
            if in_str:
                if esc:           esc = False
                elif ch == '\\':  esc = True
                elif ch == '"':   in_str = False
            else:
                if ch == '"':     in_str = True
                elif ch == '{':   depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            yield json.loads(text[i:j + 1])
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
            j += 1
        i = j + 1 if j < n else n


def _salvage_per_cell_entries(text: str, id_strs: set) -> dict:
    """Recover individual ``"cell_N": {...}`` entries from a truncated answer.

    Used when no fully-balanced answer JSON is present (model hit
    max_tokens mid-output). Returns whatever per-cell objects the model
    managed to close before the cut.
    """
    salvaged: dict = {}
    for m in re.finditer(r'"([^"\\]+)"\s*:\s*\{', text):
        key = m.group(1)
        if key not in id_strs or key in salvaged:
            continue
        depth = 0
        in_str = False
        esc = False
        start = m.end() - 1  # position of '{'
        j = start
        n = len(text)
        while j < n:
            ch = text[j]
            if in_str:
                if esc:           esc = False
                elif ch == '\\':  esc = True
                elif ch == '"':   in_str = False
            else:
                if ch == '"':     in_str = True
                elif ch == '{':   depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            salvaged[key] = json.loads(text[start:j + 1])
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break
            j += 1
    return salvaged


def extract_cluster_answer(
    text: str, ids: list, *, strip_think: bool = False,
) -> dict:
    """Parse LLM response → {id: {"annotation": str|None, "rationale": str|None}}.

    Strategy:
      1. Scan ``text`` for every balanced ``{...}`` that parses as JSON, and
         pick the dict whose top-level keys overlap most with ``ids`` — this
         is the answer JSON, regardless of surrounding thinking traces,
         markdown fences, or example JSON snippets quoted inside reasoning.
      2. If the best candidate is incomplete (e.g. model hit max_tokens
         mid-output and never closed the wrapping ``}``), salvage individual
         ``"cell_N": {...}`` entries from the raw text and merge them in.

    When ``strip_think=True`` (open-source thinking models — Qwen3 with
    reasoning_parser, DeepSeek-V4, …), the search is restricted to the
    slice past the last ``</think>`` close tag (defends against pseudo-
    JSON in earlier drafts). Falls back to the full text if the post-
    think slice yields nothing.
    """
    empty = {i: {'annotation': None, 'rationale': None} for i in ids}
    id_strs = {str(i) for i in ids}

    def _scan(scope: str) -> tuple[dict | None, int]:
        b: dict | None = None
        b_score = 0
        for obj in _iter_balanced_json_objects(scope):
            if not isinstance(obj, dict):
                continue
            score = sum(1 for k in obj.keys() if str(k) in id_strs)
            if score > b_score:
                b, b_score = obj, score
        return b, b_score

    if strip_think:
        think_end = text.rfind('</think>')
        scope = (text[think_end + len('</think>'):]
                 if think_end >= 0 else text)
        best, best_score = _scan(scope)
        if best is None and think_end >= 0:
            best, best_score = _scan(text)
    else:
        best, best_score = _scan(text)

    if best is None or best_score < len(ids):
        salvaged = _salvage_per_cell_entries(text, id_strs)
        if salvaged:
            merged = dict(best) if isinstance(best, dict) else {}
            for k, v in salvaged.items():
                merged.setdefault(k, v)
            best = merged

    if best is None:
        return empty

    result: dict = {}
    for i in ids:
        entry = best.get(str(i))
        if entry is None and str(i).isdigit():
            entry = best.get(int(i))
        if isinstance(entry, dict):
            result[i] = {
                'annotation': entry.get('annotation'),
                'rationale':  entry.get('rationale'),
            }
        elif isinstance(entry, str):
            result[i] = {'annotation': entry, 'rationale': None}
        else:
            result[i] = {'annotation': None, 'rationale': None}
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

    Predictions only — no accuracy/F1/balanced_acc fields. To compute
    metrics, run ``python -m src.llm.postprocess.anchor_metric --eval_path``
    on the resulting pred.json; that writes ``pred_anchor_metrics.json``.

    Schema::

        {
          "meta": {model, ablation, dataset_path, n_samples, n_steps},
          "samples": {
            <sample>: {
              "steps": {
                <step>: {
                  "x_marker": str, "y_marker": str, "options": [...],
                  "cells": [
                    {cell_id, cell_index, x, y, gt, prediction, rationale}
                  ]
                }
              }
            }
          }
        }
    """
    samples: dict = {}
    reasoning_iter = (reasoning_texts if reasoning_texts is not None
                      else [''] * len(raw_texts))
    for group, text, reasoning in zip(groups, raw_texts, reasoning_iter):
        cell_ids = [c['cell_id'] for c in group['cells']]
        parsed = extract_cluster_answer(text, cell_ids,
                                        strip_think=strip_think)

        sample_entry = samples.setdefault(group['sample'], {'steps': {}})
        cells = []
        for c in group['cells']:
            p = parsed.get(c['cell_id']) or {}
            cells.append({
                'cell_id':    c['cell_id'],
                'cell_index': c['cell_index'],
                'x':          c['x'],
                'y':          c['y'],
                'gt':         c['gt_label'],
                'prediction': p.get('annotation'),
                'rationale':  p.get('rationale'),
            })
        step_entry = {
            'x_marker': group.get('x_marker'),
            'y_marker': group.get('y_marker'),
            'options':  group['options'],
            'cells':    cells,
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


def iter_anchor_records(eval_data) -> Iterator[dict]:
    """Yield flat per-anchor records from the hierarchical dict
    (``{meta, samples}``).
    """
    for sample_name, sample in eval_data.get('samples', {}).items():
        for step_str, step in sample.get('steps', {}).items():
            for c in step.get('cells', []):
                yield {
                    'sample':     sample_name,
                    'step':       int(step_str),
                    'cell_id':    c['cell_id'],
                    'cell_index': c.get('cell_index'),
                    'x':          c.get('x'),
                    'y':          c.get('y'),
                    'gt':         c.get('gt'),
                    'prediction': c.get('prediction'),
                    'rationale':  c.get('rationale'),
                }


def group_by_sample_step(eval_data) -> dict[tuple[str, int], list[dict]]:
    """(sample, step) → list of per-anchor records."""
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in iter_anchor_records(eval_data):
        out[(r['sample'], int(r['step']))].append(r)
    return out
