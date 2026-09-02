#!/usr/bin/env python3
"""YAML model config → shell variable assignments for the agent-gate runner.

Reads a per-model YAML and prints a block of shell statements that
``run_agent_gate.sh`` eval's to populate:

    MODEL                — OpenAI model id (e.g. ``gpt-5.4``)
    SERVED_NAME          — display / output-path label (defaults to MODEL)
    AGENT_ARGS  (array)  — flags appended to ``python -m
                           src.agent_gate.inference.run_agent``

Null / missing values become "don't pass the flag" — the runner relies
on this so e.g. an unset ``temperature`` falls through to ``run_agent``'s
own default instead of being passed as the literal string ``None``.

Agent-gate is OpenAI-hosted only — the YAML must set ``model.openai_id``;
the vLLM ``model.hf_id`` / ``server:`` schema from llm_gate is NOT
supported here.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml


# (yaml_key, cli_flag, kind)
# kind = "value"  → emit [flag, str(val)] when val is not None
# kind = "switch" → emit [flag] when val is truthy; omit otherwise
GENERATION_FLAGS = [
    ('max_tokens',        '--max_tokens',        'value'),
    ('max_turns',         '--max-turns',         'value'),
    ('temperature',       '--temperature',       'value'),
    ('top_p',             '--top-p',             'value'),
    ('presence_penalty',  '--presence-penalty',  'value'),
    ('frequency_penalty', '--frequency-penalty', 'value'),
]


def _stringify(val):
    if isinstance(val, bool):
        return 'true' if val else 'false'
    if isinstance(val, (dict, list)):
        return json.dumps(val, separators=(',', ':'))
    return str(val)


def collect_args(section: dict, flags) -> list[str]:
    out: list[str] = []
    for key, flag, kind in flags:
        val = section.get(key)
        if val is None:
            continue
        if kind == 'switch':
            if val:
                out.append(flag)
        else:
            out.extend([flag, _stringify(val)])
    return out


def emit_array(name: str, items: list[str]) -> str:
    quoted = ' '.join(shlex.quote(s) for s in items)
    # ``-g`` keeps the array global even when this string is eval'd inside
    # a function — without it bash treats ``declare`` as ``local``, which
    # silently drops args before they reach the caller.
    return f'declare -ga {name}=({quoted})'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    args = ap.parse_args()

    try:
        cfg = yaml.safe_load(args.config.read_text()) or {}
    except yaml.YAMLError as e:
        print(f'[config_to_shell] {args.config}: YAML parse error: {e}',
              file=sys.stderr)
        return 2

    model = cfg.get('model') or {}
    generation = cfg.get('generation') or {}

    openai_id = model.get('openai_id')
    if not openai_id:
        print(f'[config_to_shell] {args.config}: model.openai_id is '
              'required (agent-gate is OpenAI-hosted only).', file=sys.stderr)
        return 2
    if model.get('hf_id'):
        print(f'[config_to_shell] {args.config}: model.hf_id is not '
              'supported by agent-gate (use src.llm_gate for vLLM).',
              file=sys.stderr)
        return 2

    served = model.get('served_name') or str(openai_id)
    agent_args = collect_args(generation, GENERATION_FLAGS)

    print(f'MODEL={shlex.quote(str(openai_id))}')
    print(f'SERVED_NAME={shlex.quote(served)}')
    print(emit_array('AGENT_ARGS', agent_args))
    return 0


if __name__ == '__main__':
    sys.exit(main())
