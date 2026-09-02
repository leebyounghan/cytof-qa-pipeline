#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI async runner for vlm-gate — text + scatter image, single chat
turn per (sample, step). Emits the same hierarchical ``pred.json`` that
``src.vlm_gate.postprocess.propagate`` consumes (and that
``src.llm_gate`` / ``src.agent_gate`` produce in identical schema).

What this is:
  - one chat completion per (sample, step);
  - user message = ``cells.md`` text + a ``data:image/png`` URI carrying
    the decorated scatter (axes + ticks + grid) of the parent
    population;
  - no tools, no loop — the model commits its JSON answer in that single
    turn.

Why it exists:
  - clean ablation that isolates the effect of "having the image" from
    the effect of "having an iterative verification loop". Compare
    against ``src.llm_gate`` (text only, single-shot) and
    ``src.agent_gate`` (text + image + tool-call loop).

Usage::

    export OPENAI_API_KEY=sk-...

    python -m src.vlm_gate.inference.run_openai \\
        --dataset_path benchmark/Acute2020 \\
        --model        gpt-5.4 \\
        --concurrency  10 \\
        --output_path  results/vlm_gpt-5.4_full/Acute2020/pred.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.bench import parse_perturbations
from src.vlm_gate.inference.data_loader import load_gate_tasks
from src.vlm_gate.inference.output_schema import assemble_eval_output
from src.vlm_gate.inference.system_prompts import get_system_prompt
from src.vlm_gate.utils.scatter_render import render_input_scatter

try:
    from openai import AsyncOpenAI
except ImportError:
    print('[ERROR] openai package not found. Run: pip install openai',
          file=sys.stderr)
    sys.exit(1)

load_dotenv()


async def _call_one(
    client: AsyncOpenAI, group: dict, *,
    model: str, system_prompt: str,
    semaphore: asyncio.Semaphore, max_tokens: int,
    temperature: float,
    top_p: float | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
) -> str:
    """One (sample, step) → final JSON text. Builds the multimodal user
    message inline so each step's scatter is its own ``image_url``.

    matplotlib renders the scatter blocking + thread-unsafe in pyplot;
    ``render_input_scatter`` uses Figure + FigureCanvasAgg directly to
    sidestep that, but we still hop to a worker thread so the event
    loop stays responsive under ``--concurrency``.
    """
    async with semaphore:
        scatter_uri = await asyncio.to_thread(
            render_input_scatter,
            scatter_bg_path=group.get('scatter_bg_path'),
            scatter_extent=group['scatter_extent'],
            x_marker=group.get('x_marker', 'x'),
            y_marker=group.get('y_marker', 'y'),
            step_label=f'{group["sample"]} step_{group["step"]:02d}',
        )
        kwargs = dict(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': group['prompt_text']},
                    {'type': 'image_url',
                     'image_url': {'url': scatter_uri}},
                ]},
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        if top_p is not None:
            kwargs['top_p'] = top_p
        if presence_penalty is not None:
            kwargs['presence_penalty'] = presence_penalty
        if frequency_penalty is not None:
            kwargs['frequency_penalty'] = frequency_penalty
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ''


async def run_online(
    groups: list, model: str, ablation: str, dataset_path: str,
    output_path: str, concurrency: int, max_tokens: int,
    system_prompt: str, perturbation: str | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    save_raw: bool = False,
) -> None:
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    print(f'Sending {len(groups)} multimodal requests '
          f'(concurrency={concurrency}) ...')
    outputs = await asyncio.gather(
        *[_call_one(client, g,
                    model=model,
                    system_prompt=system_prompt,
                    semaphore=semaphore,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty)
          for g in groups],
        return_exceptions=True,
    )

    raw_texts: list[str] = []
    for group, output in zip(groups, outputs):
        if isinstance(output, Exception):
            print(f'  [WARN] {group["sample"]} step{group["step"]}: {output}')
            raw_texts.append('')
        else:
            raw_texts.append(output)

    eval_data = assemble_eval_output(
        groups, raw_texts,
        model=model, ablation=ablation, dataset_path=dataset_path,
        perturbation=perturbation,
        save_raw=save_raw,
        reasoning_texts=None,
        strip_think=False,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    n_steps = sum(len(sample['steps'])
                  for sample in eval_data['samples'].values())
    n_gates = sum(
        sum(1 for g in step['gates'].values() if g is not None)
        for sample in eval_data['samples'].values()
        for step in sample['steps'].values()
    )
    print(f'Wrote {n_steps:,} steps / {n_gates:,} gates → {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='OpenAI async runner (vlm-gate, text + scatter image, '
                    'single turn per step)',
    )
    parser.add_argument('--dataset_path', required=True,
                        help='benchmark/{dataset} directory')
    parser.add_argument('--model', default='gpt-5.4')
    parser.add_argument('--output_path', required=True,
                        help='Results JSON path')
    parser.add_argument('--ablation_slug', default='full',
                        help='Prompt directory slug. Default: full')
    parser.add_argument('--sample', nargs='+', default=None,
                        help='Sample names to include')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Random subsample of groups (0 = all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Concurrent chat requests. Default lower than '
                             'llm_gate (10 vs 20) — each request carries an '
                             'image, so per-request latency + tokens are '
                             'higher.')
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument(
        '--temperature', type=float, default=0.0,
        help='Sampling temperature. Default 0.0 (greedy).',
    )
    parser.add_argument('--top-p', type=float, default=None,
                        help='Nucleus sampling. Default: server side.')
    parser.add_argument('--presence-penalty', type=float, default=None)
    parser.add_argument('--frequency-penalty', type=float, default=None)
    parser.add_argument(
        '--save-raw', action='store_true',
        help='Save full raw model response text under '
             'samples[<sample>].steps[<step>].raw.',
    )
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'passed to src.vlm_gate.task). Used to locate '
             'gate_task__{slug}.json + gate_{ablation}__{slug}/ prompts; '
             'recorded in pred.json meta.',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    from src.hard_depletions import (
        add_hard_depletion_args, resolve_hard_depletion_args,
    )
    from src.hard_shifts import (
        add_hard_shift_args, resolve_hard_shift_args,
    )
    add_hard_depletion_args(parser)
    add_hard_shift_args(parser)
    args = parser.parse_args()

    resolve_hard_depletion_args(args)
    resolve_hard_shift_args(
        args,
        meta_channels_lookup=lambda ds: json.loads(
            (Path('data') / ds / 'parquet' / 'meta.json').read_text()
        )['channels'],
    )

    if getattr(args, '_resolved_perturbation', None) is not None:
        perturbation_name = args.perturbation_name
    else:
        perturbation_name, _p = parse_perturbations(
            args.perturbation or [], name_override=args.perturbation_name,
        )

    sample_filter = set(args.sample) if args.sample else None
    groups = load_gate_tasks(
        args.dataset_path,
        ablation_slug=args.ablation_slug,
        sample_filter=sample_filter,
        perturbation_name=perturbation_name,
    )
    if not groups:
        print('No groups found. Exiting.')
        return

    if args.max_samples > 0:
        random.seed(args.seed)
        groups = random.sample(groups, min(args.max_samples, len(groups)))

    print(f'Groups: {len(groups):,}')
    missing_bg = sum(1 for g in groups if not g.get('scatter_bg_path'))
    if missing_bg:
        print(f'  [WARN] {missing_bg} group(s) have no scatter_bg.png — '
              f're-run src.vlm_gate.task to populate the background; '
              f'those steps will receive a blank-canvas image.')

    asyncio.run(run_online(
        groups,
        model=args.model,
        ablation=args.ablation_slug,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        system_prompt=get_system_prompt(),
        perturbation=perturbation_name,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        save_raw=args.save_raw,
    ))


if __name__ == '__main__':
    main()
