#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenAI async eval runner for LLM-Gate — emits the hierarchical pred
JSON (per-category rectangular gates), nothing else.

Reads ``gate_task.json`` + prompts from a per-ablation sibling directory
``gate_{ablation_slug}/cells.md``. Each (sample, step) is one chat
request, run concurrently up to ``--concurrency``.

Usage::

    export OPENAI_API_KEY=sk-...

    python -m src.llm_gate.inference.run_openai \\
        --dataset_path benchmark/Acute2020 \\
        --model        gpt-5.4 \\
        --ablation_slug full \\
        --concurrency  20 \\
        --output_path  results/gate_gpt-5.4_full/Acute2020/pred.json
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
from src.llm_gate.inference.data_loader import load_gate_tasks
from src.llm_gate.inference.output_schema import assemble_eval_output
from src.llm_gate.inference.system_prompts import get_system_prompt

try:
    from openai import AsyncOpenAI
except ImportError:
    print('[ERROR] openai package not found. Run: pip install openai',
          file=sys.stderr)
    sys.exit(1)

load_dotenv()


async def _call_one(
    client: AsyncOpenAI, user_prompt: str, model: str,
    semaphore: asyncio.Semaphore, max_tokens: int, system_prompt: str,
    extra_body: dict | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    seed: int | None = None,
) -> tuple[str, str]:
    """Return (content, reasoning_content). reasoning_content is empty
    when the server / model does not expose a separate thinking trace."""
    async with semaphore:
        kwargs = dict(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        if seed is not None:
            kwargs['seed'] = seed
        if top_p is not None:
            kwargs['top_p'] = top_p
        if presence_penalty is not None:
            kwargs['presence_penalty'] = presence_penalty
        if frequency_penalty is not None:
            kwargs['frequency_penalty'] = frequency_penalty
        if extra_body:
            kwargs['extra_body'] = extra_body
        resp = await client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = msg.content or ''
        reasoning = getattr(msg, 'reasoning_content', None) or ''
        return content, reasoning


async def run_online(
    groups: list, model: str, ablation: str, dataset_path: str,
    output_path: str, concurrency: int, max_tokens: int,
    system_prompt: str, perturbation: str | None = None,
    extra_body: dict | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    save_raw: bool = False,
    strip_think: bool = False,
    seed: int | None = None,
) -> None:
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    print(f'Sending {len(groups)} requests (concurrency={concurrency}) ...')
    if extra_body:
        print(f'  extra_body = {extra_body}')
    if seed is not None:
        print(f'  seed = {seed}')
    outputs = await asyncio.gather(
        *[_call_one(client, g['prompt_text'], model, semaphore, max_tokens,
                    system_prompt, extra_body=extra_body,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    seed=seed)
          for g in groups],
        return_exceptions=True,
    )

    raw_texts: list[str] = []
    reasoning_texts: list[str] = []
    for group, output in zip(groups, outputs):
        if isinstance(output, Exception):
            print(f'  [WARN] {group["sample"]} step{group["step"]}: {output}')
            raw_texts.append('')
            reasoning_texts.append('')
        else:
            content, reasoning = output
            raw_texts.append(content)
            reasoning_texts.append(reasoning)

    eval_data = assemble_eval_output(
        groups, raw_texts,
        model=model, ablation=ablation, dataset_path=dataset_path,
        perturbation=perturbation,
        save_raw=save_raw,
        reasoning_texts=reasoning_texts if save_raw else None,
        strip_think=strip_think,
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
        description='OpenAI async eval (LLM-Gate, one prompt per step)',
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
    parser.add_argument('--concurrency', type=int, default=20)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument(
        '--temperature', type=float, default=0.0,
        help='Sampling temperature. Default 0.0 (greedy).',
    )
    parser.add_argument(
        '--top-p', type=float, default=None,
        help='Nucleus sampling. OpenAI-native. Default: server side.',
    )
    parser.add_argument(
        '--top-k', type=int, default=None,
        help='Top-k sampling. vLLM extension '
             '(forwarded as extra_body.top_k).',
    )
    parser.add_argument(
        '--min-p', type=float, default=None,
        help='Min-p sampling. vLLM extension '
             '(forwarded as extra_body.min_p).',
    )
    parser.add_argument(
        '--presence-penalty', type=float, default=None,
        help='OpenAI-native presence penalty. Qwen3 thinking-mode '
             'official recommendation: 1.5.',
    )
    parser.add_argument(
        '--frequency-penalty', type=float, default=None,
        help='OpenAI-native frequency penalty.',
    )
    parser.add_argument(
        '--repetition-penalty', type=float, default=None,
        help='vLLM extension (multiplicative penalty, 1.0 = off). '
             'Forwarded as extra_body.repetition_penalty.',
    )
    parser.add_argument(
        '--thinking', choices=['on', 'off', ''], default='',
        help='DeepSeek-V4 / Qwen3 chat template flag '
             '(forwarded as extra_body.chat_template_kwargs.thinking + '
             'enable_thinking). Empty = use template default.',
    )
    parser.add_argument(
        '--reasoning-effort', choices=['low', 'medium', 'high', 'max', ''], default='',
        help='DeepSeek-V4 only. Empty = unset (None).',
    )
    parser.add_argument(
        '--thinking-token-budget', type=int, default=None,
        help='vLLM-only sampling parameter (Qwen3 / DeepSeek-V4 reasoning '
             'parsers): forces the model to emit reasoning_end_str once '
             'the thinking trace reaches this many tokens. Forwarded as '
             'extra_body.thinking_token_budget. Ignored by upstream OpenAI.',
    )
    parser.add_argument(
        '--chat-template-kwargs', default='',
        help='Additional chat-template kwargs as JSON. '
             'Merged after --thinking / --reasoning-effort.',
    )
    parser.add_argument(
        '--save-raw', action='store_true',
        help='Save full raw model response text under '
             'samples[<sample>].steps[<step>].raw — useful for debugging '
             'parse failures and prompt iteration. Increases pred.json size.',
    )
    parser.add_argument(
        '--strip-think', action='store_true',
        help='Use the robust JSON extractor for open-source thinking '
             'models (Qwen3 with --reasoning-parser, DeepSeek-V4, …): '
             'slices past </think>, prefers the last fenced block over '
             'earlier drafts. Default off (suitable for OpenAI-class '
             'single-shot models).',
    )
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'passed to src.llm_gate.task). Used to locate '
             'gate_task__{slug}.json + gate_{ablation}__{slug}/ prompts; '
             'recorded in pred.json meta.',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    from src.hard_depletions import add_hard_depletion_args, resolve_hard_depletion_args
    from src.hard_shifts import add_hard_shift_args, resolve_hard_shift_args
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

    system_prompt = get_system_prompt()

    # Build extra_body.chat_template_kwargs from CLI flags
    ct_kwargs: dict = {}
    if args.thinking:
        on = (args.thinking == 'on')
        ct_kwargs['thinking'] = on
        ct_kwargs['enable_thinking'] = on
    if args.reasoning_effort:
        ct_kwargs['reasoning_effort'] = args.reasoning_effort
    if args.chat_template_kwargs:
        ct_kwargs.update(json.loads(args.chat_template_kwargs))
    extra_body: dict = {}
    if ct_kwargs:
        extra_body['chat_template_kwargs'] = ct_kwargs
    if args.top_k is not None:
        extra_body['top_k'] = args.top_k
    if args.min_p is not None:
        extra_body['min_p'] = args.min_p
    if args.repetition_penalty is not None:
        extra_body['repetition_penalty'] = args.repetition_penalty
    if args.thinking_token_budget is not None:
        extra_body['thinking_token_budget'] = args.thinking_token_budget
    extra_body_arg = extra_body if extra_body else None

    asyncio.run(run_online(
        groups,
        model=args.model,
        ablation=args.ablation_slug,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        system_prompt=system_prompt,
        perturbation=perturbation_name,
        extra_body=extra_body_arg,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        save_raw=args.save_raw,
        strip_think=args.strip_think,
        seed=args.seed,
    ))


if __name__ == '__main__':
    main()
