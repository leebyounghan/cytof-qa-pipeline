#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vLLM batch evaluation runner.

Reads ``c2s.json`` + rendered prompts from a per-ablation sibling
directory ``c2s_{ablation_slug}/cells.md``. Each (sample, step) is a
standalone QA the model answers; per-cell predictions are written as a
hierarchical eval JSON matching the shape downstream postprocess + plot
tools consume.

Usage::

    python -m src.llm.inference.run_vllm \\
        --dataset_path   benchmark/Acute2020 \\
        --model_name     Qwen/Qwen3-8B \\
        --ablation_slug  full \\
        --output_path    output/eval/Acute2020_qwen3_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

from src.llm.inference.data_loader import load_c2s
from src.llm.inference.output_schema import assemble_eval_output
from src.llm.inference.pool import build_pool, shard_pool, shard_summary
from src.llm.inference.system_prompts import get_system_prompt


# ── Thinking-token stripper ───────────────────────────────────────────────────

_THINK_CLOSED_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r'<think>.*', re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models.

    Also strips unclosed ``<think>`` blocks (model hit max_tokens while
    still reasoning) so downstream JSON parsing can find the answer.
    """
    text = _THINK_CLOSED_RE.sub('', text)
    text = _THINK_UNCLOSED_RE.sub('', text)
    return text.strip()


def _parse_shard(s: str) -> tuple[int, int]:
    """Parse ``i/N`` shard spec → (i, N). Defaults to (0, 1) on empty."""
    if not s or s == '0/1':
        return 0, 1
    if '/' not in s:
        raise ValueError(f'--shard must be i/N, got {s!r}')
    a, b = s.split('/', 1)
    i, n = int(a), int(b)
    if not 0 <= i < n:
        raise ValueError(f'--shard {s!r}: index out of range')
    return i, n


def main(args):
    pool_mode = bool(args.datasets)
    if pool_mode and args.dataset_path:
        raise SystemExit('--datasets and --dataset_path are mutually exclusive')
    if not pool_mode and not args.dataset_path:
        raise SystemExit('Must pass either --dataset_path (single cohort) '
                         'or --datasets (pool mode)')

    if pool_mode:
        if not args.benchmark:
            raise SystemExit('--benchmark required when using --datasets')
        if not args.output_dir:
            raise SystemExit('--output_dir required in pool mode')
        if args.max_samples > 0:
            raise SystemExit('--max_samples not supported in pool mode '
                             '(use --split for deterministic filtering)')
        shard_idx, n_shards = _parse_shard(args.shard)
        print(f'Pool mode    : datasets={args.datasets}  '
              f'shard={shard_idx}/{n_shards}  balance={args.balance}'
              + (f'  split={args.split}' if args.split else ''))
        pool = build_pool(
            benchmark=args.benchmark,
            datasets=args.datasets,
            ablation_slug=args.ablation_slug,
            split=args.split or None,
            splits_path=args.splits_path,
        )
        if not pool:
            print('No groups in pool. Exiting.')
            return
        # Print balance preview only on shard 0 to avoid 4× repeated logs
        if shard_idx == 0:
            print(shard_summary(pool, n_shards, balance=args.balance))
        groups, _ = shard_pool(pool, shard_idx, n_shards, balance=args.balance)
        if not groups:
            print(f'shard {shard_idx}/{n_shards} is empty. Exiting.')
            return
        total_cells = sum(len(g['cells']) for g in groups)
        print(f'Shard groups : {len(groups):,}  |  Total cells: {total_cells:,}')
    else:
        print(f'Dataset      : {args.dataset_path}')
        sample_filter = set(args.sample) if args.sample else None
        groups = load_c2s(
            args.dataset_path,
            ablation_slug=args.ablation_slug,
            sample_filter=sample_filter,
        )
        if not groups:
            print('No groups found. Exiting.')
            return
        if args.max_samples > 0:
            random.seed(args.seed)
            groups = random.sample(groups, min(args.max_samples, len(groups)))
        total_cells = sum(len(g['cells']) for g in groups)
        print(f'Groups       : {len(groups):,}  |  Total cells: {total_cells:,}')

    # ── Init vLLM + tokenizer ────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f'\nInitializing vLLM: {args.model_name}')
    llm_kwargs = dict(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_executor_backend='mp',
        trust_remote_code=True,
        enable_prefix_caching=args.enable_prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    if args.max_model_len > 0:
        llm_kwargs['max_model_len'] = args.max_model_len
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True,
    )
    temperature = args.temperature
    if temperature == 0.0 and args.enable_thinking:
        temperature = 0.6
        print(f'  [INFO] Thinking enabled → temperature auto-set to {temperature}')
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    # ── Format prompts via chat template ─────────────────────────────────────
    system_prompt = get_system_prompt(args.system_variant)

    # Detect whether the tokenizer supports enable_thinking (Qwen3, etc.)
    _test_msg = [{'role': 'user', 'content': 'test'}]
    try:
        tokenizer.apply_chat_template(
            _test_msg, tokenize=False, enable_thinking=args.enable_thinking,
        )
        _supports_thinking = True
    except TypeError:
        _supports_thinking = False
        print('  [INFO] Tokenizer does not support enable_thinking — ignored.')

    prompts = []
    for g in groups:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user',   'content': g['prompt_text']},
        ]
        kwargs = {'enable_thinking': args.enable_thinking} if _supports_thinking else {}
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs,
        )
        prompts.append(text)

    raw_texts: list[str] = [''] * len(prompts)

    for chunk_start in range(0, len(prompts), args.chunk_size):
        chunk_end     = min(chunk_start + args.chunk_size, len(prompts))
        batch_prompts = prompts[chunk_start:chunk_end]

        print(f'\nChunk {chunk_start // args.chunk_size + 1} / '
              f'{(len(prompts) - 1) // args.chunk_size + 1} ...')

        outputs = llm.generate(batch_prompts, sampling_params)
        for local_i, output in enumerate(outputs):
            raw_texts[chunk_start + local_i] = strip_thinking(output.outputs[0].text)

    if pool_mode:
        # Split this shard's groups by cohort, write one partial JSON per
        # (cohort, shard). The reducer will deep-merge across shards.
        shard_idx, n_shards = _parse_shard(args.shard)
        partial_dir = os.path.join(args.output_dir, '_partial')
        os.makedirs(partial_dir, exist_ok=True)
        per_ds: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            per_ds.setdefault(g['dataset'], []).append(i)
        for ds, idxs in per_ds.items():
            sub_groups = [groups[i] for i in idxs]
            sub_raw    = [raw_texts[i] for i in idxs]
            ds_path    = sub_groups[0].get('dataset_path', ds)
            eval_data = assemble_eval_output(
                sub_groups, sub_raw,
                model=args.model_name,
                ablation=args.ablation_slug,
                dataset_path=ds_path,
            )
            out_path = os.path.join(
                partial_dir, f'{ds}__shard{shard_idx}of{n_shards}.json'
            )
            with open(out_path, 'w') as f:
                json.dump(eval_data, f, ensure_ascii=False, indent=2)
            n_cells = sum(len(step['cells'])
                          for sample in eval_data['samples'].values()
                          for step in sample['steps'].values())
            print(f'Wrote partial: {ds}  '
                  f'{len(sub_groups)} groups / {n_cells:,} cells → {out_path}')
        print('Reduce per-cohort outputs with: '
              f'python -m src.llm.scripts.merge_pool_partials '
              f'--output_dir {args.output_dir} --n_shards {n_shards}')
        return

    # Legacy single-cohort path
    if args.output_path:
        out_path = args.output_path
    else:
        model_short = args.model_name.split('/')[-1]
        out_path = f'./{model_short}_results.json'
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    eval_data = assemble_eval_output(
        groups, raw_texts,
        model=args.model_name,
        ablation=args.ablation_slug,
        dataset_path=args.dataset_path,
    )

    with open(out_path, 'w') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    n_cells = sum(len(step['cells'])
                  for sample in eval_data['samples'].values()
                  for step in sample['steps'].values())
    print(f'Wrote {n_cells:,} cells → {out_path}')
    print('Compute anchor metrics with: '
          f'python -m src.llm.postprocess.anchor_metric --eval_path {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='vLLM batch evaluation for CyTOF/Flow c2s'
    )
    parser.add_argument(
        '--dataset_path', default='',
        help='Single-cohort mode: benchmark/{dataset} directory '
             '(e.g. benchmark/Acute2020). Mutually exclusive with --datasets.',
    )
    parser.add_argument(
        '--benchmark', default='',
        help='Pool mode: benchmark root directory (e.g. benchmark/).',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='Pool mode: cohort names under --benchmark to pool together '
             '(e.g. Acute2020 Acute2021).',
    )
    parser.add_argument(
        '--shard', default='0/1',
        help='Pool mode: i/N where i is this worker (0..N-1) and N is the '
             'total worker count. Default 0/1 = no sharding.',
    )
    parser.add_argument(
        '--balance', choices=('lpt', 'rr'), default='lpt',
        help='Pool mode: shard balancing strategy. lpt = compute-balanced '
             'by prompt length (default). rr = round-robin count-balance.',
    )
    parser.add_argument(
        '--split', default='',
        help='Pool mode: filter samples by split (test/val/train) using '
             'splits.json. Empty = all samples.',
    )
    parser.add_argument(
        '--splits_path', default='benchmark/splits.json',
        help='Pool mode: path to splits.json (default benchmark/splits.json).',
    )
    parser.add_argument(
        '--output_dir', default='',
        help='Pool mode: directory for per-cohort partial JSONs '
             '(written under {output_dir}/_partial/).',
    )
    parser.add_argument('--model_name', default='Qwen/Qwen3-8B')
    parser.add_argument(
        '--output_path', default='',
        help='Output JSON path (default: ./<model>_results.json)',
    )
    parser.add_argument('--tensor_parallel_size', type=int, default=8)
    parser.add_argument(
        '--sample', nargs='+', default=None,
        help='Sample names to include (default: all)',
    )
    parser.add_argument(
        '--max_samples', type=int, default=0,
        help='Random subsample of groups (0 = all)',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--chunk_size', type=int, default=2048,
        help='Groups per vLLM generate() call (default: 2048)',
    )
    parser.add_argument(
        '--max_tokens', type=int, default=12164,
        help='Max generation tokens (default: 12164)',
    )
    parser.add_argument(
        '--ablation_slug', default='full',
        help='Prompt directory slug (full / nohvp / hvp10 / ...). '
             'Default: full',
    )
    parser.add_argument(
        '--system_variant', default='_notag',
        help='System-prompt variant suffix passed to get_system_prompt. '
             "Default: '_notag' (current prompts omit zone tags).",
    )
    parser.add_argument(
        '--temperature', type=float, default=0.0,
        help='Sampling temperature (default: 0.0). '
             'Auto-set to 0.6 when enable_thinking is true and temperature is 0.',
    )
    parser.add_argument(
        '--repetition_penalty', type=float, default=1.0,
        help='Repetition penalty (default: 1.0 = off). '
             'Values > 1.0 discourage repetitive output (e.g., 1.05).',
    )
    parser.add_argument(
        '--enable_thinking', type=lambda s: s.lower() == 'true', default=False,
        help='Qwen3 enable_thinking override for apply_chat_template. '
             'Default: false (thinking disabled). Pass "true" to enable.',
    )
    parser.add_argument(
        '--max_model_len', type=int, default=0,
        help='vLLM max sequence length (prompt+output). 0 = use model default. '
             'Lower values shrink KV-cache per slot → more concurrent batches.',
    )
    parser.add_argument(
        '--gpu_memory_utilization', type=float, default=0.92,
        help='Fraction of GPU memory vLLM may use for weights+KV (default: 0.92).',
    )
    parser.add_argument(
        '--enable_prefix_caching', type=lambda s: s.lower() == 'true',
        default=True,
        help='Cache shared prefix (system prompt) across requests. Default: true.',
    )
    args = parser.parse_args()
    main(args)
