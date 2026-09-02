"""Async OpenAI runner for the flowsom_c2s whole-cell pipeline.

One LLM request per sample (a single prompt enumerates every cluster +
the full leaf list). Concurrency is controlled by --concurrency.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m src.flowsom_c2s.inference.run_openai \\
        --benchmark benchmark_flat/Lyoplate_tcell \\
        --model gpt-5.4 --ablation_slug hvp10 \\
        --output_path results/flowsom_gpt5_hvp10/Lyoplate_tcell/pred.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.flowsom_c2s.inference.output_schema import (
    extract_cluster_answers,
    assemble_pred,
)
from src.flowsom_c2s.inference.system_prompts import get_system_prompt

try:
    from openai import AsyncOpenAI
except ImportError:
    print("[ERROR] openai package not found. Run: pip install openai", file=sys.stderr)
    sys.exit(1)


def load_samples(
    benchmark_dir: Path,
    plan_slug: str,
    ablation: str,
    sample_filter: list[str] | None,
) -> list[dict]:
    """Walk ``benchmark_flat/<ds>/<sample>/<plan_slug>/flowsom_<ablation>/cells.md``.

    Returns one entry per sample with: name, prompt_text, leaf_categories,
    cluster_ids, task.
    """
    out: list[dict] = []
    for sample_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
        if sample_filter and sample_dir.name not in sample_filter:
            continue
        plan_dir = sample_dir / plan_slug
        prompt_path = plan_dir / f"flowsom_{ablation}" / "cells.md"
        task_path = plan_dir / "task.json"
        c2s_path = sample_dir / "c2s.json"
        if not (prompt_path.exists() and task_path.exists() and c2s_path.exists()):
            continue
        with open(task_path) as f:
            task = json.load(f)
        with open(c2s_path) as f:
            c2s = json.load(f)
        with open(prompt_path) as f:
            prompt_text = f.read()
        out.append({
            "name": sample_dir.name,
            "prompt_text": prompt_text,
            "leaf_categories": list(task["leaf_categories"]),
            "cluster_ids": sorted(c2s["clusters"].keys(), key=lambda k: int(k)),
            "task_id": task["task_id"],
        })
    return out


async def _call_one(
    client: AsyncOpenAI,
    user_prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> str:
    async with semaphore:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


async def run(
    samples: list[dict],
    model: str,
    concurrency: int,
    max_tokens: int,
) -> list[str]:
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_call_one(client, s["prompt_text"], model, semaphore, max_tokens) for s in samples]
    print(f"Sending {len(tasks)} requests (concurrency={concurrency}) ...")
    return await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="flowsom_c2s OpenAI inference")
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="benchmark_flat/<dataset>/  -- the per-dataset dir")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ablation_slug", default="full",
                        help="prompt ablation: full | hvp10 | nohvp ...")
    parser.add_argument("--gating-plan", type=Path, default=None,
                        help="Used to auto-derive plan-slug for input prompts and output path")
    parser.add_argument("--plan-slug", type=str, default=None,
                        help="Override plan slug (default: auto from --gating-plan, else 'default')")
    parser.add_argument("--sample", nargs="*", help="restrict to these sample IDs")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--perturbation", type=str, default=None)
    args = parser.parse_args()

    from src.flat_eval import gating_plan_slug as _slug
    if args.plan_slug:
        plan_slug = args.plan_slug
    elif args.gating_plan is not None:
        plan_slug = _slug(args.gating_plan)
    else:
        plan_slug = "default"

    samples = load_samples(args.benchmark, plan_slug, args.ablation_slug, args.sample)
    if not samples:
        print(f"[ERROR] no samples found under {args.benchmark} for ablation {args.ablation_slug}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(samples)} samples from {args.benchmark}")

    raw_texts = asyncio.run(run(samples, args.model, args.concurrency, args.max_tokens))

    per_sample: dict = {}
    for s, text in zip(samples, raw_texts):
        leaf_set = set(s["leaf_categories"])
        parsed, n_invalid = extract_cluster_answers(text, leaf_set)
        # Backfill missing clusters with leaf=None
        for cid in s["cluster_ids"]:
            key = f"cluster_{cid}"
            if key not in parsed:
                parsed[key] = {"leaf": None, "rationale": None}
                n_invalid = max(n_invalid, 0) + 1
        per_sample[s["name"]] = {"clusters": parsed, "n_invalid": n_invalid}

    payload = assemble_pred(
        model=args.model,
        ablation=args.ablation_slug,
        benchmark_dir=str(args.benchmark),
        per_sample=per_sample,
        perturbation=args.perturbation,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output_path}")
    print(f"  samples: {len(per_sample)}")
    print(f"  parse-failed clusters (total): {payload['meta']['n_parse_fail_clusters']}")


if __name__ == "__main__":
    main()
