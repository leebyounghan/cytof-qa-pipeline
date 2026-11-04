#!/usr/bin/env python3
"""LLM-Gate (flat) — sequential cascading gating walk.

For each sample, walks ``gating_plan.json`` step by step. At step *k* the
parent population is computed from the LLM's own predictions for steps
1..k-1 (not GT). If at any step the predicted parent is empty, that step
is skipped (an all-``Unassigned`` column is recorded so downstream steps
referencing it resolve naturally) and the cascade continues — sibling
branches whose parent expressions don't depend on the empty step are
unaffected. The cascade is only hard-aborted on executor exceptions.

Per-step output: ``{output_dir}/{dataset}/{sample}/step_{NN}/prediction.json``
with ``predicted_labels`` of length = predicted-parent count. Skipped
steps have no ``step_NN/`` directory; their column still appears in
``predicted_steps.parquet`` as all ``Unassigned``.

Per-sample artifacts:
- ``cascade_meta.json`` — n_steps_completed, ``skipped_steps`` list,
  ``aborted`` flag (hard executor errors only).
- ``predicted_steps.parquet`` — full-N table of all Step columns
  (cells outside a step's predicted parent appear as ``Unassigned``).

Usage::

    export OPENAI_API_KEY=sk-...
    uv run python -m src.llm_gate_flat.runner \\
        --benchmark   benchmark/ \\
        --data-dir    data/ \\
        --datasets    Acute2020 \\
        --gating-plan data/Acute2020/gating_plan_L1_coarse.json \\
        --output-dir  results_flat/llm_gate_flat/ \\
        --model       gpt-5.4
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.bench import BenchmarkLoader
from src.hard_depletions import resolve_hard_depletion_args
from src.hard_shifts import resolve_hard_shift_args
from src.llm_gate.inference.system_prompts import get_system_prompt

from .cascade_state import CascadeState
from .plot import plot_sample_summary, plot_step
from .step_executor import StepResult, execute_step

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not found. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

load_dotenv()


class _OpenAICall:
    """Sync OpenAI chat call with per-attempt inference accounting."""

    def __init__(self, model: str, max_tokens: int, temperature: float = 0.0,
                 reasoning_effort: str | None = None, max_retries: int = 3):
        self._client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        pricing_path = Path(__file__).with_name("pricing.json")
        with pricing_path.open() as handle:
            pricing = json.load(handle)
        if model not in pricing:
            raise ValueError(f"No inference cost rates for model {model!r} in {pricing_path}")
        self.pricing = pricing[model]
        self._log_path: Path | None = None
        self._dataset: str | None = None
        self._sample: str | None = None
        self._step: int | None = None
        self._records: list[dict] = []
        self.last_call_info: dict | None = None

    def begin_sample(self, dataset: str, sample: str, sample_out: Path) -> None:
        self._dataset = dataset
        self._sample = sample
        self._step = None
        self._records = []
        self.last_call_info = None
        self._log_path = sample_out / "inference_calls.jsonl"
        if self._log_path.exists():
            raise FileExistsError(f"Inference log already exists: {self._log_path}")

    def set_step(self, step: int) -> None:
        self._step = step
        self.last_call_info = None

    def _record(self, record: dict) -> None:
        if self._log_path is None or self._dataset is None or self._sample is None or self._step is None:
            raise RuntimeError("Inference log context was not initialized")
        record.update({
            "schema_version": 1,
            "dataset": self._dataset,
            "sample": self._sample,
            "step": self._step,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        })
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records.append(record)
        if record["status"] == "success":
            self.last_call_info = record

    def summary(self) -> dict:
        success = [row for row in self._records if row["status"] == "success"]
        return {
            "n_api_attempts": len(self._records),
            "n_successful_calls": len(success),
            "n_failed_attempts": len(self._records) - len(success),
            "api_elapsed_seconds": sum(row["elapsed_seconds"] for row in self._records),
            "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in success),
            "cached_prompt_tokens": sum(row["usage"]["cached_prompt_tokens"] for row in success),
            "completion_tokens": sum(row["usage"]["completion_tokens"] for row in success),
            "reasoning_tokens": sum(row["usage"]["reasoning_tokens"] for row in success),
            "estimated_cost_usd": sum(row["estimated_cost_usd"] for row in success),
            "pricing_usd_per_million_tokens": self.pricing,
        }

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            try:
                kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                if self.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                resp = self._client.chat.completions.create(**kwargs)
                usage = resp.usage
                if usage is None:
                    raise RuntimeError(f"OpenAI response {resp.id} omitted usage; cannot account for cost")
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                if prompt_tokens is None or completion_tokens is None:
                    raise RuntimeError(f"OpenAI response {resp.id} omitted token counts")
                prompt_details = usage.prompt_tokens_details
                completion_details = usage.completion_tokens_details
                cached_tokens = (prompt_details.cached_tokens or 0) if prompt_details else 0
                reasoning_tokens = (completion_details.reasoning_tokens or 0) if completion_details else 0
                if not 0 <= cached_tokens <= prompt_tokens:
                    raise ValueError(f"Invalid cached token count in OpenAI response {resp.id}")
                cost = (
                    (prompt_tokens - cached_tokens) * self.pricing["input"]
                    + cached_tokens * self.pricing["cached_input"]
                    + completion_tokens * self.pricing["output"]
                ) / 1_000_000
                self._record({
                    "attempt": attempt + 1,
                    "status": "success",
                    "started_at_utc": started_at,
                    "elapsed_seconds": time.perf_counter() - started,
                    "response_id": resp.id,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "cached_prompt_tokens": cached_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning_tokens,
                    },
                    "estimated_cost_usd": cost,
                })
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                self._record({
                    "attempt": attempt + 1,
                    "status": "error",
                    "started_at_utc": started_at,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(e).__name__,
                })
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} retries: {last_err}")


def _enumerate_samples(loader: BenchmarkLoader, args) -> list[tuple[str, str]]:
    """Yield (dataset, sample) pairs respecting --datasets/--samples/--split."""
    ds_names = args.datasets if args.datasets else loader.list_datasets()
    split_map: dict[str, list[str]] = {}
    if args.split:
        split_map = loader._split_samples(Path(args.split), args.split_set)  # noqa: SLF001
    out: list[tuple[str, str]] = []
    for ds in ds_names:
        ds_dir = loader.benchmark_dir / ds
        if not ds_dir.is_dir():
            continue
        sample_names = sorted(d.name for d in ds_dir.iterdir() if d.is_dir())
        if args.split:
            allowed = set(split_map.get(ds, []))
            sample_names = [s for s in sample_names if s in allowed]
        if args.samples:
            wanted = set(args.samples)
            sample_names = [s for s in sample_names if s in wanted]
        for s in sample_names:
            parquet_path = loader.data_dir / ds / "parquet" / f"{s}.parquet"
            if parquet_path.exists():
                out.append((ds, s))
    return out


def _step_dir_name(step_num: int) -> str:
    return f"step_{step_num:02d}"


def _write_prediction_json(
    out_path: Path,
    dataset: str,
    sample: str,
    step_num: int,
    annotation_col: str,
    method: str,
    predicted_labels: list,
    perturbation_name: str | None,
    info: dict,
    gates: dict | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": f"{dataset}/{sample}/step_{step_num:02d}",
        "method": method,
        "annotation_column": annotation_col,
        "predicted_labels": predicted_labels,
        "info": info,
    }
    if gates is not None:
        payload["gates"] = gates
    if perturbation_name:
        payload["perturbation"] = perturbation_name
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_sample(
    loader: BenchmarkLoader,
    dataset: str,
    sample: str,
    gating_plan: dict,
    output_dir: Path,
    llm_call: _OpenAICall,
    system_prompt: str,
    tiebreak: str,
    method_name: str,
    save_cascade_parquet: bool,
    do_plots: bool,
) -> dict:
    df, _channels = loader._load_sample(dataset, sample)  # noqa: SLF001
    cfg = loader._load_config(dataset)  # noqa: SLF001
    channel_marker_map = cfg["channel_marker_map"]
    cofactor = cfg.get("cofactor")
    modality = "cytof"

    state = CascadeState(df, dataset, sample)
    sample_out = output_dir / dataset / sample
    sample_out.mkdir(parents=True, exist_ok=True)
    llm_call.begin_sample(dataset, sample, sample_out)

    aborted = False
    abort_step: int | None = None
    abort_reason: str | None = None
    n_completed = 0
    skipped_steps: list[dict] = []
    plot_records: list[dict] = []

    steps = gating_plan.get("steps", [])
    print(f"  [{dataset}/{sample}] {len(steps)} steps")

    for step in steps:
        step_num = int(step["step"])
        llm_call.set_step(step_num)
        try:
            result = execute_step(
                state, step,
                channel_marker_map=channel_marker_map,
                cofactor=cofactor,
                modality=modality,
                cohort=dataset,
                llm_call=llm_call,
                system_prompt=system_prompt,
                tiebreak=tiebreak,
            )
        except RuntimeError as e:
            aborted = True
            abort_step = step_num
            abort_reason = f"step_executor_error: {e}"
            print(f"    step_{step_num:02d}: ABORT — {e}")
            break

        if result is None:
            annotation_col = step["annotation column name"]
            state.set(
                annotation_col,
                np.array(["Unassigned"] * state.n, dtype=object),
            )
            skipped_steps.append({
                "step": step_num,
                "annotation_col": annotation_col,
                "reason": "empty_predicted_parent",
            })
            print(f"    step_{step_num:02d}: SKIP — empty predicted parent")
            continue

        step_dir = sample_out / _step_dir_name(step_num)
        _write_prediction_json(
            step_dir / "prediction.json",
            dataset=dataset, sample=sample, step_num=step_num,
            annotation_col=result.annotation_col,
            method=method_name,
            predicted_labels=result.predicted_subset,
            perturbation_name=loader.perturbation_name,
            info=result.info,
            gates=result.gates,
        )
        n_completed += 1
        print(
            f"    step_{step_num:02d}: n_parent={result.n_parent} "
            f"gates_valid={result.info['n_gates_valid']}/{result.info['n_gates_total']}"
        )

        if do_plots:
            plot_step(
                step_dir / "plot.png",
                step_num=result.step_num,
                parent_expr=result.parent_expr,
                x_marker_display=result.x_marker_display,
                y_marker_display=result.y_marker_display,
                x_arr=result.x_arr,
                y_arr=result.y_arr,
                predicted_labels=result.predicted_subset,
                gates=result.gates,
                options=result.options,
            )
            plot_records.append({
                "step_num": result.step_num,
                "x_marker_display": result.x_marker_display,
                "y_marker_display": result.y_marker_display,
                "x_arr": result.x_arr,
                "y_arr": result.y_arr,
                "predicted_labels": result.predicted_subset,
                "gates": result.gates,
                "options": result.options,
            })

    meta = {
        "dataset": dataset,
        "sample": sample,
        "method": method_name,
        "model": llm_call.model,
        "tiebreak": tiebreak,
        "perturbation": loader.perturbation_name,
        "n_steps_total": len(steps),
        "n_steps_completed": n_completed,
        "n_steps_skipped": len(skipped_steps),
        "skipped_steps": skipped_steps,
        "aborted": aborted,
        "abort_step": abort_step,
        "abort_reason": abort_reason,
        "gating_plan_steps": [int(s["step"]) for s in steps],
        "inference": llm_call.summary(),
    }
    with open(sample_out / "cascade_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if save_cascade_parquet and state.predicted_cols():
        state.to_predicted_parquet().to_parquet(
            sample_out / "predicted_steps.parquet", index=False,
        )

    if do_plots and plot_records:
        plot_sample_summary(
            sample_out / "cascade_summary.png",
            plot_records, sample=sample, dataset=dataset,
        )

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential cascading LLM-Gate runner (flat-annotation pipeline).",
    )
    parser.add_argument(
        "--gating-plan", type=Path, required=True,
        help="Path to gating_plan.json (or L1/L2 variant) describing the cascade.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_flat/llm_gate_flat/"),
        help="Where to write per-step prediction.json and per-sample artifacts.",
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-reasoning-effort", choices=("none", "low", "medium", "high", "xhigh"))
    parser.add_argument("--tiebreak", default="smallest", choices=("smallest", "priority"))
    parser.add_argument("--method-name", default="llm_gate_flat",
                        help="Method identifier embedded in prediction.json (default llm_gate_flat).")
    parser.add_argument("--no-save-cascade-parquet", dest="save_cascade_parquet",
                        action="store_false", default=True,
                        help="Skip writing predicted_steps.parquet per sample.")
    parser.add_argument("--no-plot", dest="do_plots",
                        action="store_false", default=True,
                        help="Skip per-step plot.png and cascade_summary.png.")
    BenchmarkLoader.add_cli_args(parser)
    args = parser.parse_args()

    resolve_hard_depletion_args(args)
    resolve_hard_shift_args(
        args,
        meta_channels_lookup=lambda ds: json.loads(
            (Path("data") / ds / "parquet" / "meta.json").read_text()
        )["channels"],
    )

    if not args.gating_plan.exists():
        print(f"[ERROR] gating plan not found: {args.gating_plan}", file=sys.stderr)
        sys.exit(2)
    with open(args.gating_plan) as f:
        gating_plan = json.load(f)

    # Auto-append plan_slug so L1/L2/L3 outputs land in distinct subdirs.
    # Shared with src/flowsom_c2s/ — both pipelines key results by plan_slug.
    from src.flat_eval import gating_plan_slug
    plan_slug = gating_plan_slug(args.gating_plan)
    args.output_dir = args.output_dir / plan_slug
    print(f"plan_slug={plan_slug} -> writing under {args.output_dir}")

    loader = BenchmarkLoader.from_cli(args)
    samples = _enumerate_samples(loader, args)
    if not samples:
        print("[WARN] No samples matched the filters; nothing to do.")
        return

    print(
        f"Cascade: {len(samples)} sample(s), {len(gating_plan.get('steps', []))} steps, "
        f"model={args.model}, tiebreak={args.tiebreak}",
    )

    llm_call = _OpenAICall(model=args.model, max_tokens=args.max_tokens,
                           temperature=args.temperature,
                           reasoning_effort=args.api_reasoning_effort)
    system_prompt = get_system_prompt()

    summaries: list[dict] = []
    for ds, sample in samples:
        meta = run_sample(
            loader, ds, sample, gating_plan,
            output_dir=args.output_dir,
            llm_call=llm_call,
            system_prompt=system_prompt,
            tiebreak=args.tiebreak,
            method_name=args.method_name,
            save_cascade_parquet=args.save_cascade_parquet,
            do_plots=args.do_plots,
        )
        summaries.append(meta)

    n_aborted = sum(1 for m in summaries if m["aborted"])
    n_with_skips = sum(1 for m in summaries if m.get("n_steps_skipped", 0) > 0)
    n_full = len(summaries) - n_aborted - n_with_skips
    print(
        f"Done: {n_full}/{len(summaries)} samples completed full cascade with no skips, "
        f"{n_with_skips} had empty-parent skips, {n_aborted} hard-aborted.",
    )


if __name__ == "__main__":
    main()
