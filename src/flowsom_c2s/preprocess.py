"""flowsom_c2s preprocess: parquet + gating_plan.json -> per-sample task.json.

Emits ONE task.json per sample. Schema:

  - ``leaf_categories``  — sorted list from ``src.flat_eval.derive_leaf_set``
                           (terminal categories not used as a parent in any
                           later step). SHARED with src/llm_gate_flat/.
  - ``leaf_to_step_col`` — {leaf_name -> annotation_column} for postprocess.
  - ``steps``            — full ordered list copied from gating_plan.json.
  - ``gating_plan_path`` — provenance of the leaf vocabulary.

The same ``--gating-plan`` must be passed to ``src.flat_eval`` so both
sides of the comparison induce the same leaf set (L1 / L2 / L3 plan
variants are supported transparently).

Usage:
    python -m src.flowsom_c2s.preprocess \\
        --data-dir data/ --output-dir benchmark_flat/ \\
        --datasets Acute2020 \\
        --gating-plan data/Acute2020/gating_plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.preprocess import load_config, load_sample
from src.flat_eval import derive_leaf_set, gating_plan_slug
from src.flowsom_c2s.utils import build_leaf_to_step_col


def build_task_json(
    dataset_name: str,
    sample_id: str,
    df: pd.DataFrame,
    gating_plan: dict,
    gating_plan_path: Path,
    config: dict,
    expr_cols: list[str],
) -> dict:
    """Assemble the per-sample task.json payload."""
    steps = gating_plan["steps"]
    leaf_set = derive_leaf_set(gating_plan)
    leaf_to_step_col = build_leaf_to_step_col(steps, leaf_set)

    serialized_steps = []
    for s in steps:
        serialized_steps.append({
            "step": s["step"],
            "parent": s["parent"],
            "x_marker": s["x_marker"],
            "y_marker": s["y_marker"],
            "annotation_column": s["annotation column name"],
            "categories": list(s["annotation categories"]) + ["Unassigned"],
            "note": s.get("note", ""),
            "tips": s.get("tips", ""),
        })

    return {
        "task_id": f"{dataset_name}__{sample_id}__{gating_plan_slug(gating_plan_path)}",
        "dataset": dataset_name,
        "sample": sample_id,
        "n_cells": int(len(df)),
        "modality": "flow" if (config.get("cofactor") or 0) >= 100 else "cytof",
        "cofactor": config.get("cofactor"),
        "all_protein_markers": list(expr_cols),
        "gating_plan_path": str(gating_plan_path),
        "plan_slug": gating_plan_slug(gating_plan_path),
        "steps": serialized_steps,
        "leaf_categories": sorted(leaf_set),
        "leaf_to_step_col": leaf_to_step_col,
    }


def process_sample(
    sample_path: Path,
    dataset_name: str,
    config: dict,
    meta_channels: dict,
    gating_plan: dict,
    gating_plan_path: Path,
    output_dir: Path,
) -> None:
    df, expr_cols, _ = load_sample(sample_path, meta_channels, config.get("category_map", {}))
    sample_id = sample_path.stem

    task = build_task_json(
        dataset_name, sample_id, df, gating_plan, gating_plan_path, config, expr_cols
    )

    plan_slug = task["plan_slug"]
    plan_dir = output_dir / sample_id / plan_slug
    plan_dir.mkdir(parents=True, exist_ok=True)
    with open(plan_dir / "task.json", "w") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    print(
        f"  {sample_id} [{plan_slug}]: n_cells={len(df):,}, "
        f"leaves={len(task['leaf_categories'])}, "
        f"steps={len(task['steps'])}"
    )


def process_dataset(
    dataset_dir: Path,
    output_base: Path,
    gating_plan_path: Path,
    blacklist: set[str] | None = None,
    allowed: set[str] | None = None,
) -> None:
    parquet_dir = dataset_dir / "parquet"
    if not (parquet_dir / "meta.json").exists():
        print(f"[SKIP] No parquet/meta.json in {dataset_dir}")
        return

    if not gating_plan_path.exists():
        print(f"[SKIP] gating plan not found: {gating_plan_path}")
        return

    config, _, meta_channels = load_config(dataset_dir)
    dataset_name = config["name"]
    with open(gating_plan_path) as f:
        gating_plan = json.load(f)

    leaf_set = derive_leaf_set(gating_plan)
    print(f"  gating plan: {gating_plan_path} ({len(leaf_set)} leaves)")
    if not leaf_set:
        print(f"[WARN] {gating_plan_path}: derive_leaf_set produced 0 leaves — check the plan")

    sample_files = sorted(parquet_dir.glob("*.parquet"))
    if allowed is not None:
        missing = sorted(allowed - {p.stem for p in sample_files})
        if missing:
            print(f"[WARN] {dataset_name}: {len(missing)} split sample(s) missing")
        sample_files = [p for p in sample_files if p.stem in allowed]
    if blacklist:
        sample_files = [p for p in sample_files if p.stem not in blacklist]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} ({len(sample_files)} samples)")
    print(f"{'='*60}")

    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_path in sample_files:
        process_sample(
            sample_path, dataset_name, config, meta_channels,
            gating_plan, gating_plan_path, output_dir,
        )


def _resolve_gating_plan(args, dataset_name: str) -> Path:
    """Pick the gating plan to use for this dataset.

    Resolution order:
      1. ``--gating-plan`` (explicit override)
      2. ``benchmark_flat/<dataset>/gating_plan.json`` (canonical location —
         keep gating plans alongside the per-sample task.json output so
         every flat tool reads from the same place)
      3. ``data/<dataset>/gating_plan.json`` (legacy fallback)
    """
    if args.gating_plan is not None:
        return args.gating_plan
    bench_default = args.output_dir / dataset_name / "gating_plan.json"
    if bench_default.exists():
        return bench_default
    return args.data_dir / dataset_name / "gating_plan.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="flowsom_c2s task.json generator")
    parser.add_argument("--data-dir", type=Path, default=Path("data/"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_flat/"))
    parser.add_argument("--datasets", nargs="*", help="Specific dataset names")
    parser.add_argument("--samples", nargs="*", help="Restrict to these sample IDs")
    parser.add_argument("--gating-plan", type=Path, default=None,
                        help="Path to gating_plan.json (default: data/<dataset>/gating_plan.json). "
                             "Pass an L1/L2/L3 variant for the matching difficulty level.")
    parser.add_argument("--splits", type=Path, default=None,
                        help="Path to splits.json (default: <output-dir>/splits.json or benchmark/splits.json)")
    args = parser.parse_args()

    dataset_dirs = sorted(args.data_dir.iterdir())
    dataset_dirs = [d for d in dataset_dirs if d.is_dir() and (d / "parquet" / "meta.json").exists()]
    if args.datasets:
        dataset_dirs = [d for d in dataset_dirs if d.name in args.datasets]
    print(f"Found {len(dataset_dirs)} datasets: {[d.name for d in dataset_dirs]}")

    splits_path = args.splits or (args.output_dir / "splits.json")
    if not splits_path.exists():
        splits_path = Path("benchmark/splits.json")

    blacklists: dict[str, set[str]] = {}
    allowed: dict[str, set[str]] = {}
    if splits_path.exists():
        print(f"Using splits: {splits_path}")
        with open(splits_path) as f:
            splits = json.load(f)
        for panel_group in splits.values():
            if not isinstance(panel_group, dict):
                continue
            for cohort, info in panel_group.items():
                if not isinstance(info, dict):
                    continue
                if info.get("blacklist"):
                    blacklists.setdefault(cohort, set()).update(info["blacklist"])
                cohort_allowed: set[str] = set()
                for split_key in ("train", "val", "test"):
                    cohort_allowed.update(info.get(split_key, []) or [])
                if cohort_allowed:
                    allowed.setdefault(cohort, set()).update(cohort_allowed)

    if args.samples:
        # --samples is an explicit override: process only the listed IDs,
        # ignoring whatever splits.json has for these datasets.
        explicit = set(args.samples)
        for ds_dir in dataset_dirs:
            allowed[ds_dir.name] = explicit

    for dataset_dir in dataset_dirs:
        gp_path = _resolve_gating_plan(args, dataset_dir.name)
        process_dataset(
            dataset_dir,
            args.output_dir,
            gating_plan_path=gp_path,
            blacklist=blacklists.get(dataset_dir.name),
            allowed=allowed.get(dataset_dir.name),
        )


if __name__ == "__main__":
    main()
