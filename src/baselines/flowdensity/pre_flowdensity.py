"""
pre_flowdensity.py — Stage 1 of the flowDensity pipeline.

Reads parquet data and benchmark task definitions (via BenchmarkLoader), then
writes per-step inputs into the results directory for the R gating stage.

Output per step (results/{dataset}/{sample}/step_NN/):
  input.csv   — x, y, gt_label columns (parent population cells)
  meta.json   — task_id, categories

Usage:
    python -m src.baselines.flowdensity.pre_flowdensity \\
        --benchmark benchmark/ \\
        --data-dir data/ \\
        --output-dir results/flowdensity/ \\
        --split benchmark/splits.json --split-set test
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.bench import BenchmarkLoader, TaskInput


def write_step(ti: TaskInput, out_dir: Path, overwrite: bool = False) -> bool:
    """Write input.csv + meta.json for one task. Returns True if written."""
    input_csv = out_dir / "input.csv"
    meta_json = out_dir / "meta.json"

    if not overwrite and input_csv.exists() and meta_json.exists():
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    X = ti.X_2d
    pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "gt_label": ti.y.astype(str)}).to_csv(
        input_csv, index=False
    )
    with open(meta_json, "w") as f:
        json.dump({"task_id": ti.task_id, "categories": ti.categories}, f, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(description="flowDensity pipeline — Stage 1: prepare inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("results/flowdensity/"))
    parser.add_argument("--overwrite", action="store_true", help="Re-write existing inputs")
    BenchmarkLoader.add_cli_args(parser)
    args = parser.parse_args()

    loader = BenchmarkLoader.from_cli(args)
    print(f"flowDensity pipeline — Stage 1: pre → {args.output_dir}")

    n_written = n_skipped = 0
    current_ds = current_sample = None
    for ti in loader.iter_tasks_from_cli(args):
        if ti.dataset != current_ds:
            current_ds, current_sample = ti.dataset, None
            print(f"\n{'='*60}\nDataset: {ti.dataset}\n{'='*60}")
        if ti.sample != current_sample:
            current_sample = ti.sample
            print(f"  {ti.sample}")

        out_dir = args.output_dir / ti.dataset / ti.sample / ti.step_dir_name
        if write_step(ti, out_dir, args.overwrite):
            n_written += 1
        else:
            n_skipped += 1

    print(f"\nWritten: {n_written}  Skipped: {n_skipped}")
    print(f"Next: Rscript src/baselines/flowdensity/flowdensity_gate.R --results-dir {args.output_dir}")


if __name__ == "__main__":
    main()
