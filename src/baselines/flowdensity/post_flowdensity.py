"""
post_flowdensity.py — Stage 3 of the flowDensity pipeline.

Reads per-step outputs from the R stage (thresholds.json + input.csv + meta.json),
assigns cells to rectangular regions, applies Hungarian matching, and writes
prediction.json.

Input  (per step): results/{dataset}/{sample}/step_NN/input.csv
                   results/{dataset}/{sample}/step_NN/meta.json
                   results/{dataset}/{sample}/step_NN/thresholds.json
Output (per step): results/{dataset}/{sample}/step_NN/prediction.json

Usage:
    python -m src.baselines.flowdensity.post_flowdensity \\
        --results-dir results/flowdensity/

    python -m src.baselines.flowdensity.post_flowdensity \\
        --results-dir results/flowdensity/ --overwrite
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hungarian matching
# ---------------------------------------------------------------------------


def hungarian_match(
    region_ids: np.ndarray,
    gt_labels: np.ndarray,
    categories: list[str],
) -> dict[int, str | None]:
    """Greedy region → GT category mapping by majority vote (many-to-one)."""
    if len(region_ids) == 0:
        return {}
    ct = pd.crosstab(region_ids, gt_labels).reindex(columns=categories, fill_value=0)
    best_cat = ct.idxmax(axis=1)
    max_count = ct.max(axis=1)
    return {
        int(rid): (cat if cnt > 0 else None)
        for rid, cat, cnt in zip(best_cat.index, best_cat.values, max_count.values)
    }


# ---------------------------------------------------------------------------
# Region assignment
# ---------------------------------------------------------------------------


def assign_regions(
    x: np.ndarray,
    y: np.ndarray,
    x_thresholds: list[float],
    y_thresholds: list[float],
) -> np.ndarray:
    """Map each cell to a unique integer region ID based on threshold grid.

    Example: x_thresholds=[t1], y_thresholds=[t2]
      → 4 regions (0,0), (0,1), (1,0), (1,1) encoded as 0,1,2,3
    """
    x_bins = np.digitize(x, sorted(x_thresholds))
    y_bins = np.digitize(y, sorted(y_thresholds))
    n_y_bins = len(y_thresholds) + 1
    return x_bins * n_y_bins + y_bins


# ---------------------------------------------------------------------------
# Per-step post-processing
# ---------------------------------------------------------------------------


def postprocess_step(step_dir: Path, overwrite: bool = False, cleanup: bool = False) -> str:
    """Process one step directory. Returns 'written', 'skipped', or 'error:...'."""
    pred_path       = step_dir / "prediction.json"
    input_csv_path  = step_dir / "input.csv"
    meta_path       = step_dir / "meta.json"
    thresholds_path = step_dir / "thresholds.json"

    if not overwrite and pred_path.exists():
        if cleanup and input_csv_path.exists():
            input_csv_path.unlink()
        return "skipped"

    for p in (input_csv_path, meta_path):
        if not p.exists():
            return f"error: missing {p.name}"

    df = pd.read_csv(input_csv_path)
    with open(meta_path) as f:
        meta = json.load(f)

    # The R stage emits thresholds.json for nrow >= 1 (using a median
    # fallback when deGate cannot find density valleys at very small n).
    # If thresholds.json is missing the R stage skipped (e.g. empty input),
    # so emit an all-None prediction — eval scores the step (as all-wrong)
    # instead of dropping it.
    if not thresholds_path.exists():
        prediction = {
            "task_id": meta["task_id"],
            "method": "flowdensity",
            "predicted_labels": [None] * len(df),
            "info": {"skipped": "missing thresholds.json", "n_parent": len(df)},
        }
        with open(pred_path, "w") as f:
            json.dump(prediction, f, indent=2, ensure_ascii=False)
        if cleanup:
            input_csv_path.unlink()
        return "written"

    with open(thresholds_path) as f:
        thresholds = json.load(f)

    x_vals = df["x"].values.astype(float)
    y_vals = df["y"].values.astype(float)
    gt_labels = df["gt_label"].values.astype(str)

    x_thresholds = [float(t) for t in thresholds["x_thresholds"]]
    y_thresholds = [float(t) for t in thresholds["y_thresholds"]]

    region_ids = assign_regions(x_vals, y_vals, x_thresholds, y_thresholds)
    mapping    = hungarian_match(region_ids, gt_labels, meta["categories"])
    pred_labels = pd.Series(region_ids).map(mapping).to_numpy(dtype=object)

    prediction = {
        "task_id": meta["task_id"],
        "method": "flowdensity",
        "predicted_labels": pred_labels.tolist(),
        "info": {
            "x_thresholds": x_thresholds,
            "y_thresholds": y_thresholds,
            "n_regions": len(set(region_ids)),
            "region_mapping": {str(k): v for k, v in mapping.items()},
        },
    }
    with open(pred_path, "w") as f:
        json.dump(prediction, f, indent=2, ensure_ascii=False)

    if cleanup:
        input_csv_path.unlink()

    return "written"


# ---------------------------------------------------------------------------
# Batch over results_dir
# ---------------------------------------------------------------------------


def process_results_dir(results_dir: Path, overwrite: bool = False, cleanup: bool = False):
    step_dirs = sorted(results_dir.rglob("step_*/"))
    # filter to dirs that have at least input.csv or prediction.json
    step_dirs = [d for d in step_dirs if (d / "input.csv").exists() or (d / "prediction.json").exists()]

    print(f"Found {len(step_dirs)} step directories under {results_dir}")

    n_written = n_skipped = n_error = 0
    for step_dir in step_dirs:
        status = postprocess_step(step_dir, overwrite, cleanup)
        if status == "written":
            n_written += 1
        elif status == "skipped":
            n_skipped += 1
        else:
            print(f"  [{status}] {step_dir}")
            n_error += 1

    print(f"\nDone. Written: {n_written}  Skipped: {n_skipped}  Errors: {n_error}")


def main():
    parser = argparse.ArgumentParser(description="flowDensity pipeline — Stage 3: post-process")
    parser.add_argument("--results-dir", type=Path, default=Path("results/flowdensity/"),
                        help="Root of flowdensity results (output of pre + R stages)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-write existing prediction.json files")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete input.csv after prediction.json is written")
    args = parser.parse_args()

    print("flowDensity pipeline — Stage 3: post")
    process_results_dir(args.results_dir, args.overwrite, args.cleanup)


if __name__ == "__main__":
    main()
