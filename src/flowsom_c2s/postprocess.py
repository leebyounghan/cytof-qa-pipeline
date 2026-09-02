"""Decompose flowsom_c2s pred.json -> per-sample ``predicted_steps.parquet``.

Per cluster, the LLM picked a SET of leaves (multi-label). For each leaf
in that set we populate the matching Step* column for all cells in the
cluster — so the resulting parquet is consumable by ``src.flat_eval``
exactly like ``src/llm_gate_flat/runner.py``'s output.

Cells whose cluster picked ``["Discard"]`` (or an empty/invalid leaf list)
get NaN at every leaf-step column → ``src.flat_eval`` reduces them to the
synthetic ``Discard`` class.

Usage:
    python -m src.flowsom_c2s.postprocess \\
        --pred results_flat/flowsom_<method>/<dataset>/pred.json \\
        --benchmark-flat benchmark_flat/ \\
        --output-dir results_flat/flowsom_<method>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _slot_step_column_values(
    pred_clusters: dict,
    leaf_to_step_col: dict[str, str],
    n_slots: int,
    step_cols: list[str],
) -> tuple[dict[str, np.ndarray], dict]:
    """For each step column, build an (n_slots,) object array whose k-th entry
    is the leaf the LLM picked for that slot at that step (or None).

    Multi-label inversion: cluster k's ``leaves`` list contains 0+ leaves;
    each populates exactly one step column (per ``leaf_to_step_col``).
    """
    out: dict[str, np.ndarray] = {c: np.full(n_slots, None, dtype=object) for c in step_cols}
    n_invalid_clusters = 0
    n_offcol = 0
    for k in range(n_slots):
        entry = pred_clusters.get(f"cluster_{k+1}")
        if not entry:
            n_invalid_clusters += 1
            continue
        leaves = entry.get("leaves") or []
        if not leaves:
            n_invalid_clusters += 1
            continue
        for leaf in leaves:
            col = leaf_to_step_col.get(leaf)
            if col is None:
                # Leaf not mapped to a step column — e.g. "Discard" sentinel.
                # Discard intentionally has no column; skip silently.
                if leaf != "Discard":
                    n_offcol += 1
                continue
            if col not in out:
                n_offcol += 1
                continue
            # If two leaves try to populate the same column for one cluster,
            # keep the first (rare; could happen on noisy LLM output).
            if out[col][k] is None:
                out[col][k] = leaf
    return out, {"n_invalid_clusters": n_invalid_clusters, "n_offcol_leaves": n_offcol}


def process(
    pred: dict,
    benchmark_flat_dir: Path,
    output_dir: Path,
    plan_slug: str,
    dataset_filter: list[str] | None = None,
) -> None:
    samples_pred = pred.get("samples", {})
    benchmark_label = pred.get("meta", {}).get("benchmark", "")
    inferred_ds = Path(benchmark_label).name if benchmark_label else None

    if not samples_pred:
        print("[WARN] pred.json has no samples")
        return

    n_samples_written = 0
    for sample, sample_pred in samples_pred.items():
        if dataset_filter:
            candidates = dataset_filter
        elif inferred_ds:
            candidates = [inferred_ds]
        else:
            candidates = [d.name for d in benchmark_flat_dir.iterdir() if d.is_dir()]

        sample_dir = None
        dataset = None
        for ds in candidates:
            cand = benchmark_flat_dir / ds / sample
            if cand.is_dir():
                sample_dir = cand
                dataset = ds
                break
        if sample_dir is None:
            print(f"[SKIP] no benchmark_flat dir for sample {sample!r}")
            continue

        task_path = sample_dir / plan_slug / "task.json"
        c2c_path = sample_dir / "cell2cluster.npz"
        if not (task_path.exists() and c2c_path.exists()):
            print(f"[SKIP] missing task.json (plan={plan_slug}) / cell2cluster.npz under {sample_dir}")
            continue
        with open(task_path) as f:
            task = json.load(f)
        leaf_to_step_col = task["leaf_to_step_col"]
        step_cols = [s["annotation_column"] for s in task["steps"]]

        npz = np.load(c2c_path)
        cluster_labels = npz["cluster_labels"]
        n_cells = int(cluster_labels.size)
        n_slots = int(cluster_labels.max()) + 1 if n_cells else 0

        slot_step, stats = _slot_step_column_values(
            sample_pred.get("clusters", {}),
            leaf_to_step_col,
            n_slots,
            step_cols,
        )

        cols_data: dict[str, np.ndarray] = {
            c: slot_step[c][cluster_labels] for c in step_cols
        }

        out_sample_dir = output_dir / dataset / sample
        out_sample_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(cols_data)
        df.to_parquet(out_sample_dir / "predicted_steps.parquet", index=False)

        with open(out_sample_dir / "postprocess_meta.json", "w") as f:
            json.dump(
                {
                    "n_cells": n_cells,
                    "n_slots": n_slots,
                    "n_invalid_clusters": stats["n_invalid_clusters"],
                    "n_offcol_leaves": stats["n_offcol_leaves"],
                },
                f,
                indent=2,
            )
        n_samples_written += 1
        print(
            f"  {dataset}/{sample}: n_cells={n_cells:,}, K={n_slots}, "
            f"invalid_clusters={stats['n_invalid_clusters']}, "
            f"offcol_leaves={stats['n_offcol_leaves']}"
        )

    print(f"\nWrote {n_samples_written} predicted_steps.parquet under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="flowsom_c2s post-processing -> predicted_steps.parquet")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--benchmark-flat", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Base output dir; <plan_slug> is auto-appended.")
    parser.add_argument("--gating-plan", type=Path, default=None,
                        help="Used to auto-derive plan_slug for input task.json + output dir")
    parser.add_argument("--plan-slug", type=str, default=None,
                        help="Override plan slug (default: auto from --gating-plan, else 'default')")
    parser.add_argument("--datasets", nargs="*")
    args = parser.parse_args()

    with open(args.pred) as f:
        pred = json.load(f)

    from src.flat_eval import gating_plan_slug as _slug
    if args.plan_slug:
        plan_slug = args.plan_slug
    elif args.gating_plan is not None:
        plan_slug = _slug(args.gating_plan)
    else:
        plan_slug = "default"

    out_root = args.output_dir / plan_slug
    print(f"plan_slug={plan_slug} -> writing under {out_root}")

    process(
        pred=pred,
        benchmark_flat_dir=args.benchmark_flat,
        output_dir=out_root,
        plan_slug=plan_slug,
        dataset_filter=args.datasets,
    )


if __name__ == "__main__":
    main()
