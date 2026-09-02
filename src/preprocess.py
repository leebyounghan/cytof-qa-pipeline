"""
Benchmark task generator for parquet-based cytometry datasets.

Reads parquet files + meta.json + gating_plan.json, generates per-sample
per-step task.json and biaxial plot images.

Usage:
    python -m src.preprocess --data-dir data/ --output-dir benchmark/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(dataset_dir: Path) -> tuple[dict, dict, dict]:
    """Load meta.json and gating_plan.json. Returns (config, gating_plan, meta_channels)."""
    with open(dataset_dir / "parquet" / "meta.json") as f:
        meta = json.load(f)
    with open(dataset_dir / "gating_plan.json") as f:
        gating_plan = json.load(f)

    channel_marker_map = {
        ch: info[0] for ch, info in meta["channels"].items()
    }

    config = {
        "name": dataset_dir.name,
        "cofactor": meta.get("cofactor"),
        "channel_marker_map": channel_marker_map,
        "category_map": meta.get("category_map", {}),
    }
    return config, gating_plan, meta["channels"]


def load_sample(path: Path, meta_channels: dict, category_map: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load a parquet sample. Returns (df, expr_cols, obs_cols).

    - Applies category_map to Step columns.
    - Replaces empty strings with NaN in Step columns.
    """
    df = pd.read_parquet(path)

    expr_cols = [c for c in df.columns if c in meta_channels]
    obs_cols = [c for c in df.columns if c.startswith("Step")]

    # Normalize label columns: empty string → NaN, apply category_map
    for col in obs_cols:
        s = df[col].astype(object)
        s.replace("", np.nan, inplace=True)
        if category_map:
            s.replace(category_map, inplace=True)
        df[col] = s

    return df, expr_cols, obs_cols


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_channel(marker: str, var_names: list[str], channel_marker_map: dict) -> str | None:
    """Resolve a marker name to the actual channel name.

    If marker is already a column name, return as-is.
    Otherwise, reverse-lookup through channel_marker_map.
    """
    if marker in var_names:
        return marker
    for channel, mapped_marker in channel_marker_map.items():
        if mapped_marker == marker and channel in var_names:
            return channel
    return None


def get_parent_mask(df: pd.DataFrame, parent_str: str) -> np.ndarray:
    """Get boolean mask for cells belonging to a step's parent population."""
    n = len(df)
    if parent_str == "ALL":
        return np.ones(n, dtype=bool)

    mask = np.ones(n, dtype=bool)
    for part in parent_str.split("&"):
        part = part.strip()
        if "==" in part:
            col, val = [x.strip() for x in part.split("==")]
            mask &= (df[col].values == val) if col in df.columns else np.zeros(n, dtype=bool)
        elif "!=" in part:
            col, val = [x.strip() for x in part.split("!=")]
            mask &= (df[col].values != val) if col in df.columns else np.zeros(n, dtype=bool)
        elif " in " in part:
            col, vals_str = [x.strip() for x in part.split(" in ")]
            vals = [v.strip() for v in vals_str.split(",")]
            mask &= np.isin(df[col].values, vals) if col in df.columns else np.zeros(n, dtype=bool)
    return mask


def get_gt_labels(df: pd.DataFrame, mask: np.ndarray, col: str) -> np.ndarray:
    """Get GT labels for parent population, replacing NaN with 'Unassigned'."""
    labels = df.loc[mask, col].values.copy()
    labels = np.array(labels, dtype=object)
    for i, v in enumerate(labels):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            labels[i] = "Unassigned"
    return labels


def compute_gt_summary(n_parent: int, step: dict, labels: np.ndarray) -> dict:
    """Compute ground truth summary statistics."""
    categories = step["annotation categories"] + ["Unassigned"]
    n_cells_per_cat = {cat: int(np.sum(labels == cat)) for cat in categories}
    proportions = {
        cat: round(n / n_parent, 4) if n_parent > 0 else 0.0
        for cat, n in n_cells_per_cat.items()
    }
    return {
        "n_parent_cells": n_parent,
        "n_cells_per_category": n_cells_per_cat,
        "population_proportions": proportions,
    }


def build_task_json(
    dataset_name: str, sample_id: str, step: dict,
    gt_summary: dict, x_marker: str, y_marker: str, gating_context: str,
) -> dict:
    step_num = step["step"]
    return {
        "task_id": f"{dataset_name}__{sample_id}__step_{step_num:02d}",
        "dataset": dataset_name,
        "sample": sample_id,
        "step": step_num,
        "parent": step["parent"],
        "x_marker": x_marker,
        "y_marker": y_marker,
        "annotation_column": step["annotation column name"],
        "categories": list(gt_summary["n_cells_per_category"].keys()),
        "gating_context": gating_context,
        "gt_summary": gt_summary,
    }


def build_gating_context(steps: list[dict], current_step: dict) -> str:
    parent_str = current_step["parent"]
    if parent_str == "ALL":
        return "Root"

    conditions = []
    for part in parent_str.split("&"):
        part = part.strip()
        if "==" in part:
            col, val = [x.strip() for x in part.split("==")]
            conditions.append(f"{col}({val})")
        elif " in " in part:
            col, vals = [x.strip() for x in part.split(" in ")]
            conditions.append(f"{col}({vals})")
        elif "!=" in part:
            col, val = [x.strip() for x in part.split("!=")]
            conditions.append(f"{col}(not {val})")
    return "Root -> " + " -> ".join(conditions)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def compute_density(x: np.ndarray, y: np.ndarray, bins: int = 200) -> np.ndarray:
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    return hist[xi, yi]


def render_biaxial_plot(
    x_vals: np.ndarray, y_vals: np.ndarray,
    x_marker: str, y_marker: str, output_path: Path,
    labels: np.ndarray | None = None,
    categories: list[str] | None = None,
    title: str = "",
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)

    if labels is not None and categories is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(categories), 10)))
        for i, cat in enumerate(categories):
            cat_mask = labels == cat
            if np.sum(cat_mask) > 0:
                ax.scatter(
                    x_vals[cat_mask], y_vals[cat_mask],
                    s=1, alpha=0.3, c=[colors[i % len(colors)]],
                    label=f"{cat} ({np.sum(cat_mask):,})", rasterized=True,
                )
        ax.legend(fontsize=7, markerscale=5, loc="best")
    else:
        density = compute_density(x_vals, y_vals)
        order = np.argsort(density)
        ax.scatter(
            x_vals[order], y_vals[order],
            s=1, alpha=0.3, c=density[order], cmap="viridis", rasterized=True,
        )

    ax.set_xlabel(x_marker, fontsize=10)
    ax.set_ylabel(y_marker, fontsize=10)
    if title:
        ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------


def process_sample(
    df: pd.DataFrame,
    expr_cols: list[str],
    sample_id: str,
    dataset_name: str,
    channel_marker_map: dict,
    gating_plan: dict,
    output_dir: Path,
    no_plots: bool = False,
) -> int:
    """Process a single sample: generate task.json + plots for all steps."""
    steps = gating_plan["steps"]
    sample_dir = output_dir / sample_id
    n_steps_created = 0

    for step in steps:
        step_num = step["step"]
        col = step["annotation column name"]

        if col not in df.columns:
            continue

        mask = get_parent_mask(df, step["parent"])
        n_parent = int(np.sum(mask))
        if n_parent == 0:
            continue

        # Resolve marker → column
        x_marker = step["x_marker"]
        y_marker = step["y_marker"]
        x_ch = resolve_channel(x_marker, expr_cols, channel_marker_map)
        y_ch = resolve_channel(y_marker, expr_cols, channel_marker_map)
        if x_ch is None or y_ch is None:
            print(f"  [SKIP] step {step_num:02d}: cannot resolve markers {x_marker}/{y_marker}")
            continue

        x_vals = df.loc[mask, x_ch].values.astype(np.float32)
        y_vals = df.loc[mask, y_ch].values.astype(np.float32)

        labels = get_gt_labels(df, mask, col)
        gt_summary = compute_gt_summary(n_parent, step, labels)
        gating_context = build_gating_context(steps, step)

        task = build_task_json(
            dataset_name, sample_id, step, gt_summary, x_marker, y_marker, gating_context
        )

        step_dir = sample_dir / f"step_{step_num:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        with open(step_dir / "task.json", "w") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)

        gt_categories = list(gt_summary["n_cells_per_category"].keys())
        if not no_plots:
            render_biaxial_plot(x_vals, y_vals, x_marker, y_marker, step_dir / "biaxial_plot.png")
            render_biaxial_plot(
                x_vals, y_vals, x_marker, y_marker,
                step_dir / "biaxial_plot_gt.png",
                labels=labels, categories=gt_categories,
            )

        n_steps_created += 1

    print(f"  {sample_id}: {n_steps_created} steps created")
    return n_steps_created


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def process_dataset(
    dataset_dir: Path,
    output_base: Path,
    blacklist: set[str] | None = None,
    allowed: set[str] | None = None,
    no_plots: bool = False,
    n_workers: int = 1,
):
    """Process all samples in a dataset directory.

    If `allowed` is provided, only samples whose stem is in that set are
    processed (train + val + test from splits.json). `blacklist` is still
    honored as a safety net.
    """
    parquet_dir = dataset_dir / "parquet"
    if not (parquet_dir / "meta.json").exists():
        print(f"[SKIP] No parquet/meta.json in {dataset_dir}")
        return

    config, gating_plan, meta_channels = load_config(dataset_dir)
    dataset_name = config["name"]
    channel_marker_map = config.get("channel_marker_map", {})
    category_map = config.get("category_map", {})
    sample_files = sorted(parquet_dir.glob("*.parquet"))

    if allowed is not None:
        missing = sorted(allowed - {p.stem for p in sample_files})
        if missing:
            print(f"[WARN] {dataset_name}: {len(missing)} split sample(s) missing from parquet: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        dropped = [p.stem for p in sample_files if p.stem not in allowed]
        sample_files = [p for p in sample_files if p.stem in allowed]
        if dropped:
            print(f"[SKIP] {dataset_name}: {len(dropped)} sample(s) not listed in splits.json")

    if blacklist:
        skipped = [p.stem for p in sample_files if p.stem in blacklist]
        sample_files = [p for p in sample_files if p.stem not in blacklist]
        for s in skipped:
            print(f"[SKIP] {dataset_name}/{s} (blacklisted)")

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} ({len(sample_files)} samples)")
    print(f"{'='*60}")

    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_meta = {
        "name": dataset_name,
        "panel": gating_plan.get("panel", ""),
        "instrument": gating_plan.get("instrument", ""),
        "n_steps": len(gating_plan["steps"]),
        "n_samples": len(sample_files),
        "cofactor": config.get("cofactor"),
        "channel_marker_map": channel_marker_map,
        "gating_plan": gating_plan["steps"],
    }
    with open(output_dir / "panel_meta.yaml", "w") as f:
        yaml.dump(panel_meta, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    total_steps = 0
    if n_workers and n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # Per-worker job: load + process one sample. matplotlib is process-safe
        # only when each worker uses its own backend; we do so via fork (default).
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {
                ex.submit(
                    _process_one_sample, str(sp), str(output_dir),
                    str(dataset_dir / 'parquet' / 'meta.json'),
                    str(dataset_dir / 'gating_plan.json'),
                    dataset_name, no_plots,
                ): sp.stem for sp in sample_files
            }
            for fut in as_completed(futs):
                total_steps += fut.result()
    else:
        for sample_path in sample_files:
            df, expr_cols, obs_cols = load_sample(sample_path, meta_channels, category_map)
            sample_id = sample_path.stem
            n = process_sample(
                df, expr_cols, sample_id, dataset_name,
                channel_marker_map, gating_plan, output_dir,
                no_plots=no_plots,
            )
            total_steps += n

    print(f"\nTotal: {len(sample_files)} samples, {total_steps} task instances")


def _process_one_sample(parquet_path: str, output_dir: str,
                        meta_path: str, gating_plan_path: str,
                        dataset_name: str, no_plots: bool = False) -> int:
    """Re-load config + sample inside a worker process (avoids pickling
    large dataframes / gating plans across the pool).
    """
    from pathlib import Path
    parquet_path = Path(parquet_path)
    output_dir = Path(output_dir)
    with open(meta_path) as f:
        meta = json.load(f)
    with open(gating_plan_path) as f:
        gating_plan = json.load(f)
    channel_marker_map = {ch: info[0] for ch, info in meta['channels'].items()}
    category_map = meta.get('category_map', {})
    df, expr_cols, _ = load_sample(parquet_path, meta['channels'], category_map)
    return process_sample(
        df, expr_cols, parquet_path.stem, dataset_name,
        channel_marker_map, gating_plan, output_dir,
        no_plots=no_plots,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark tasks")
    parser.add_argument("--data-dir", type=Path, default=Path("data/"), help="Root data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/"), help="Output directory")
    parser.add_argument("--datasets", nargs="*", help="Specific dataset names to process (default: all)")
    parser.add_argument("--splits", type=Path, default=None, help="Path to splits.json (default: <output-dir>/splits.json)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel sample-level workers via ProcessPoolExecutor (default: 1).")
    parser.add_argument("--no-plots", action="store_true", help="Skip biaxial_plot.png / biaxial_plot_gt.png rendering")
    args = parser.parse_args()

    dataset_dirs = sorted(args.data_dir.iterdir())
    dataset_dirs = [
        d for d in dataset_dirs
        if d.is_dir() and (d / "parquet" / "meta.json").exists()
    ]

    if args.datasets:
        dataset_dirs = [d for d in dataset_dirs if d.name in args.datasets]

    print(f"Found {len(dataset_dirs)} datasets: {[d.name for d in dataset_dirs]}")

    splits_path = args.splits or (args.output_dir / "splits.json")
    blacklists: dict[str, set[str]] = {}
    allowed: dict[str, set[str]] = {}
    if splits_path.exists():
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
                cohort_allowed = set()
                for split_key in ("train", "val", "test"):
                    cohort_allowed.update(info.get(split_key, []) or [])
                if cohort_allowed:
                    allowed.setdefault(cohort, set()).update(cohort_allowed)

    for dataset_dir in dataset_dirs:
        process_dataset(
            dataset_dir,
            args.output_dir,
            blacklist=blacklists.get(dataset_dir.name),
            allowed=allowed.get(dataset_dir.name),
            no_plots=args.no_plots,
            n_workers=args.n_workers,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
