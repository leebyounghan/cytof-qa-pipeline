"""
Evaluation script for CytoGate-Bench.

Compares prediction.json files against ground truth from parquet files.
Method-agnostic: any baseline that outputs the standard prediction format
can be evaluated with this script.

Prediction format (prediction.json):
{
  "task_id": "...",
  "method": "flowdensity",
  "predicted_labels": ["CD4", "CD8", "Unassigned", ...],  # per-cell labels
}

Usage:
    python -m src.eval --predictions results/flowdensity/ --benchmark benchmark/ --data-dir data/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.bench import BenchmarkLoader, Perturbation, parse_perturbations  # noqa: F401


def _load_meta(data_dir: Path, dataset: str) -> dict:
    """Load meta.json for a dataset."""
    with open(data_dir / dataset / "parquet" / "meta.json") as f:
        meta = json.load(f)
    return meta


def load_gt_labels(
    data_dir: Path,
    task: dict,
    benchmark_dir: Path | None = None,
    perturbation: Perturbation | None = None,
) -> np.ndarray:
    """Load ground truth labels from parquet for a task.

    When a perturbation is given, applies the same sample-level cell filter
    used by baselines (via BenchmarkLoader) so that GT length matches
    the prediction length.
    """
    if perturbation is not None:
        loader = BenchmarkLoader(
            benchmark_dir or Path("benchmark/"), data_dir, perturbation=perturbation
        )
        df, _ = loader._load_sample(task["dataset"], task["sample"])
        mask = loader._parent_mask(df, task["parent"])
        col = task["annotation_column"]
        labels = (
            np.array(df.loc[mask, col].values, dtype=object)
            if col in df.columns
            else np.array(["Unassigned"] * int(mask.sum()), dtype=object)
        )
    else:
        parquet_path = data_dir / task["dataset"] / "parquet" / f"{task['sample']}.parquet"
        df = pd.read_parquet(parquet_path)
        meta = _load_meta(data_dir, task["dataset"])
        category_map = meta.get("category_map", {})
        col = task["annotation_column"]
        s = df[col].astype(object)
        s.replace("", np.nan, inplace=True)
        if category_map:
            s.replace(category_map, inplace=True)
        df[col] = s
        mask = _parse_parent_mask(df, task["parent"])
        labels = df.loc[mask, col].fillna("Unassigned").to_numpy(dtype=object)

    na_mask = pd.isna(labels)
    if na_mask.any():
        labels[na_mask] = "Unassigned"
    return labels


def _prepare_sample_df(
    data_dir: Path,
    dataset: str,
    sample: str,
    benchmark_dir: Path | None = None,
    perturbation: Perturbation | None = None,
    loader: "BenchmarkLoader | None" = None,
) -> tuple[pd.DataFrame, dict, "BenchmarkLoader | None"]:
    """Load and preprocess a sample's parquet once (all Step* label columns mapped,
    perturbation applied). Returns (df, ch_map, loader). Pass the returned loader
    back in to reuse its internal sample cache across calls.
    """
    meta = _load_meta(data_dir, dataset)
    ch_map = {ch: info[0] for ch, info in meta["channels"].items()}

    if perturbation is not None:
        if loader is None:
            loader = BenchmarkLoader(
                benchmark_dir or Path("benchmark/"), data_dir, perturbation=perturbation
            )
        df, _ = loader._load_sample(dataset, sample)
        _categoricalize_step_cols(df, category_map=None)
    else:
        parquet_path = data_dir / dataset / "parquet" / f"{sample}.parquet"
        df = pd.read_parquet(parquet_path)
        _categoricalize_step_cols(df, category_map=meta.get("category_map") or None)

    return df, ch_map, loader


def _categoricalize_step_cols(df: pd.DataFrame, category_map: dict | None) -> None:
    """Normalize Step* columns to Categorical in a single pass per column.

    Replaces the previous object→replace→object sequence; folding the empty-
    string normalization and category_map rename directly into the category
    index instead of running two prior full-column replace passes.

    "Unassigned" is pre-added so downstream `.fillna("Unassigned")` is O(n_na)
    instead of touching the dtype.
    """
    step_cols = [c for c in df.columns if c.startswith("Step")]
    for c in step_cols:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cat = s.values  # already Categorical (perturbation path may pre-normalize)
            # Rewrite "" → NaN / apply category_map inside the category index only.
            cats = cat.categories
            rename: dict = {}
            remove: list = []
            for v in cats:
                if v == "":
                    remove.append(v)
                elif category_map and v in category_map:
                    rename[v] = category_map[v]
            if rename:
                cat = cat.rename_categories(rename)
            if remove:
                cat = cat.remove_categories(remove)
        else:
            # Object column from parquet: factorize once. pd.Categorical handles
            # empty strings as regular categories; we drop them post-hoc so the
            # underlying codes reflect "" rows as -1 (NaN), matching the old
            # replace("", np.nan) semantics.
            cat = pd.Categorical(s)
            if "" in cat.categories:
                cat = cat.remove_categories([""])
            if category_map:
                rename = {v: category_map[v] for v in cat.categories if v in category_map}
                if rename:
                    cat = cat.rename_categories(rename)
        if "Unassigned" not in cat.categories:
            cat = cat.add_categories(["Unassigned"])
        df[c] = cat


def _extract_labels_xy(
    df: pd.DataFrame, task: dict, ch_map: dict
) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract (labels, xy) for a task from a pre-processed sample df."""
    mask = _parse_parent_mask(df, task["parent"])
    col = task["annotation_column"]
    if col in df.columns:
        labels = df.loc[mask, col].fillna("Unassigned").to_numpy(dtype=object)
        # Guard: object columns can still carry None that fillna missed in edge cases.
        na_mask = pd.isna(labels)
        if na_mask.any():
            labels[na_mask] = "Unassigned"
    else:
        labels = np.full(int(mask.sum()), "Unassigned", dtype=object)

    x_ch = _resolve_channel(task["x_marker"], list(df.columns), ch_map)
    y_ch = _resolve_channel(task["y_marker"], list(df.columns), ch_map)
    if x_ch is None or y_ch is None:
        return labels, None
    xy = df.loc[mask, [x_ch, y_ch]].to_numpy(dtype=np.float32)
    return labels, xy


def load_gt_labels_and_xy(
    data_dir: Path,
    task: dict,
    benchmark_dir: Path | None = None,
    perturbation: Perturbation | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load GT labels and x/y marker values for a task in a single pass.

    Thin wrapper around `_prepare_sample_df` + `_extract_labels_xy`. For batch
    evaluation, prefer calling those directly with a shared sample cache to avoid
    reloading the same parquet per task.
    """
    df, ch_map, _ = _prepare_sample_df(
        data_dir, task["dataset"], task["sample"], benchmark_dir, perturbation
    )
    return _extract_labels_xy(df, task, ch_map)


def _parse_parent_mask(df: pd.DataFrame, parent_str: str) -> np.ndarray:
    """Parse parent condition string into boolean mask.

    When the target column is a Categorical (as produced by `_prepare_sample_df`),
    comparisons are done on integer codes — a Step* column with N rows scans in
    O(N) int ops instead of object-dtype `==`. NaN rows carry code -1, which
    preserves the semantics of the prior object-dtype behavior:
      * `col == val`  — NaN never equals val ⇒ False (code -1 ≠ code(val))
      * `col != val`  — NaN is not equal to val ⇒ True
      * `col in [..]` — NaN is not in the list ⇒ False
    """
    n = len(df)
    if parent_str == "ALL":
        return np.ones(n, dtype=bool)

    def _codes_for(s: pd.Series):
        """Return (is_categorical, codes_array, category_index). Codes is None if not categorical."""
        if isinstance(s.dtype, pd.CategoricalDtype):
            return True, s.cat.codes.to_numpy(), s.cat.categories
        return False, None, None

    mask = np.ones(n, dtype=bool)
    for part in parent_str.split("&"):
        part = part.strip()
        if "==" in part:
            col, val = [x.strip() for x in part.split("==")]
            if col not in df.columns:
                mask[:] = False
                continue
            is_cat, codes, cats = _codes_for(df[col])
            if is_cat:
                if val in cats:
                    mask &= (codes == cats.get_loc(val))
                else:
                    mask[:] = False
            else:
                mask &= (df[col].values == val)
        elif "!=" in part:
            col, val = [x.strip() for x in part.split("!=")]
            if col not in df.columns:
                mask[:] = False
                continue
            is_cat, codes, cats = _codes_for(df[col])
            if is_cat:
                # val not in cats ⇒ every row "!= val" is True ⇒ no constraint.
                if val in cats:
                    mask &= (codes != cats.get_loc(val))
            else:
                mask &= (df[col].values != val)
        elif " in " in part:
            col, vals_str = [x.strip() for x in part.split(" in ")]
            vals = [v.strip() for v in vals_str.split(",")]
            if col not in df.columns:
                mask[:] = False
                continue
            is_cat, codes, cats = _codes_for(df[col])
            if is_cat:
                target_codes = [cats.get_loc(v) for v in vals if v in cats]
                if target_codes:
                    mask &= np.isin(codes, target_codes)
                else:
                    mask[:] = False
            else:
                mask &= np.isin(df[col].values, vals)

    return mask


def _trim_quantile(points: np.ndarray, q: tuple[float, float] | None) -> np.ndarray:
    """Drop points outside per-axis quantile bounds (reduces outlier sensitivity).

    Uses a single `np.quantile(axis, [lo, hi])` call per axis — numpy sorts
    once and returns both percentiles, vs 4 separate calls (2 sorts per axis).
    Output values are identical to the previous implementation.
    """
    if q is None or points is None or len(points) == 0:
        return points
    lo, hi = q
    x, y = points[:, 0], points[:, 1]
    xlo, xhi = np.quantile(x, [lo, hi])
    ylo, yhi = np.quantile(y, [lo, hi])
    keep = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    return points[keep]


def _convex_hull_polygon(
    points: np.ndarray, trim_quantile: tuple[float, float] | None = None
) -> np.ndarray | None:
    """Return ordered vertices of the 2D convex hull, or None if degenerate.

    If `trim_quantile` is given (e.g. (0.01, 0.99)), points outside per-axis
    quantile bounds are dropped before hull computation to reduce outlier sensitivity.
    """
    from scipy.spatial import ConvexHull
    if points is None or len(points) < 3:
        return None
    pts = _trim_quantile(points, trim_quantile)
    if len(pts) < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    return pts[hull.vertices]


def _polygon_area(poly: np.ndarray | None) -> float:
    """Shoelace area of a polygon; 0 for <3 vertices or None."""
    if poly is None or len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """Return polygon vertices in counter-clockwise order."""
    x, y = poly[:, 0], poly[:, 1]
    signed_area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return poly if signed_area > 0 else poly[::-1]


def _clip_convex(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: clip convex subject polygon by convex clipper polygon.

    Both inputs must have >=3 vertices. Returns an (M, 2) array of the clipped
    polygon's vertices (M may be 0 if the intersection is empty).
    """
    subject = _ensure_ccw(subject)
    clipper = _ensure_ccw(clipper)

    output = [tuple(p) for p in subject]
    n_c = len(clipper)

    for i in range(n_c):
        if not output:
            break
        a = clipper[i]
        b = clipper[(i + 1) % n_c]
        ex, ey = b[0] - a[0], b[1] - a[1]

        input_list = output
        output = []
        m = len(input_list)
        for j in range(m):
            p = input_list[j]
            q = input_list[(j + 1) % m]
            # CCW: inside the edge if cross(edge, point-a) >= 0
            p_in = ex * (p[1] - a[1]) - ey * (p[0] - a[0]) >= 0
            q_in = ex * (q[1] - a[1]) - ey * (q[0] - a[0]) >= 0
            if q_in:
                if not p_in:
                    # intersection of segment p→q with line a→b
                    dx, dy = q[0] - p[0], q[1] - p[1]
                    denom = ex * dy - ey * dx
                    if denom != 0:
                        t = (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / -denom
                        output.append((p[0] + t * dx, p[1] + t * dy))
                output.append(q)
            elif p_in:
                dx, dy = q[0] - p[0], q[1] - p[1]
                denom = ex * dy - ey * dx
                if denom != 0:
                    t = (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / -denom
                    output.append((p[0] + t * dx, p[1] + t * dy))

    return np.asarray(output, dtype=np.float64) if output else np.empty((0, 2))


def _convex_hull_iou(
    xy_gt: np.ndarray,
    xy_pred: np.ndarray,
    trim_quantile: tuple[float, float] | None = (0.01, 0.99),
) -> float | None:
    """IoU of convex hulls of two 2D point sets. Returns None if GT hull is degenerate.

    Within a task (fixed x_marker, y_marker) the IoU is a ratio of areas in the
    same coordinate system, so it is scale-invariant; cross-task averaging simply
    averages ratios. By default, 1~99% per-axis quantile trimming is applied
    to each point set before hull computation to reduce outlier sensitivity.
    """
    hull_gt = _convex_hull_polygon(xy_gt, trim_quantile=trim_quantile)
    if hull_gt is None:
        return None
    hull_pred = _convex_hull_polygon(xy_pred, trim_quantile=trim_quantile)
    area_gt = _polygon_area(hull_gt)
    if hull_pred is None:
        return 0.0
    area_pred = _polygon_area(hull_pred)
    inter = _polygon_area(_clip_convex(hull_gt, hull_pred))
    union = area_gt + area_pred - inter
    return inter / union if union > 0 else 0.0


def compute_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    categories: list[str],
    xy: np.ndarray | None = None,
) -> dict:
    """Compute evaluation metrics for a single task.

    Returns dict with:
        - accuracy
        - balanced_accuracy
        - f1_macro
        - f1_weighted
        - population_proportion_error: mean |pred_prop - gt_prop| across categories
        - iou_hull_macro: mean convex-hull IoU over categories with a valid GT hull (if xy given)
        - per_category: {cat: {precision, recall, f1, gt_prop, pred_prop, iou_hull?}}

    Implementation note: builds a single confusion matrix from integer codes
    and derives all label-based metrics from it (one pass over the label
    arrays), instead of calling sklearn 4× and running a per-category
    boolean-mask loop over object-dtype arrays.
    """
    n = len(gt_labels)

    # Union of labels that appear in gt/pred, with `categories` first (stable order).
    gt_unique = pd.unique(gt_labels)
    pred_unique = pd.unique(pred_labels)
    seen = set(categories)
    extras: list[str] = []
    for v in (*gt_unique, *pred_unique):
        if v not in seen:
            seen.add(v)
            extras.append(v)
    all_labels = list(categories) + extras
    n_labels = len(all_labels)
    code = {lab: i for i, lab in enumerate(all_labels)}

    # Vectorized label → int code.
    gt_codes = pd.Categorical(gt_labels, categories=all_labels).codes.astype(np.int64)
    pred_codes = pd.Categorical(pred_labels, categories=all_labels).codes.astype(np.int64)

    # Single-pass confusion matrix via flat bincount (faster than np.add.at).
    flat = gt_codes * n_labels + pred_codes
    cm = np.bincount(flat, minlength=n_labels * n_labels).reshape(n_labels, n_labels)

    tp = np.diag(cm).astype(np.float64)
    gt_sum = cm.sum(axis=1).astype(np.float64)    # support per true class
    pred_sum = cm.sum(axis=0).astype(np.float64)  # support per predicted class

    with np.errstate(divide="ignore", invalid="ignore"):
        precision_all = np.where(pred_sum > 0, tp / pred_sum, 0.0)
        recall_all = np.where(gt_sum > 0, tp / gt_sum, 0.0)
        denom = precision_all + recall_all
        f1_all = np.where(denom > 0, 2 * precision_all * recall_all / denom, 0.0)

    # Overall accuracy: matches sklearn.accuracy_score(gt, pred).
    acc = float(tp.sum() / n) if n > 0 else 0.0

    # Balanced accuracy: mean recall over classes with GT support > 0 (matches sklearn).
    has_gt = gt_sum > 0
    bal_acc = float(recall_all[has_gt].mean()) if has_gt.any() else 0.0

    # F1 macro / weighted over the listed categories only.
    cat_idx = np.fromiter((code[c] for c in categories), dtype=np.int64, count=len(categories))
    f1_cats = f1_all[cat_idx]
    gt_sum_cats = gt_sum[cat_idx]
    f1_mac = float(f1_cats.mean()) if len(f1_cats) else 0.0
    f1_wt = (
        float(np.dot(f1_cats, gt_sum_cats) / gt_sum_cats.sum())
        if gt_sum_cats.sum() > 0
        else 0.0
    )

    # Proportions over the full population (matches old numerator/denominator).
    gt_props_cats = gt_sum_cats / n if n > 0 else np.zeros_like(gt_sum_cats)
    pred_props_cats = pred_sum[cat_idx] / n if n > 0 else np.zeros_like(gt_sum_cats)
    prop_err = float(np.abs(gt_props_cats - pred_props_cats).mean())

    per_category: dict = {}
    hull_ious: list = []
    # Hull IoU still needs per-category masks — only build them for non-Unassigned
    # categories and only when xy is provided.
    for i, cat in enumerate(categories):
        ci = cat_idx[i]
        entry = {
            "precision": round(float(precision_all[ci]), 4),
            "recall": round(float(recall_all[ci]), 4),
            "f1": round(float(f1_all[ci]), 4),
            "gt_proportion": round(float(gt_props_cats[i]), 4),
            "pred_proportion": round(float(pred_props_cats[i]), 4),
        }
        if xy is not None and cat != "Unassigned":
            gt_mask = gt_codes == ci
            pred_mask = pred_codes == ci
            iou = _convex_hull_iou(xy[gt_mask], xy[pred_mask])
            if iou is not None:
                entry["iou_hull"] = round(iou, 4)
                hull_ious.append(iou)
        per_category[cat] = entry

    out = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "population_proportion_error": round(prop_err, 4),
        "per_category": per_category,
    }
    if xy is not None:
        out["iou_hull_macro"] = round(float(np.mean(hull_ious)), 4) if hull_ious else None
        out["iou_hull_n"] = len(hull_ious)
    return out


def evaluate_predictions(
    predictions_dir: Path,
    benchmark_dir: Path,
    data_dir: Path,
    dataset: str,
    perturbation: Perturbation | None = None,
    perturbation_name: str | None = None,
    sample_filter: set[str] | None = None,
) -> dict:
    """Evaluate all prediction.json files for a single dataset.

    Scans only ``predictions_dir/{dataset}/`` so per-sample parquet caches
    live for one dataset at a time — the caller loops across datasets and
    the cache is released between iterations.

    If ``sample_filter`` is given, only prediction files whose sample name
    (parent directory of ``step_NN/``) is in the set are scored.
    """
    ds_root = predictions_dir / dataset
    pred_files = sorted(ds_root.rglob("prediction.json")) if ds_root.exists() else []
    if sample_filter is not None:
        pred_files = [p for p in pred_files if p.parent.parent.name in sample_filter]
    if not pred_files:
        print(f"No prediction.json files found in {ds_root}")
        return {}

    all_results = []
    step_results = defaultdict(list)

    # Cache the preprocessed df per (dataset, sample) so we read each parquet once.
    sample_cache: dict[tuple[str, str], tuple[pd.DataFrame, dict]] = {}
    shared_loader: BenchmarkLoader | None = None

    for pred_path in pred_files:
        with open(pred_path) as f:
            pred = json.load(f)

        task_id = pred["task_id"]

        # Parse from directory structure: results/{method}/{dataset}/{sample}/step_{NN}/
        # More robust than splitting task_id (sample names may contain "__")
        step_str = pred_path.parent.name  # step_NN
        sample = pred_path.parent.parent.name
        step_num = int(step_str.replace("step_", ""))

        # Load task.json
        task_path = benchmark_dir / dataset / sample / step_str / "task.json"
        if not task_path.exists():
            print(f"  [SKIP] task.json not found: {task_path}")
            continue

        with open(task_path) as f:
            task = json.load(f)

        # Sanity-check perturbation consistency between eval flag and prediction file.
        pred_perturb = pred.get("perturbation")
        if perturbation_name is not None and pred_perturb not in (None, perturbation_name):
            print(
                f"  [SKIP] {task_id}: perturbation mismatch "
                f"(prediction={pred_perturb!r}, eval={perturbation_name!r})"
            )
            continue
        if perturbation_name is None and pred_perturb is not None:
            print(
                f"  [SKIP] {task_id}: prediction has perturbation={pred_perturb!r} "
                "but eval was called without --perturbation — scoring against "
                "unperturbed GT would be meaningless. Re-run eval with the "
                "same --perturbation spec to score this file."
            )
            continue

        # Load GT + x/y marker values (with perturbation if given), via sample cache
        cache_key = (dataset, sample)
        if cache_key not in sample_cache:
            df_prepared, ch_map, shared_loader = _prepare_sample_df(
                data_dir, dataset, sample,
                benchmark_dir=benchmark_dir, perturbation=perturbation,
                loader=shared_loader,
            )
            sample_cache[cache_key] = (df_prepared, ch_map)
        df_prepared, ch_map = sample_cache[cache_key]
        gt_labels, gt_xy = _extract_labels_xy(df_prepared, task, ch_map)
        pred_labels = np.array(pred["predicted_labels"], dtype=object)

        if len(gt_labels) != len(pred_labels):
            print(f"  [SKIP] {task_id}: label count mismatch (gt={len(gt_labels)}, pred={len(pred_labels)})")
            continue

        n_total = len(gt_labels)
        if n_total == 0:
            print(f"  [SKIP] {task_id}: empty parent gate")
            continue

        # Treat Unassigned as a regular category: include all cells in the eval.
        gt_eval = gt_labels
        pred_eval = pred_labels.copy()
        # None (HDBSCAN noise) → "__noise__" so it always counts as wrong
        pred_eval[pred_eval == None] = "__noise__"  # noqa: E711
        xy_eval = gt_xy

        categories = list(task["categories"])  # includes "Unassigned" if present
        metrics = compute_metrics(gt_eval, pred_eval, categories, xy=xy_eval)
        metrics["task_id"] = task_id
        metrics["dataset"] = dataset
        metrics["sample"] = sample
        metrics["step"] = step_num
        metrics["n_cells"] = n_total
        metrics["n_cells_total"] = n_total
        metrics["unassigned_ratio"] = round(
            float(np.sum(gt_labels == "Unassigned") / n_total), 4
        )

        all_results.append(metrics)
        step_results[step_num].append(metrics)

    summary = {
        "method": pred.get("method", "unknown") if pred_files else "unknown",
        "dataset": dataset,
        "n_tasks": len(all_results),
        "overall": _aggregate_metrics(all_results),
        "by_step": {step: _aggregate_metrics(res) for step, res in sorted(step_results.items())},
        "per_task": all_results,
    }

    return summary


def _aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate metrics across multiple tasks."""
    if not results:
        return {}

    keys = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "population_proportion_error"]
    agg = {}
    for k in keys:
        values = [r[k] for r in results]
        agg[f"{k}_mean"] = round(np.mean(values), 4)
        agg[f"{k}_std"] = round(np.std(values), 4)

    iou_vals = [r["iou_hull_macro"] for r in results if r.get("iou_hull_macro") is not None]
    if iou_vals:
        agg["iou_hull_macro_mean"] = round(float(np.mean(iou_vals)), 4)
        agg["iou_hull_macro_std"] = round(float(np.std(iou_vals)), 4)
        agg["iou_hull_macro_n"] = len(iou_vals)

    agg["n_tasks"] = len(results)
    agg["n_cells_evaluated"] = sum(r["n_cells"] for r in results)
    agg["n_cells_total"] = sum(r.get("n_cells_total", r["n_cells"]) for r in results)

    return agg



def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth")
    parser.add_argument("--predictions", type=Path, required=True, help="Directory with prediction.json files")
    parser.add_argument(
        "--output", type=Path, default=None,
        help=argparse.SUPPRESS,  # deprecated: per-dataset JSONs now always go to "
                                  # f"{predictions}/{dataset}/eval_summary.json
    )
    parser.add_argument("--plot", action="store_true", help="Generate GT vs prediction comparison plots (opt-in, ~1 min/sample)")
    parser.add_argument("--max-cells", type=int, default=5000, help="Max cells per plot panel (default: 5000)")
    parser.add_argument(
        "--plot-per-cohort", type=int, default=0,
        help="If >0, generate plots for at most N samples per cohort (sorted by sample name). "
             "Default 0 = plot every sample.",
    )
    BenchmarkLoader.add_cli_args(parser)
    args = parser.parse_args()

    loader_for_perturb = BenchmarkLoader.from_cli(args)
    perturbation = loader_for_perturb.perturbation
    perturbation_name = loader_for_perturb.perturbation_name

    if args.output is not None:
        print(
            "[warn] --output is deprecated and ignored. "
            f"Per-dataset summaries are written to {args.predictions}/{{dataset}}/eval_summary.json"
        )

    if args.datasets:
        datasets = list(args.datasets)
    else:
        datasets = sorted(
            p.name for p in args.predictions.iterdir()
            if p.is_dir() and p.name != "plots" and any(p.rglob("prediction.json"))
        )

    if not datasets:
        print(f"No datasets found under {args.predictions}")
        return

    print(f"Evaluating predictions from: {args.predictions}")
    print(f"Datasets: {datasets}")

    split_filter_by_ds: dict[str, set[str]] = {}
    if getattr(args, "split", None):
        try:
            with open(args.split) as f:
                splits_data = json.load(f)
        except FileNotFoundError:
            splits_data = {}
        split_set = getattr(args, "split_set", "test") or "test"
        for split_group in splits_data.values():
            if not isinstance(split_group, dict):
                continue
            for ds, info in split_group.items():
                if isinstance(info, dict) and split_set in info:
                    split_filter_by_ds[ds] = set(info[split_set])

    for dataset in datasets:
        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}")
        sample_filter = split_filter_by_ds.get(dataset) if split_filter_by_ds else None
        if sample_filter is not None:
            print(f"  [split filter] {len(sample_filter)} samples in '{getattr(args, 'split_set', 'test')}'")
        summary = evaluate_predictions(
            args.predictions,
            args.benchmark,
            args.data_dir,
            dataset=dataset,
            perturbation=perturbation,
            perturbation_name=perturbation_name,
            sample_filter=sample_filter,
        )
        if not summary:
            continue

        overall = summary["overall"]
        if overall:
            print(f"Method: {summary['method']} | Tasks: {summary['n_tasks']}")
            print(f"  F1 (macro):    {overall['f1_macro_mean']:.4f} ± {overall['f1_macro_std']:.4f}")
            print(f"  F1 (weighted): {overall['f1_weighted_mean']:.4f} ± {overall['f1_weighted_std']:.4f}")
            print(f"  Balanced Acc:  {overall['balanced_accuracy_mean']:.4f} ± {overall['balanced_accuracy_std']:.4f}")
            print(f"  Pop Prop Err:  {overall['population_proportion_error_mean']:.4f} ± {overall['population_proportion_error_std']:.4f}")
            if "iou_hull_macro_mean" in overall:
                print(f"  Hull IoU:      {overall['iou_hull_macro_mean']:.4f} ± {overall['iou_hull_macro_std']:.4f} (n={overall['iou_hull_macro_n']})")

        output_path = args.predictions / dataset / "eval_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")

        if args.plot:
            plot_dir = args.predictions / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            generate_comparison_plots(
                summary, args.predictions, args.benchmark, args.data_dir, plot_dir,
                max_cells=args.max_cells, perturbation=perturbation,
                per_cohort_limit=args.plot_per_cohort,
                perturbation_name=perturbation_name,
            )

        # Drop per-task results before moving to next dataset so the prior
        # summary's arrays can be GC'd alongside the per-sample parquet
        # cache that lived inside evaluate_predictions.
        del summary


def generate_comparison_plots(
    summary: dict,
    predictions_dir: Path,
    benchmark_dir: Path,
    data_dir: Path,
    plot_dir: Path,
    max_cells: int = 5000,
    perturbation: Perturbation | None = None,
    per_cohort_limit: int = 0,
    perturbation_name: str | None = None,
):
    """Generate GT vs prediction comparison plots for all tasks (parquet-based)."""
    import matplotlib.pyplot as plt

    sample_tasks = defaultdict(list)
    for t in summary["per_task"]:
        sample_tasks[(t["dataset"], t["sample"])].append(t)

    if per_cohort_limit > 0:
        per_cohort = defaultdict(list)
        for (dataset, sample) in sample_tasks:
            per_cohort[dataset].append(sample)
        keep = set()
        for dataset, samples in per_cohort.items():
            for s in sorted(samples)[:per_cohort_limit]:
                keep.add((dataset, s))
        sample_tasks = {k: v for k, v in sample_tasks.items() if k in keep}

    for (dataset, sample), tasks in sample_tasks.items():
        tasks = sorted(tasks, key=lambda x: x["step"])

        parquet_path = data_dir / dataset / "parquet" / f"{sample}.parquet"
        if not parquet_path.exists():
            print(f"  [SKIP] parquet not found: {parquet_path}")
            continue

        df, ch_map, _ = _prepare_sample_df(
            data_dir, dataset, sample,
            benchmark_dir=benchmark_dir, perturbation=perturbation,
        )

        n_steps = len(tasks)
        fig, axes = plt.subplots(n_steps, 4, figsize=(24, 5 * n_steps))
        if n_steps == 1:
            axes = axes[np.newaxis, :]

        for row, task_info in enumerate(tasks):
            step_num = task_info["step"]
            step_str = f"step_{step_num:02d}"

            task_path = benchmark_dir / dataset / sample / step_str / "task.json"
            with open(task_path) as f:
                task = json.load(f)

            pred_path = predictions_dir / dataset / sample / step_str / "prediction.json"
            with open(pred_path) as f:
                pred = json.load(f)

            x_ch = _resolve_channel(task["x_marker"], list(df.columns), ch_map)
            y_ch = _resolve_channel(task["y_marker"], list(df.columns), ch_map)
            if x_ch is None or y_ch is None:
                continue

            mask = _parse_parent_mask(df, task["parent"])
            x_vals = df.loc[mask, x_ch].values.astype(np.float32)
            y_vals = df.loc[mask, y_ch].values.astype(np.float32)

            col = task["annotation_column"]
            gt = df.loc[mask, col].fillna("Unassigned").to_numpy(dtype=object)
            na_mask = pd.isna(gt)
            if na_mask.any():
                gt[na_mask] = "Unassigned"

            pred_labels = np.array(pred["predicted_labels"], dtype=object)

            # Keep full-resolution arrays for hull computation (match eval IoU)
            x_full, y_full, gt_full, pred_full = x_vals, y_vals, gt, pred_labels

            # Subsample for fast scatter plotting
            n_total = len(x_vals)
            if n_total > max_cells:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_total, max_cells, replace=False)
                x_vals, y_vals = x_vals[idx], y_vals[idx]
                gt, pred_labels = gt[idx], pred_labels[idx]

            categories = task["categories"]
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(categories), 10)))

            ax = axes[row, 0]
            _plot_density(ax, x_vals, y_vals)
            ax.set_xlabel(task["x_marker"], fontsize=9)
            ax.set_ylabel(task["y_marker"], fontsize=9)
            ax.set_title(f"Step {step_num}: Input ({n_total:,} cells, showing {len(x_vals):,})", fontsize=10)

            ax = axes[row, 1]
            _plot_categories(ax, x_vals, y_vals, gt, categories, colors)
            ax.set_xlabel(task["x_marker"], fontsize=9)
            ax.set_ylabel(task["y_marker"], fontsize=9)
            ax.set_title(f"Step {step_num}: Ground Truth", fontsize=10)

            ax = axes[row, 2]
            _plot_categories(ax, x_vals, y_vals, pred_labels, categories, colors, show_noise=True)
            ax.set_xlabel(task["x_marker"], fontsize=9)
            ax.set_ylabel(task["y_marker"], fontsize=9)
            f1 = task_info["f1_macro"]
            ax.set_title(f"Step {step_num}: {summary['method']} (F1={f1:.3f})", fontsize=10)

            ax = axes[row, 3]
            _plot_hulls(
                ax, x_vals, y_vals, gt, x_full, y_full, gt_full, pred_full,
                [c for c in categories if c != "Unassigned"], colors,
            )
            ax.set_xlabel(task["x_marker"], fontsize=9)
            ax.set_ylabel(task["y_marker"], fontsize=9)
            iou_macro = task_info.get("iou_hull_macro")
            iou_str = f"{iou_macro:.3f}" if iou_macro is not None else "n/a"
            ax.set_title(f"Step {step_num}: Hull GT vs Pred (IoU={iou_str})", fontsize=10)

        plt.tight_layout()
        safe_sample = sample.replace(" ", "_")[:80]
        if perturbation_name:
            fname = f"{dataset}__{perturbation_name}__{safe_sample}.png"
        else:
            fname = f"{dataset}__{safe_sample}.png"
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot: {dataset}/{sample}")


def _resolve_channel(marker: str, columns: list[str], ch_map: dict) -> str | None:
    if marker in columns:
        return marker
    for ch, m in ch_map.items():
        if m == marker and ch in columns:
            return ch
    return None


def _plot_density(ax, x, y):
    hist, xedges, yedges = np.histogram2d(x, y, bins=200)
    xi = np.clip(np.digitize(x, xedges) - 1, 0, 199)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, 199)
    density = hist[xi, yi]
    order = np.argsort(density)
    ax.scatter(x[order], y[order], s=6, alpha=0.4, c=density[order], cmap="viridis", rasterized=True)


def _plot_hulls(
    ax, x_sub, y_sub, gt_sub, x_full, y_full, gt_full, pred_full, categories, colors,
):
    """Scatter (subsampled GT points) with per-category GT (solid) and Pred (dashed)
    convex hulls overlaid. Hulls are computed on the full (pre-subsample) arrays so
    they match the eval-time IoU."""
    from matplotlib.patches import Polygon as MplPolygon

    # Background: Unassigned cells in light gray (context for the parent gate region)
    unass_mask = gt_sub == "Unassigned"
    n_unass = int(unass_mask.sum())
    if n_unass > 0:
        ax.scatter(x_sub[unass_mask], y_sub[unass_mask], s=6, alpha=0.15, c="#bbbbbb",
                   rasterized=True, label=f"Unassigned ({n_unass:,})")

    # Foreground: GT-colored scatter per category (subsampled)
    for i, cat in enumerate(categories):
        m = gt_sub == cat
        n = int(np.sum(m))
        if n > 0:
            ax.scatter(x_sub[m], y_sub[m], s=6, alpha=0.25, c=[colors[i % len(colors)]],
                       rasterized=True)

    # Per-category hulls from full arrays
    for i, cat in enumerate(categories):
        color = colors[i % len(colors)]
        gt_mask = gt_full == cat
        pred_mask = pred_full == cat

        gt_xy = np.column_stack([x_full[gt_mask], y_full[gt_mask]])
        pred_xy = np.column_stack([x_full[pred_mask], y_full[pred_mask]])

        gt_hull = _convex_hull_polygon(gt_xy, trim_quantile=(0.01, 0.99))
        pred_hull = _convex_hull_polygon(pred_xy, trim_quantile=(0.01, 0.99))

        n_gt = int(gt_mask.sum())
        n_pred = int(pred_mask.sum())
        label = f"{cat} (GT={n_gt:,}, Pred={n_pred:,})"

        if gt_hull is not None:
            ax.add_patch(MplPolygon(
                gt_hull, closed=True, facecolor=color, edgecolor=color,
                alpha=0.15, linewidth=1.8, linestyle="-", label=label,
            ))
            # Outline stroke on top without fill for clarity
            ax.add_patch(MplPolygon(
                gt_hull, closed=True, fill=False, edgecolor=color,
                linewidth=1.8, linestyle="-",
            ))
        if pred_hull is not None:
            ax.add_patch(MplPolygon(
                pred_hull, closed=True, facecolor=color, edgecolor=color,
                alpha=0.15, linewidth=1.5, linestyle="--",
                label=label if gt_hull is None else None,
            ))
            ax.add_patch(MplPolygon(
                pred_hull, closed=True, fill=False, edgecolor=color,
                linewidth=1.5, linestyle="--",
            ))

    ax.legend(fontsize=6, markerscale=2, loc="best")


def _plot_categories(ax, x, y, labels, categories, colors, show_noise=False):
    for i, cat in enumerate(categories):
        m = labels == cat
        n = int(np.sum(m))
        if n > 0:
            ax.scatter(x[m], y[m], s=6, alpha=0.4, c=[colors[i % len(colors)]],
                       label=f"{cat} ({n:,})", rasterized=True)
    if show_noise:
        noise_mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in labels])
        n_noise = int(np.sum(noise_mask))
        if n_noise > 0:
            ax.scatter(x[noise_mask], y[noise_mask], s=6, alpha=0.3, c=["#aaaaaa"],
                       label=f"Noise ({n_noise:,})", rasterized=True)
    ax.legend(fontsize=6, markerscale=3, loc="best")


if __name__ == "__main__":
    main()
