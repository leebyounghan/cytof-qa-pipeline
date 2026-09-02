#!/usr/bin/env python3
"""Flat-annotation evaluation — shared by ``src/llm_gate_flat/`` and ``src/flowsom_c2s/``.

Both pipelines emit ``{output_dir}/{dataset}/{sample}/predicted_steps.parquet``
with the same Step* columns as the GT parquet. This module reduces each
cell to its multi-label leaf set + primary leaf and reports identical
metrics for both methods, enabling apples-to-apples comparison.

Definition of a *leaf*: a category that NEVER appears in any later step's
parent expression in ``gating_plan.json``. Per cell, the *flat label set*
is the set of leaves encountered along its path (Step columns walked in
plan order). Cells that never hit a leaf (rejected somewhere in the
cleanup chain or cascade aborted) get the synthetic class ``Discard``
instead of an empty set.

Why ``Discard`` is a real class: a cell rejected at step 1 (cascade) and
a cell rejected at step 5 (GT) are functionally equivalent — both ended
up *not* being a real cell type. Only the binary outcome matters
(reached a leaf vs not), not where in the cleanup chain the rejection
happened. Treating ``Discard`` as a class means cleanup-agreement
contributes positive TPs to F1 instead of being neutral zero-vectors.

In plans with orthogonal axes (L2/L3 helper subtype × memory subtype) a
single cell can legitimately carry multiple leaf labels — multi-label
framework handles both cases uniformly.

Metrics:
  - Multi-label (over leaves ∪ {Discard}): subset accuracy, Hamming
    loss, Jaccard (micro/macro), F1 (micro/macro/weighted), per-leaf
    precision/recall/F1.
  - Primary-leaf view: deepest leaf in path, or ``Discard`` when none.
    Standard accuracy, balanced accuracy, F1 macro/weighted, confusion
    matrix.

Usage::

    uv run python -m src.flat_eval \\
        --output-dir  results_flat/<method>/ \\
        --gating-plan data/<dataset>/gating_plan.json \\
        --data-dir    data/ \\
        --benchmark   benchmark/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
)
from sklearn.preprocessing import MultiLabelBinarizer  # noqa: E402

from src.bench import BenchmarkLoader  # noqa: E402

DISCARD_LABEL = "Discard"


def gating_plan_slug(path) -> str:
    """Derive a stable short tag from a gating-plan filename.

    Used by flat pipelines (``src/flowsom_c2s/`` and ``src/llm_gate_flat/``)
    to keep outputs from different plan depths (L1 / L2 / L3) cleanly
    separated under matching ``<plan_slug>`` subdirectories.

    Examples:
        ``gating_plan.json``                -> ``"default"`` (canonical/L3)
        ``gating_plan_L1_coarse.json``      -> ``"L1_coarse"``
        ``gating_plan_L2_identity.json``    -> ``"L2_identity"``
        ``something_else.json``             -> ``"something_else"``
    """
    from pathlib import Path
    stem = Path(str(path)).stem
    if stem == "gating_plan":
        return "default"
    if stem.startswith("gating_plan_"):
        return stem[len("gating_plan_"):]
    return stem


def derive_leaf_set(gating_plan: dict) -> set[str]:
    """A category is a leaf iff it never appears in any later step's parent."""
    parent_used: set[str] = set()
    for step in gating_plan.get("steps", []):
        parent_expr = step.get("parent", "ALL") or "ALL"
        if parent_expr == "ALL":
            continue
        for part in parent_expr.split("&"):
            part = part.strip()
            for op in ("==", "!=", " in "):
                if op in part:
                    _, val_str = part.split(op, 1)
                    val_str = val_str.strip()
                    if op == " in ":
                        for v in val_str.split(","):
                            parent_used.add(v.strip())
                    else:
                        parent_used.add(val_str)
                    break
    leaves: set[str] = set()
    for step in gating_plan.get("steps", []):
        for cat in step.get("annotation categories", []) or []:
            if cat not in parent_used:
                leaves.add(cat)
    return leaves


def _flat_labels_multilabel(
    df: pd.DataFrame,
    step_cols: list[str],
    leaf_set: set[str],
) -> tuple[list[set[str]], np.ndarray]:
    """Returns (multi_label_sets, primary_label_array).

    Cells that hit at least one leaf get the set of leaves they hit, and
    primary = the deepest leaf in path. Cells that never hit a leaf get
    ``{DISCARD_LABEL}`` and primary = ``DISCARD_LABEL``.
    """
    n = len(df)
    multi: list[set[str]] = [set() for _ in range(n)]
    primary = np.array([DISCARD_LABEL] * n, dtype=object)
    for col in step_cols:
        if col not in df.columns:
            continue
        vals = df[col].astype(object).values
        for i, v in enumerate(vals):
            if isinstance(v, str) and v in leaf_set:
                multi[i].add(v)
                primary[i] = v
    for i, s in enumerate(multi):
        if not s:
            s.add(DISCARD_LABEL)
    return multi, primary


def _plot_confusion(
    cm: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, where=row_sum > 0)
    cm_norm = np.nan_to_num(cm_norm)

    side = max(6.0, 0.45 * len(labels) + 4.0)
    fig, ax = plt.subplots(figsize=(side, side), dpi=110)
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted (primary leaf)")
    ax.set_ylabel("GT (primary leaf)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            if cm[i, j] > 0:
                ax.text(
                    j, i, f"{int(cm[i, j])}",
                    ha="center", va="center", fontsize=7,
                    color="white" if v < 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_per_leaf_bars(
    leaves: list[str],
    metrics_per_leaf: dict[str, dict],
    gt_support: dict[str, int],
    out_path: Path,
    title: str,
) -> None:
    rows = [(l, metrics_per_leaf.get(l, {})) for l in leaves]
    precision = [r[1].get("precision", 0.0) for r in rows]
    recall = [r[1].get("recall", 0.0) for r in rows]
    f1 = [r[1].get("f1-score", 0.0) for r in rows]
    support = [gt_support.get(l, 0) for l in leaves]

    n = len(leaves)
    width = max(8.0, 0.4 * n + 3.0)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(width, 7.0), dpi=110, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    x = np.arange(n)
    bw = 0.27
    ax_top.bar(x - bw, precision, bw, label="precision", color="#4c72b0")
    ax_top.bar(x, recall, bw, label="recall", color="#dd8452")
    ax_top.bar(x + bw, f1, bw, label="F1", color="#55a467")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_ylabel("Score")
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.set_title(title, fontsize=10)
    ax_top.grid(axis="y", linestyle=":", alpha=0.5)

    ax_bot.bar(x, support, color="#888")
    ax_bot.set_ylabel("GT support")
    ax_bot.set_yscale("symlog")
    ax_bot.grid(axis="y", linestyle=":", alpha=0.5)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(leaves, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _render_cascade_plots(
    sample_out: Path,
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gating_plan: dict,
    dataset: str,
    sample: str,
) -> int:
    """Re-render per-step + cascade-summary biaxial plots for cascade methods.

    Detects a cascade-style output by the presence of ``step_NN/prediction.json``
    files alongside ``predicted_steps.parquet``. For non-cascade outputs (e.g.
    flowsom_c2s) returns 0 silently — those don't have step gates to plot.
    """
    step_dirs = sorted(
        d for d in sample_out.iterdir()
        if d.is_dir() and d.name.startswith("step_")
        and (d / "prediction.json").exists()
    )
    if not step_dirs:
        return 0

    try:
        from src.llm_gate_flat.plot import plot_sample_summary, plot_step
    except ImportError:
        return 0

    steps_by_num = {int(s["step"]): s for s in gating_plan.get("steps", [])}
    plot_records: list[dict] = []

    for step_dir in step_dirs:
        try:
            step_num = int(step_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        step = steps_by_num.get(step_num)
        if step is None:
            continue
        with open(step_dir / "prediction.json") as f:
            pred = json.load(f)
        gates = pred.get("gates", {}) or {}
        predicted_labels = pred.get("predicted_labels", []) or []
        info = pred.get("info", {}) or {}
        x_ch = info.get("x_channel")
        y_ch = info.get("y_channel")
        if x_ch is None or y_ch is None or x_ch not in gt_df.columns or y_ch not in gt_df.columns:
            continue

        parent_expr = step.get("parent", "ALL") or "ALL"
        parent_mask = BenchmarkLoader._parent_mask(pred_df, parent_expr)  # noqa: SLF001
        n_parent = int(parent_mask.sum())
        if n_parent == 0 or n_parent != len(predicted_labels):
            # parent_mask reconstructed from predicted_steps.parquet must agree
            # with the saved predicted_labels length; if not, skip this step
            # rather than mis-align points and gates.
            continue
        parent_idx = np.where(parent_mask)[0]
        x_arr = gt_df[x_ch].values[parent_idx]
        y_arr = gt_df[y_ch].values[parent_idx]
        options = list(step.get("annotation categories", []) or [])

        plot_step(
            step_dir / "plot.png",
            step_num=step_num,
            parent_expr=parent_expr,
            x_marker_display=step.get("x_marker", x_ch),
            y_marker_display=step.get("y_marker", y_ch),
            x_arr=x_arr, y_arr=y_arr,
            predicted_labels=predicted_labels,
            gates=gates, options=options,
        )
        plot_records.append({
            "step_num": step_num,
            "x_marker_display": step.get("x_marker", x_ch),
            "y_marker_display": step.get("y_marker", y_ch),
            "x_arr": x_arr, "y_arr": y_arr,
            "predicted_labels": predicted_labels,
            "gates": gates, "options": options,
        })

    if plot_records:
        plot_sample_summary(
            sample_out / "cascade_summary.png",
            plot_records, sample=sample, dataset=dataset,
        )
    return len(plot_records)


def evaluate_sample(
    loader: BenchmarkLoader,
    dataset: str,
    sample: str,
    output_dir: Path,
    gating_plan: dict,
) -> dict | None:
    sample_out = output_dir / dataset / sample
    pred_path = sample_out / "predicted_steps.parquet"
    if not pred_path.exists():
        print(f"  [skip] no predicted_steps.parquet for {dataset}/{sample}")
        return None
    pred_df = pd.read_parquet(pred_path)
    gt_df, _ = loader._load_sample(dataset, sample)  # noqa: SLF001

    step_cols = [s["annotation column name"] for s in gating_plan["steps"]]
    pred_cols = [c for c in step_cols if c in pred_df.columns]
    if not pred_cols:
        print(f"  [skip] no overlapping step columns for {dataset}/{sample}")
        return None

    leaf_set = derive_leaf_set(gating_plan)
    classes_sorted = sorted(leaf_set | {DISCARD_LABEL})

    gt_multi, gt_primary = _flat_labels_multilabel(gt_df, pred_cols, leaf_set)
    pred_multi, pred_primary = _flat_labels_multilabel(pred_df, pred_cols, leaf_set)

    n = min(len(gt_multi), len(pred_multi))
    gt_multi = gt_multi[:n]
    pred_multi = pred_multi[:n]
    gt_primary = gt_primary[:n]
    pred_primary = pred_primary[:n]

    # Multi-label encoding (leaves ∪ {Discard})
    mlb = MultiLabelBinarizer(classes=classes_sorted)
    gt_bin = mlb.fit_transform(gt_multi)
    pred_bin = mlb.transform(pred_multi)

    ml_subset_acc = float(accuracy_score(gt_bin, pred_bin))
    ml_hamming = float(hamming_loss(gt_bin, pred_bin))
    ml_f1_micro = float(f1_score(gt_bin, pred_bin, average="micro", zero_division=0))
    ml_f1_macro = float(f1_score(gt_bin, pred_bin, average="macro", zero_division=0))
    ml_f1_weighted = float(f1_score(gt_bin, pred_bin, average="weighted", zero_division=0))
    ml_jaccard_micro = float(jaccard_score(gt_bin, pred_bin, average="micro", zero_division=0))
    ml_jaccard_macro = float(jaccard_score(gt_bin, pred_bin, average="macro", zero_division=0))

    ml_report = classification_report(
        gt_bin, pred_bin, target_names=classes_sorted,
        output_dict=True, zero_division=0,
    )
    per_class = {l: ml_report[l] for l in classes_sorted if l in ml_report}

    n_gt_discard = int(sum(1 for s in gt_multi if s == {DISCARD_LABEL}))
    n_pred_discard = int(sum(1 for s in pred_multi if s == {DISCARD_LABEL}))
    n_gt_multi = int(sum(1 for s in gt_multi if len(s) > 1))
    n_pred_multi = int(sum(1 for s in pred_multi if len(s) > 1))
    gt_support = {l: int(gt_bin[:, i].sum()) for i, l in enumerate(classes_sorted)}
    pred_support = {l: int(pred_bin[:, i].sum()) for i, l in enumerate(classes_sorted)}

    # Primary-leaf single-label view (for viz + intuition)
    sl_labels = sorted(set(gt_primary) | set(pred_primary))
    sl_acc = float(accuracy_score(gt_primary, pred_primary))
    sl_bal_acc = float(balanced_accuracy_score(gt_primary, pred_primary))
    sl_f1_macro = float(f1_score(gt_primary, pred_primary, average="macro", zero_division=0))
    sl_f1_weighted = float(f1_score(gt_primary, pred_primary, average="weighted", zero_division=0))
    sl_cm = confusion_matrix(gt_primary, pred_primary, labels=sl_labels)

    metrics = {
        "dataset": dataset,
        "sample": sample,
        "n_cells": int(n),
        "n_steps_walked": len(pred_cols),
        "classes": classes_sorted,
        "leaf_set": sorted(leaf_set),
        "n_cells_gt_discard": n_gt_discard,
        "n_cells_pred_discard": n_pred_discard,
        "n_cells_multi_gt": n_gt_multi,
        "n_cells_multi_pred": n_pred_multi,
        "multi_label": {
            "subset_accuracy": ml_subset_acc,
            "hamming_loss": ml_hamming,
            "f1_micro": ml_f1_micro,
            "f1_macro": ml_f1_macro,
            "f1_weighted": ml_f1_weighted,
            "jaccard_micro": ml_jaccard_micro,
            "jaccard_macro": ml_jaccard_macro,
            "per_class": per_class,
            "gt_support": gt_support,
            "pred_support": pred_support,
        },
        "primary_leaf": {
            "labels": sl_labels,
            "accuracy": sl_acc,
            "balanced_accuracy": sl_bal_acc,
            "f1_macro": sl_f1_macro,
            "f1_weighted": sl_f1_weighted,
            "confusion_matrix": sl_cm.tolist(),
            "gt_counts": {l: int((gt_primary == l).sum()) for l in sl_labels},
            "pred_counts": {l: int((pred_primary == l).sum()) for l in sl_labels},
        },
    }
    with open(sample_out / "eval_flat.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_confusion(
        sl_cm, sl_labels,
        sample_out / "confusion_matrix.png",
        title=(
            f"{dataset} / {sample} — primary-leaf confusion matrix\n"
            f"acc={sl_acc:.3f}  bal_acc={sl_bal_acc:.3f}  "
            f"F1_macro={sl_f1_macro:.3f}  F1_weighted={sl_f1_weighted:.3f}\n"
            f"(multi-label F1_micro={ml_f1_micro:.3f}  Jaccard_macro={ml_jaccard_macro:.3f})"
        ),
    )
    _plot_per_leaf_bars(
        classes_sorted, per_class, gt_support,
        sample_out / "per_leaf_metrics.png",
        title=(
            f"{dataset} / {sample} — multi-label per-class metrics  "
            f"(leaves ∪ Discard;  F1_macro={ml_f1_macro:.3f}, "
            f"Jaccard_macro={ml_jaccard_macro:.3f})"
        ),
    )

    n_cascade = _render_cascade_plots(
        sample_out, gt_df, pred_df, gating_plan, dataset, sample,
    )
    if n_cascade > 0:
        print(f"  cascade plots: {n_cascade} step(s) -> cascade_summary.png")

    return metrics


def _enumerate_output_samples(output_dir: Path, datasets, samples) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ds_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        ds = ds_dir.name
        if datasets and ds not in datasets:
            continue
        for sample_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            s = sample_dir.name
            if samples and s not in samples:
                continue
            out.append((ds, s))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flat-annotation evaluation (leaf-set + multi-label) for cascade outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Cascade output dir (contains {dataset}/{sample}/predicted_steps.parquet).")
    parser.add_argument("--gating-plan", type=Path, required=True,
                        help="Same gating_plan.json that was used for the cascade.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/"))
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark/"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--samples", nargs="*", default=None)
    args = parser.parse_args()

    if not args.gating_plan.exists():
        raise SystemExit(f"[ERROR] gating plan not found: {args.gating_plan}")
    with open(args.gating_plan) as f:
        gating_plan = json.load(f)

    leaf_set = derive_leaf_set(gating_plan)
    print(f"Discovered {len(leaf_set)} leaf categories from gating plan: "
          f"{sorted(leaf_set)}")

    loader = BenchmarkLoader(benchmark_dir=args.benchmark, data_dir=args.data_dir)

    pairs = _enumerate_output_samples(args.output_dir, args.datasets, args.samples)
    if not pairs:
        print("[WARN] no samples matched in output_dir; nothing to evaluate.")
        return

    summaries: list[dict] = []
    for ds, sample in pairs:
        print(f"[eval] {ds}/{sample}")
        m = evaluate_sample(loader, ds, sample, args.output_dir, gating_plan)
        if m is None:
            continue
        summaries.append(m)
        ml = m["multi_label"]
        sl = m["primary_leaf"]
        print(
            f"  n={m['n_cells']:,}  "
            f"gt_discard={m['n_cells_gt_discard']:,}  "
            f"pred_discard={m['n_cells_pred_discard']:,}  "
            f"multi_gt={m['n_cells_multi_gt']:,}  multi_pred={m['n_cells_multi_pred']:,}"
        )
        print(
            f"  multi-label  F1_micro={ml['f1_micro']:.3f}  "
            f"F1_macro={ml['f1_macro']:.3f}  "
            f"Jaccard_macro={ml['jaccard_macro']:.3f}  "
            f"Hamming={ml['hamming_loss']:.4f}"
        )
        print(
            f"  primary-leaf acc={sl['accuracy']:.3f}  "
            f"bal_acc={sl['balanced_accuracy']:.3f}  "
            f"F1_macro={sl['f1_macro']:.3f}"
        )

    if summaries:
        agg = {
            "n_samples": len(summaries),
            "multi_label": {
                "mean_f1_micro": float(np.mean([s["multi_label"]["f1_micro"] for s in summaries])),
                "mean_f1_macro": float(np.mean([s["multi_label"]["f1_macro"] for s in summaries])),
                "mean_f1_weighted": float(np.mean([s["multi_label"]["f1_weighted"] for s in summaries])),
                "mean_jaccard_micro": float(np.mean([s["multi_label"]["jaccard_micro"] for s in summaries])),
                "mean_jaccard_macro": float(np.mean([s["multi_label"]["jaccard_macro"] for s in summaries])),
                "mean_hamming_loss": float(np.mean([s["multi_label"]["hamming_loss"] for s in summaries])),
                "mean_subset_accuracy": float(np.mean([s["multi_label"]["subset_accuracy"] for s in summaries])),
            },
            "primary_leaf": {
                "mean_accuracy": float(np.mean([s["primary_leaf"]["accuracy"] for s in summaries])),
                "mean_balanced_accuracy": float(np.mean([s["primary_leaf"]["balanced_accuracy"] for s in summaries])),
                "mean_f1_macro": float(np.mean([s["primary_leaf"]["f1_macro"] for s in summaries])),
                "mean_f1_weighted": float(np.mean([s["primary_leaf"]["f1_weighted"] for s in summaries])),
            },
            "samples": [
                {
                    "dataset": s["dataset"], "sample": s["sample"],
                    "ml_f1_macro": s["multi_label"]["f1_macro"],
                    "ml_jaccard_macro": s["multi_label"]["jaccard_macro"],
                    "primary_acc": s["primary_leaf"]["accuracy"],
                }
                for s in summaries
            ],
        }
        with open(args.output_dir / "eval_summary.json", "w") as f:
            json.dump(agg, f, indent=2)
        print(
            f"\nAggregate ({len(summaries)} sample(s)):"
            f"\n  multi-label  mean_F1_micro={agg['multi_label']['mean_f1_micro']:.3f}  "
            f"mean_F1_macro={agg['multi_label']['mean_f1_macro']:.3f}  "
            f"mean_Jaccard_macro={agg['multi_label']['mean_jaccard_macro']:.3f}"
            f"\n  primary-leaf mean_acc={agg['primary_leaf']['mean_accuracy']:.3f}  "
            f"mean_bal_acc={agg['primary_leaf']['mean_balanced_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
