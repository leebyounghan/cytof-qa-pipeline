"""Score every C2S cluster leaf against the shared flat evaluator's GT sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.bench import BenchmarkLoader
from src.flat_eval import DISCARD_LABEL, _flat_labels_multilabel, derive_leaf_set


def evaluate_sample(
    loader: BenchmarkLoader,
    dataset: str,
    sample: str,
    clusters: dict,
    c2c_path: Path,
    plan: dict,
) -> dict:
    gt_df, _ = loader._load_sample(dataset, sample)
    step_cols = [s["annotation column name"] for s in plan["steps"]]
    leaf_set = derive_leaf_set(plan)
    classes = sorted(leaf_set | {DISCARD_LABEL})
    gt_multi, _ = _flat_labels_multilabel(gt_df, step_cols, leaf_set)
    with np.load(c2c_path) as c2c:
        cluster_labels = c2c["cluster_labels"]
    if len(gt_multi) != len(cluster_labels):
        raise ValueError(f"{dataset}/{sample}: GT and cluster rows differ")
    if not len(cluster_labels):
        raise ValueError(f"{dataset}/{sample}: empty cluster assignments")

    n_clusters = int(cluster_labels.max()) + 1
    expected = {f"cluster_{k + 1}" for k in range(n_clusters)}
    if set(clusters) != expected:
        raise ValueError(f"{dataset}/{sample}: prediction cluster IDs do not match assignments")
    cluster_sets = []
    same_step_collisions = 0
    leaf_to_col = {}
    for step in plan["steps"]:
        for leaf in step.get("annotation categories", []):
            if leaf in leaf_set:
                leaf_to_col[leaf] = step["annotation column name"]
    for k in range(n_clusters):
        leaves = clusters[f"cluster_{k + 1}"].get("leaves")
        if not isinstance(leaves, list) or not leaves:
            raise ValueError(f"{dataset}/{sample}/cluster_{k + 1}: empty or invalid leaves")
        if any(not isinstance(v, str) or v not in classes for v in leaves):
            raise ValueError(f"{dataset}/{sample}/cluster_{k + 1}: unknown leaf")
        if DISCARD_LABEL in leaves and len(leaves) != 1:
            raise ValueError(f"{dataset}/{sample}/cluster_{k + 1}: Discard mixed with leaves")
        counts = {}
        for leaf in leaves:
            if leaf != DISCARD_LABEL:
                col = leaf_to_col[leaf]
                counts[col] = counts.get(col, 0) + 1
        if any(n > 1 for n in counts.values()):
            same_step_collisions += 1
        cluster_sets.append(set(leaves))

    pred_multi = [cluster_sets[int(k)] for k in cluster_labels]
    mlb = MultiLabelBinarizer(classes=classes)
    gt_bin = mlb.fit_transform(gt_multi)
    pred_bin = mlb.transform(pred_multi)
    report = classification_report(
        gt_bin, pred_bin, target_names=classes, output_dict=True, zero_division=0,
    )
    return {
        "dataset": dataset,
        "sample": sample,
        "n_cells": len(gt_multi),
        "n_clusters": n_clusters,
        "n_same_step_collision_clusters": same_step_collisions,
        "classes": classes,
        "n_cells_gt_discard": sum(s == {DISCARD_LABEL} for s in gt_multi),
        "n_cells_pred_discard": sum(s == {DISCARD_LABEL} for s in pred_multi),
        "n_cells_multi_gt": sum(len(s) > 1 for s in gt_multi),
        "n_cells_multi_pred": sum(len(s) > 1 for s in pred_multi),
        "multi_label": {
            "subset_accuracy": float(accuracy_score(gt_bin, pred_bin)),
            "hamming_loss": float(hamming_loss(gt_bin, pred_bin)),
            "f1_micro": float(f1_score(gt_bin, pred_bin, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(gt_bin, pred_bin, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(gt_bin, pred_bin, average="weighted", zero_division=0)),
            "jaccard_micro": float(jaccard_score(gt_bin, pred_bin, average="micro", zero_division=0)),
            "jaccard_macro": float(jaccard_score(gt_bin, pred_bin, average="macro", zero_division=0)),
            "per_class": {c: report[c] for c in classes},
            "gt_support": {c: int(gt_bin[:, i].sum()) for i, c in enumerate(classes)},
            "pred_support": {c: int(pred_bin[:, i].sum()) for i, c in enumerate(classes)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--prep-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("benchmark/splits.json"))
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    pred = json.loads(args.pred.read_text())
    samples = json.loads(args.splits.read_text())["in_panel"][args.dataset]["test"]
    sample_predictions = pred["samples"]
    if set(samples) != set(sample_predictions):
        raise ValueError(f"{args.dataset}: test samples differ from pred.json")
    loader = BenchmarkLoader(benchmark_dir=Path("benchmark"), data_dir=Path("data"))
    metrics = []
    for sample_index, sample in enumerate(samples, start=1):
        c2c = args.prep_root / args.dataset / sample / "cell2cluster.npz"
        if not c2c.exists():
            raise FileNotFoundError(c2c)
        value = evaluate_sample(
            loader, args.dataset, sample,
            sample_predictions[sample]["clusters"], c2c, plan,
        )
        out = args.output_root / args.dataset / sample / "eval_flat.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(value, indent=2))
        metrics.append(value)
        print(f"{args.dataset}: sample {sample_index}/{len(samples)} evaluated", flush=True)

    keys = ("subset_accuracy", "hamming_loss", "f1_micro", "f1_macro", "f1_weighted",
            "jaccard_micro", "jaccard_macro")
    summary = {
        "dataset": args.dataset,
        "plan": str(args.plan),
        "n_samples": len(metrics),
        "n_cells": sum(m["n_cells"] for m in metrics),
        "n_same_step_collision_clusters": sum(m["n_same_step_collision_clusters"] for m in metrics),
        "multi_label": {f"mean_{key}": float(np.mean([m['multi_label'][key] for m in metrics]))
                        for key in keys},
        "samples": [{"sample": m["sample"], "f1_macro": m["multi_label"]["f1_macro"],
                     "f1_micro": m["multi_label"]["f1_micro"]} for m in metrics],
    }
    output = args.output_root / f"eval_summary_{args.dataset}.json"
    output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
