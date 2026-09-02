#!/usr/bin/env python3
"""Aggregate seed-sweep eval_summary.json into per-cohort and Mean
mean±std, for any paradigm (gate / density_grid / clustering).

Usage:
    python3 scripts/aggregate_seed_sweep.py [--paradigm gate|density_grid|clustering]
                                            [--metric f1_macro_mean|balanced_accuracy_mean|iou_hull_macro_mean]
                                            [--latex]

Reports overall Mean (mean of per-cohort means) with std computed across
seeds' per-run Mean, plus per-cohort breakdown. Only seeds with all 11
cohorts present are counted.
"""
import argparse, json, os
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "results", "seed_sweep")

COHORTS = ["Acute2020", "Acute2021", "Vaccine", "Bjornson",
           "FR-FCM-Z74D_hc", "FR-FCM-Z74D_tissue", "FRDR_covid19",
           "Lyoplate_tcell", "Lyoplate_treg", "Lyoplate_bcell", "Lyoplate_DC"]

MODELS = ["qwen3_5_4b", "qwen3_5_27b", "qwen3_6_27b",
          "gemma_4_26b_a4b", "gemma_4_31b"]
# gate sweep additionally has these:
GATE_EXTRA = []

DISPLAY = {
    "qwen3_5_4b": "Qwen3.5-4B",
    "qwen3_5_27b": "Qwen3.5-27B",
    "qwen3_6_27b": "Qwen3.6-27B", "gemma_4_26b_a4b": "Gemma 4 26B A4B",
    "gemma_4_31b": "Gemma 4 31B",
}
METRICS = ["f1_macro_mean", "balanced_accuracy_mean", "iou_hull_macro_mean"]


def collect(paradigm, model):
    """Return {seed: {cohort: {metric: val}}} for complete seeds only."""
    root = os.path.join(BASE, paradigm)
    out = {}
    if not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        if not d.startswith(f"{model}_seed"):
            continue
        seed = d[len(f"{model}_seed"):]
        pred = os.path.join(root, d, "predictions")
        vals = {}
        ok = True
        for c in COHORTS:
            fp = os.path.join(pred, c, "eval_summary.json")
            if not os.path.exists(fp):
                ok = False
                break
            with open(fp) as f:
                ov = json.load(f)["overall"]
            vals[c] = {m: ov[m] for m in METRICS}
        if ok:
            out[seed] = vals
    return out


def summarize(paradigm, latex=False):
    models = MODELS + (GATE_EXTRA if paradigm == "gate" else [])
    print(f"\n{'='*90}\nParadigm: {paradigm}\n{'='*90}")
    print(f"{'Model':<20} {'#seeds':>6}   {'F1 macro':>14} {'BA':>14} {'Hull IoU':>14}")
    print("-"*90)
    rows = {}
    for model in models:
        data = collect(paradigm, model)
        n = len(data)
        if n == 0:
            print(f"{DISPLAY.get(model, model):<20} {0:>6}   (no complete seeds)")
            continue
        cell = f"{DISPLAY.get(model, model):<20} {n:>6}   "
        rows[model] = {}
        for m in METRICS:
            per_seed_mean = [np.mean([data[s][c][m] for c in COHORTS]) for s in data]
            mu, sd = np.mean(per_seed_mean), (np.std(per_seed_mean, ddof=1) if n > 1 else 0.0)
            rows[model][m] = (mu, sd)
            cell += f"  .{mu*1000:03.0f}±.{sd*1000:03.0f}"
        print(cell)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paradigm", default=None,
                    choices=["gate", "density_grid", "clustering"])
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    paradigms = [args.paradigm] if args.paradigm else ["gate", "density_grid", "clustering"]
    for p in paradigms:
        summarize(p, latex=args.latex)
