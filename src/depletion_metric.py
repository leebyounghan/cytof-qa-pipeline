"""Depleted-vs-sibling metric extraction for the depletion robustness experiment.

The headline F1 macro on the depleted candidate is *prior-sensitive*: as the
target prevalence ρ drops, precision mechanically collapses (FP from the
abundant siblings stays constant in absolute terms while TP scales with ρ),
so the F1 curve mostly tracks the metric, not the model. Two consequences:

* **Direct effect** — does the model still recall the rare cells that survived
  depletion? → mean **recall** over the depleted candidate(s).
* **Indirect / collateral effect** — does perturbing one candidate degrade
  the classification of the other (untouched) candidates of the same step,
  because the input distribution as a whole shifted? → mean **recall** over
  the sibling candidates of the same step.

Both are prior-invariant, so they are comparable across ρ. We also report
**Hull IoU** (geometric overlap of predicted vs ground-truth gate) for the
same depleted/sibling split as a geometric anchor.

Inputs: a depletion-mode ``eval_hard_summary.json`` (produced by
``src.eval_hard``) plus the matching ``hard_depletion.yaml`` (used to look up
which candidate(s) are the "depleted target" per (scenario, dataset, step)).

Outputs: a JSON with per-(scenario, ρ) means and the per-row breakdown, and a
CSV mirror of the (scenario × ρ) headline grid.

Usage::

    uv run python -m src.depletion_metric \\
        --summary results/gate_gpt-5.4_depletion/eval_hard_summary.json \\
        --hard-depletion-yaml benchmark/hard_depletion.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.hard_depletions import load_hard_depletions


# ── Aggregation ─────────────────────────────────────────────────────────────

def _safe_mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def split_metrics(
    per_task: list[dict],
    deplete_by_key: dict[tuple[str, str, int], tuple[str, ...]],
) -> list[dict]:
    """For every per_task row, compute mean recall / Hull IoU over the
    depleted candidates and over the sibling candidates separately.

    Notes
    -----
    * Candidates whose ``gt_proportion`` is 0 in the perturbed sample are
      skipped — recall is undefined (sklearn would also be 0/0 → 0) and
      including them would bias the mean toward 0 noise.
    * Hull IoU is only present when the candidate has enough GT points to
      form a hull; missing entries are skipped (instead of treated as 0).
    * "Unassigned" is excluded from both sides; it is not a candidate.
    """
    out: list[dict] = []
    for row in per_task:
        key = (row["scenario"], row["dataset"], row["step"])
        depleted_cands = deplete_by_key.get(key)
        if depleted_cands is None:
            continue
        depleted_set = set(depleted_cands)

        per_cat: dict = row.get("per_category", {})

        rec_dep: list[float] = []
        rec_sib: list[float] = []
        iou_dep: list[float] = []
        iou_sib: list[float] = []
        n_dep_cands = 0
        n_sib_cands = 0

        for cat, m in per_cat.items():
            if cat == "Unassigned":
                continue
            is_dep = cat in depleted_set
            if is_dep:
                n_dep_cands += 1
            else:
                n_sib_cands += 1

            if m.get("gt_proportion", 0) > 0 and "recall" in m:
                (rec_dep if is_dep else rec_sib).append(float(m["recall"]))
            if m.get("iou_hull") is not None:
                (iou_dep if is_dep else iou_sib).append(float(m["iou_hull"]))

        out.append({
            "scenario":   row["scenario"],
            "dataset":    row["dataset"],
            "step":       row["step"],
            "sample":     row["sample"],
            "magnitude":  row["magnitude"],
            "mag_dir":    row["mag_dir"],
            "hard_id":    row["hard_id"],
            "n_cells":    row.get("n_cells"),
            "n_depleted_cands": n_dep_cands,
            "n_sibling_cands":  n_sib_cands,
            "recall_depleted":   _safe_mean(rec_dep),
            "recall_siblings":   _safe_mean(rec_sib),
            "iou_hull_depleted": _safe_mean(iou_dep),
            "iou_hull_siblings": _safe_mean(iou_sib),
        })
    return out


def aggregate_by_scenario_magnitude(per_task_split: list[dict]) -> dict:
    """Mean across (sample, dataset, step) within each (scenario, mag_dir)."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in per_task_split:
        buckets[(r["scenario"], r["mag_dir"])].append(r)

    out: dict = defaultdict(dict)
    for (scen, mag_dir), rows in sorted(buckets.items()):
        for k in (
            "recall_depleted", "recall_siblings",
            "iou_hull_depleted", "iou_hull_siblings",
        ):
            vals = [r[k] for r in rows if r[k] is not None]
            out[scen][mag_dir] = out[scen].get(mag_dir, {})
            out[scen][mag_dir][k] = _safe_mean(vals)
            out[scen][mag_dir]["n_rows"] = len(rows)
            out[scen][mag_dir][f"n_{k}_present"] = len(vals)
    return dict(out)


def aggregate_overall(per_task_split: list[dict]) -> dict:
    """Mean across scenarios → one row per mag_dir.

    Computed as the mean of the per-(scenario, mag_dir) means rather than a
    pooled mean across raw rows so each scenario contributes equally — same
    convention as the *Mean* row in tab:shift-depletion.
    """
    by_scen = aggregate_by_scenario_magnitude(per_task_split)
    mag_keys: set[str] = set()
    for scen, blocks in by_scen.items():
        mag_keys.update(blocks.keys())

    out: dict = {}
    for mag_dir in sorted(mag_keys):
        agg: dict[str, list[float]] = defaultdict(list)
        for scen, blocks in by_scen.items():
            blk = blocks.get(mag_dir)
            if blk is None:
                continue
            for k in (
                "recall_depleted", "recall_siblings",
                "iou_hull_depleted", "iou_hull_siblings",
            ):
                v = blk.get(k)
                if v is not None:
                    agg[k].append(v)
        out[mag_dir] = {k: _safe_mean(v) for k, v in agg.items()}
        out[mag_dir]["n_scenarios"] = max(
            (len(v) for v in agg.values()), default=0,
        )
    return out


# ── Pretty printing ─────────────────────────────────────────────────────────

_METRICS = (
    ("recall_depleted",   "Recall (depleted)"),
    ("recall_siblings",   "Recall (siblings)"),
    ("iou_hull_depleted", "Hull IoU (depleted)"),
    ("iou_hull_siblings", "Hull IoU (siblings)"),
)


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  n/a "


def print_tables(by_scen: dict, overall: dict, mag_order: list[str]) -> None:
    for key, label in _METRICS:
        print(f"\n=== {label} — by scenario × ρ ===")
        header = "scenario".ljust(28) + "  ".join(m.rjust(7) for m in mag_order)
        print(header)
        print("-" * len(header))
        for scen in sorted(by_scen.keys()):
            cells = [
                _fmt(by_scen[scen].get(m, {}).get(key))
                for m in mag_order
            ]
            print(f"{scen:28s}" + "  ".join(c.rjust(7) for c in cells))
        # mean row
        cells = [_fmt(overall.get(m, {}).get(key)) for m in mag_order]
        print(f"{'MEAN':28s}" + "  ".join(c.rjust(7) for c in cells))


def write_csv(out_path: Path, by_scen: dict, overall: dict,
              mag_order: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "scenario", *mag_order])
        for key, label in _METRICS:
            for scen in sorted(by_scen.keys()):
                w.writerow([
                    label, scen,
                    *[
                        _fmt(by_scen[scen].get(m, {}).get(key)).strip()
                        for m in mag_order
                    ],
                ])
            w.writerow([
                label, "MEAN",
                *[_fmt(overall.get(m, {}).get(key)).strip() for m in mag_order],
            ])
    print(f"\nSaved CSV: {out_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recompute depletion-experiment metrics split by "
                    "depleted vs sibling candidates (recall + Hull IoU).",
    )
    ap.add_argument(
        "--summary", type=Path, required=True,
        help="Path to a depletion-mode eval_hard_summary.json "
             "(produced by src.eval_hard).",
    )
    ap.add_argument(
        "--hard-depletion-yaml", type=Path,
        default=Path("benchmark/hard_depletion.yaml"),
        help="Used to map (scenario, dataset, step) → deplete candidates.",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON. Default: <summary>.dep_vs_sib.json next to input.",
    )
    ap.add_argument(
        "--csv", type=Path, default=None,
        help="Optional CSV mirror of the per-scenario × ρ headline grid. "
             "Default: <summary>.dep_vs_sib.csv next to input.",
    )
    args = ap.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)
    if summary.get("mode") != "depletion":
        raise SystemExit(
            f"[ERROR] summary mode={summary.get('mode')!r}, expected 'depletion'."
        )

    _, hard_list = load_hard_depletions(args.hard_depletion_yaml)
    deplete_by_key = {
        (h.scenario, h.dataset, h.step): h.deplete for h in hard_list
    }

    per_task_split = split_metrics(summary["per_task"], deplete_by_key)
    by_scen = aggregate_by_scenario_magnitude(per_task_split)
    overall = aggregate_overall(per_task_split)

    mag_order = sorted({r["mag_dir"] for r in per_task_split})

    out_path = args.out or args.summary.with_suffix(".dep_vs_sib.json")
    csv_path = args.csv or args.summary.with_suffix(".dep_vs_sib.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "summary":         str(args.summary),
            "yaml":            str(args.hard_depletion_yaml),
            "n_per_task_rows": len(per_task_split),
            "mag_order":       mag_order,
            "by_scenario":     by_scen,
            "overall":         overall,
            "per_task":        per_task_split,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {out_path}")

    write_csv(csv_path, by_scen, overall, mag_order)
    print_tables(by_scen, overall, mag_order)


if __name__ == "__main__":
    main()
