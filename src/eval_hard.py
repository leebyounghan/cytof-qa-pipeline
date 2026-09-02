"""Hard (Setting 2) evaluation + magnitude-comparison plots.

Walks a result tree organised as

    {root}/{scenario}/{dataset}_step{NN}/{magnitude_dir}/{dataset}/
        {sample}/step_{NN}/prediction.json
                        /pred.json   # raw LLM output (gates) — optional

where ``{magnitude_dir}`` is one of:

* ``frac{NN}`` — depletion mode (``Deplete(fraction=NN/100)``);
  resolved against ``benchmark/hard_depletion.yaml``.
* ``scale{NN}`` — shift mode (``ChannelScale`` with magnitude knob NN/100);
  resolved against ``benchmark/hard_shift.yaml``.

A single tree contains exactly one mode (auto-detected from the first
magnitude directory encountered). For each prediction it finds:

  1. Look up the matching ``HardDepletion`` / ``HardShift`` by
     (scenario, dataset, step), parse the magnitude directory.
  2. Re-load GT for the same (dataset, sample, step) with the
     corresponding perturbation so cells line up with the prediction
     array.
  3. Compute per-task metrics via ``src.eval.compute_metrics``.
  4. Aggregate per ``(hard_id, magnitude)``, ``(scenario, magnitude)``,
     ``(dataset, magnitude)``.
  5. (optional) For each (hard_id, sample), draw a 1×K side-by-side
     panel of magnitude-level scatter plots with GT colouring and the
     LLM-emitted gates overlaid as dashed boxes / polygons.

Usage::

    # depletion tree
    uv run python -m src.eval_hard \\
        --root results/gate_gpt-5.4_depletion \\
        --plot-dir results/gate_gpt-5.4_depletion/plots

    # shift tree (mode auto-detected from {scale{NN}} subdirs)
    uv run python -m src.eval_hard \\
        --root results/gate_gpt-5.4_shift \\
        --plot-dir results/gate_gpt-5.4_shift/plots
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.bench import Deplete, Perturbation
from src.eval import (
    _aggregate_metrics,
    _extract_labels_xy,
    _parse_parent_mask,
    _prepare_sample_df,
    _resolve_channel,
    compute_metrics,
)
from src.depletion_metric import (
    aggregate_by_scenario_magnitude as _dep_agg_by_scen,
    aggregate_overall as _dep_agg_overall,
    print_tables as _dep_print_tables,
    split_metrics as _dep_split_metrics,
)
from src.hard_depletions import HardDepletion, load_hard_depletions
from src.hard_shifts import HardShift, load_hard_shifts


Mode = Literal["depletion", "shift"]
_MAGNITUDE_RE = re.compile(r"^(frac|scale)(\d+)$")
_STEP_RE = re.compile(r"^step_(\d+)$")
_PREFIX_TO_MODE: dict[str, Mode] = {"frac": "depletion", "scale": "shift"}


# ── Discovery / parsing ─────────────────────────────────────────────────────

def _parse_magnitude_dir(name: str) -> tuple[Mode, float] | None:
    """``frac50`` → ('depletion', 0.5); ``scale100`` → ('shift', 1.0)."""
    m = _MAGNITUDE_RE.match(name)
    if not m:
        return None
    return _PREFIX_TO_MODE[m.group(1)], int(m.group(2)) / 100.0


def _walk_predictions(root: Path) -> list[dict]:
    """Yield one record per prediction.json under ``root``.

    Uses os.walk(followlinks=True) so symlinked dataset dirs (the
    eval_hard-friendly layout built by scripts/build_eval_hard_symlinks.py)
    are traversed transparently.
    """
    import os as _os
    out: list[dict] = []
    pred_paths: list[Path] = []
    for dirpath, _dirnames, filenames in _os.walk(root, followlinks=True):
        if "prediction.json" in filenames:
            pred_paths.append(Path(dirpath) / "prediction.json")
    for pred_path in sorted(pred_paths):
        try:
            rel = pred_path.relative_to(root).parts
            if len(rel) < 7:
                continue
            scenario, ds_step, mag_dir, dataset, sample, step_dir = rel[:6]
        except (ValueError, IndexError):
            continue

        parsed = _parse_magnitude_dir(mag_dir)
        if parsed is None:
            continue
        mode, magnitude = parsed
        m = _STEP_RE.match(step_dir)
        if not m:
            continue
        step_num = int(m.group(1))

        raw_pred = root / scenario / ds_step / mag_dir / dataset / "pred.json"

        out.append({
            "scenario":   scenario,
            "dataset":    dataset,
            "sample":     sample,
            "step":       step_num,
            "mode":       mode,
            "magnitude":  magnitude,
            "mag_dir":    mag_dir,
            "pred_path":  pred_path,
            "raw_pred":   raw_pred,
        })
    return out


def _autodetect_mode(records: list[dict]) -> Mode:
    """A tree must contain exactly one mode; refuse to mix."""
    modes = {r["mode"] for r in records}
    if not modes:
        raise SystemExit(
            "[ERROR] No prediction.json with a recognised magnitude directory "
            "(frac{NN} or scale{NN}) found under the given root."
        )
    if len(modes) > 1:
        raise SystemExit(
            f"[ERROR] Mixed modes found in tree: {sorted(modes)}. "
            "A single root must contain one of {depletion, shift}."
        )
    return next(iter(modes))


def _meta_channels(data_dir: Path, dataset: str) -> dict | None:
    p = data_dir / dataset / "parquet" / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["channels"]


def _build_perturbation(
    mode: Mode, hard, magnitude: float, data_dir: Path,
) -> Perturbation:
    """Materialize the per-record perturbation."""
    if mode == "depletion":
        return Deplete(
            column=hard.step_column,
            values=list(hard.deplete),
            fraction=magnitude,
        )
    # shift
    cs, _slug = hard.to_perturbation(
        meta_channels=_meta_channels(data_dir, hard.dataset),
        scale=magnitude,
    )
    return cs


# ── Metrics ─────────────────────────────────────────────────────────────────

def evaluate(
    root: Path,
    benchmark_dir: Path,
    data_dir: Path,
    depletion_yaml: Path,
    shift_yaml: Path,
    scenarios: set[str] | None = None,
    magnitudes: set[float] | None = None,
    datasets: set[str] | None = None,
) -> dict:
    """Compute magnitude-aware metrics over every prediction in the tree."""
    records = _walk_predictions(root)
    n_seen = len(records)
    mode = _autodetect_mode(records)
    if datasets:
        records = [r for r in records if r["dataset"] in datasets]

    if mode == "depletion":
        _, hard_list = load_hard_depletions(depletion_yaml)
        yaml_path = depletion_yaml
    else:
        _, hard_list = load_hard_shifts(shift_yaml)
        yaml_path = shift_yaml
    by_key = {(h.scenario, h.dataset, h.step): h for h in hard_list}

    # Pre-filter and sort by (dataset, sample) so contiguous records share a
    # baseline parquet load. The single-slot cache avoids the previous per-
    # record parquet I/O — different (hard_id, mag) on the same sample now
    # apply their perturbation in-memory on top of one shared baseline df.
    # Both Deplete (row filter) and ChannelScale (channel rescale via internal
    # df.copy()) are pure functions wrt input — safe to share the baseline.
    filtered: list[tuple[dict, object]] = []
    for rec in records:
        if scenarios and rec["scenario"] not in scenarios:
            continue
        if magnitudes is not None and rec["magnitude"] not in magnitudes:
            continue
        hard = by_key.get((rec["scenario"], rec["dataset"], rec["step"]))
        if hard is None:
            continue
        filtered.append((rec, hard))
    filtered.sort(key=lambda rh: (rh[0]["dataset"], rh[0]["sample"]))

    baseline_key: tuple[str, str] | None = None
    baseline_df: pd.DataFrame | None = None
    baseline_chmap: dict | None = None

    per_task: list[dict] = []
    n_matched = 0
    n_skipped_size_mismatch = 0

    for rec, hard in filtered:
        key = (rec["dataset"], rec["sample"])
        if key != baseline_key:
            baseline_df, baseline_chmap, _ = _prepare_sample_df(
                data_dir, rec["dataset"], rec["sample"],
                benchmark_dir=benchmark_dir,
                perturbation=None,
            )
            baseline_key = key

        perturbation = _build_perturbation(mode, hard, rec["magnitude"], data_dir)
        df = perturbation.apply(
            baseline_df, context={"channel_marker_map": baseline_chmap}
        )
        ch_map = baseline_chmap

        task_path = (
            benchmark_dir / rec["dataset"] / rec["sample"]
            / f'step_{rec["step"]:02d}' / "task.json"
        )
        if not task_path.exists():
            continue
        with open(task_path) as f:
            task = json.load(f)

        with open(rec["pred_path"]) as f:
            pred = json.load(f)
        pred_labels = np.array(pred["predicted_labels"], dtype=object)

        gt_labels, gt_xy = _extract_labels_xy(df, task, ch_map)
        if len(gt_labels) != len(pred_labels):
            print(f"  [SKIP] {hard.hard_id} {rec['sample']} "
                  f"{rec['mag_dir']}: gt={len(gt_labels)} pred={len(pred_labels)}")
            n_skipped_size_mismatch += 1
            continue
        if len(gt_labels) == 0:
            continue

        pred_eval = pred_labels.copy()
        pred_eval[pred_eval == None] = "__noise__"  # noqa: E711

        metrics = compute_metrics(
            gt_labels, pred_eval, list(task["categories"]), xy=gt_xy,
        )
        metrics.update({
            "hard_id":   hard.hard_id,
            "scenario":  hard.scenario,
            "dataset":   hard.dataset,
            "step":      hard.step,
            "sample":    rec["sample"],
            "mode":      mode,
            "magnitude": rec["magnitude"],
            "mag_dir":   rec["mag_dir"],
            "n_cells":   int(len(gt_labels)),
        })
        per_task.append(metrics)
        n_matched += 1

    # Aggregations
    by_hard:  dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_scen:  dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_ds:    dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in per_task:
        by_hard[(m["hard_id"], m["mag_dir"])].append(m)
        by_scen[(m["scenario"], m["mag_dir"])].append(m)
        by_ds[(m["dataset"],  m["mag_dir"])].append(m)

    def _grouped(buckets):
        out: dict[str, dict] = defaultdict(dict)
        for (key, mag_dir), rows in sorted(buckets.items()):
            out[key][mag_dir] = {
                "n_samples": len(rows),
                **(_aggregate_metrics(rows) or {}),
            }
        return dict(out)

    out = {
        "root":             str(root),
        "mode":             mode,
        "yaml_path":        str(yaml_path),
        "n_predictions_seen":    n_seen,
        "n_predictions_matched": n_matched,
        "n_predictions_skipped_size_mismatch": n_skipped_size_mismatch,
        "per_task":         per_task,
        "by_hard_task":     _grouped(by_hard),
        "by_scenario":      _grouped(by_scen),
        "by_dataset":       _grouped(by_ds),
    }

    # Depleted-vs-sibling split (prior-invariant recall + IoU) — depletion only.
    # F1 macro on the depleted candidate is prior-sensitive; see
    # src.depletion_metric for the rationale.
    if mode == "depletion":
        deplete_by_key = {
            (h.scenario, h.dataset, h.step): h.deplete for h in hard_list
        }
        per_task_split = _dep_split_metrics(per_task, deplete_by_key)
        out["depleted_vs_sibling"] = {
            "by_scenario": _dep_agg_by_scen(per_task_split),
            "overall":     _dep_agg_overall(per_task_split),
            "per_task":    per_task_split,
            "mag_order":   sorted({r["mag_dir"] for r in per_task_split}),
        }

    return out


# ── Plotting ────────────────────────────────────────────────────────────────

def _draw_gate(ax, gate: dict, color, lw: float = 2.0):
    """Draw a single gate (rectangle or polygon) as a dashed outline."""
    if gate is None:
        return
    if "vertices" in gate:
        verts = np.array(gate["vertices"] + [gate["vertices"][0]])
        ax.plot(verts[:, 0], verts[:, 1], color=color, lw=lw, ls="--")
    else:
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle(
            (gate["x_min"], gate["y_min"]),
            gate["x_max"] - gate["x_min"],
            gate["y_max"] - gate["y_min"],
            fill=False, edgecolor=color, lw=lw, ls="--",
        ))


def _scatter_gt(ax, x, y, gt, categories, color_map, max_cells: int):
    """Scatter parent cells coloured by GT label."""
    if len(x) > max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x), max_cells, replace=False)
        x, y, gt = x[idx], y[idx], gt[idx]
    for cat in categories:
        m = gt == cat
        if not m.any():
            continue
        ax.scatter(x[m], y[m], s=4, alpha=0.45, linewidths=0,
                   c=[color_map[cat]], label=cat, rasterized=True)


def generate_comparison_plots(
    summary: dict,
    benchmark_dir: Path,
    data_dir: Path,
    plot_dir: Path,
    max_cells: int = 5000,
) -> int:
    """One side-by-side figure per (hard_id, sample), spanning all magnitudes."""
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in summary["per_task"]:
        by_pair[(t["hard_id"], t["sample"])].append(t)

    yaml_path = Path(summary["yaml_path"])
    mode: Mode = summary["mode"]
    if mode == "depletion":
        _, hard_list = load_hard_depletions(yaml_path)
    else:
        _, hard_list = load_hard_shifts(yaml_path)
    hard_by_id = {h.hard_id: h for h in hard_list}

    n_written = 0
    for (hard_id, sample), tasks in sorted(by_pair.items()):
        hard = hard_by_id.get(hard_id)
        if hard is None:
            continue
        tasks = sorted(tasks, key=lambda t: t["magnitude"])
        K = len(tasks)
        fig, axes = plt.subplots(
            1, K, figsize=(6 * K, 6),
            sharex=True, sharey=True, squeeze=False,
        )

        task_path = (
            benchmark_dir / hard.dataset / sample
            / f"step_{hard.step:02d}" / "task.json"
        )
        with open(task_path) as f:
            task = json.load(f)
        categories = list(task["categories"])
        cmap = plt.get_cmap("tab10")
        color_map = {c: cmap(i % 10) for i, c in enumerate(categories)}

        for col, t in enumerate(tasks):
            ax = axes[0, col]
            perturbation = _build_perturbation(
                mode, hard, t["magnitude"], data_dir,
            )
            df, ch_map, _ = _prepare_sample_df(
                data_dir, hard.dataset, sample,
                benchmark_dir=benchmark_dir,
                perturbation=perturbation,
            )
            x_ch = _resolve_channel(task["x_marker"], list(df.columns), ch_map)
            y_ch = _resolve_channel(task["y_marker"], list(df.columns), ch_map)
            mask = _parse_parent_mask(df, task["parent"])
            x = df.loc[mask, x_ch].values.astype(np.float32)
            y = df.loc[mask, y_ch].values.astype(np.float32)
            gt = df.loc[mask, task["annotation_column"]].fillna("Unassigned").to_numpy(dtype=object)
            na = pd.isna(gt)
            if na.any():
                gt[na] = "Unassigned"

            _scatter_gt(ax, x, y, gt, categories, color_map, max_cells)

            raw_path = Path(summary["root"]) / hard.scenario \
                / f"{hard.dataset}_step{hard.step:02d}" / t["mag_dir"] \
                / hard.dataset / "pred.json"
            if raw_path.exists():
                with open(raw_path) as f:
                    raw = json.load(f)
                step_recs = raw.get("samples", {}).get(sample, {}).get("steps", {})
                if step_recs:
                    rec = next(iter(step_recs.values()))
                    for cat, g in (rec.get("gates", {}) or {}).items():
                        if g is None:
                            continue
                        gs = g if isinstance(g, list) else [g]
                        for gi in gs:
                            _draw_gate(ax, gi, color_map.get(cat, "k"))

            ax.set_xlabel(task["x_marker"], fontsize=9)
            if col == 0:
                ax.set_ylabel(task["y_marker"], fontsize=9)
            f1 = t.get("f1_macro")
            f1_str = f"{f1:.3f}" if f1 is not None else "n/a"
            label = "frac" if mode == "depletion" else "scale"
            ax.set_title(
                f'{label}={int(round(t["magnitude"] * 100))}%   '
                f'n={t["n_cells"]:,}   F1={f1_str}',
                fontsize=10,
            )
            ax.grid(True, alpha=0.2)
            if col == K - 1:
                ax.legend(fontsize=7, markerscale=3, loc="best")

        fig.suptitle(
            f"{hard_id}  —  {sample}  (dashed = LLM gates)",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out = plot_dir / hard.scenario / f"{hard.dataset}_step{hard.step:02d}__{sample}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        n_written += 1
        print(f"  plot → {out}")

    return n_written


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hard (Setting 2) evaluation across depletion fractions / "
                    "shift scales. Mode is auto-detected from the magnitude "
                    "directory naming pattern in the tree.",
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Root of the result tree, e.g. results/gate_gpt-5.4_depletion "
             "or results/gate_gpt-5.4_shift",
    )
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark/"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/"),
                        dest="data_dir")
    parser.add_argument(
        "--hard-depletion-yaml", type=Path,
        default=Path("benchmark/hard_depletion.yaml"),
        help="Used in depletion mode (default: benchmark/hard_depletion.yaml)",
    )
    parser.add_argument(
        "--hard-shift-yaml", type=Path,
        default=Path("benchmark/hard_shift.yaml"),
        help="Used in shift mode (default: benchmark/hard_shift.yaml)",
    )
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Restrict to these scenarios (default: all).")
    parser.add_argument(
        "--magnitudes", type=float, nargs="*", default=None,
        help="Restrict to these magnitudes (e.g. 0 0.5 1.0). "
             "Default: all magnitudes found on disk.",
    )
    parser.add_argument(
        "--summary-out", type=Path, default=None,
        help="Write summary JSON here. Default: {root}/eval_hard_summary.json",
    )
    parser.add_argument(
        "--plot-dir", type=Path, default=None,
        help="If set, write per-(hard_id, sample) side-by-side plots here.",
    )
    parser.add_argument("--max-cells", type=int, default=5000,
                        help="Max cells per scatter panel (default: 5000).")
    args = parser.parse_args()

    magnitudes = set(args.magnitudes) if args.magnitudes else None
    scenarios = set(args.scenarios) if args.scenarios else None

    summary = evaluate(
        root=args.root,
        benchmark_dir=args.benchmark,
        data_dir=args.data_dir,
        depletion_yaml=args.hard_depletion_yaml,
        shift_yaml=args.hard_shift_yaml,
        scenarios=scenarios,
        magnitudes=magnitudes,
    )

    out = args.summary_out or (args.root / "eval_hard_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")
    print(f"Mode: {summary['mode']}  |  predictions seen: {summary['n_predictions_seen']}  "
          f"matched: {summary['n_predictions_matched']}  "
          f"skipped (size mismatch): {summary['n_predictions_skipped_size_mismatch']}")

    print(f"\nBy scenario × magnitude (macro F1):")
    for scen, mag_blocks in summary["by_scenario"].items():
        cells = "  ".join(
            f"{mk}={blk.get('f1_macro_mean', float('nan')):.4f}"
            f"(n={blk['n_samples']})"
            for mk, blk in sorted(mag_blocks.items())
        )
        print(f"  {scen:25s}  {cells}")

    print(f"\nBy dataset × magnitude (macro F1):")
    for ds, mag_blocks in summary["by_dataset"].items():
        cells = "  ".join(
            f"{mk}={blk.get('f1_macro_mean', float('nan')):.4f}"
            f"(n={blk['n_samples']})"
            for mk, blk in sorted(mag_blocks.items())
        )
        print(f"  {ds:25s}  {cells}")

    if "depleted_vs_sibling" in summary:
        dvs = summary["depleted_vs_sibling"]
        print("\n=== Depleted vs sibling (recall + Hull IoU, prior-invariant) ===")
        _dep_print_tables(dvs["by_scenario"], dvs["overall"], dvs["mag_order"])

    if args.plot_dir is not None:
        print(f"\nGenerating plots → {args.plot_dir}")
        n = generate_comparison_plots(
            summary,
            benchmark_dir=args.benchmark,
            data_dir=args.data_dir,
            plot_dir=args.plot_dir,
            max_cells=args.max_cells,
        )
        print(f"  wrote {n} figures")


if __name__ == "__main__":
    main()
