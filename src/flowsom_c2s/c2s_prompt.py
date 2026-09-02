"""Render whole-cell LLM prompts from task.json + c2s.json.

For each sample, write ``flowsom_{ablation}/cells.md``. The LLM sees:

  1. Sample / panel context
  2. The flat ``leaf_categories`` list (terminal leaves only — same set as
     ``src/llm_gate_flat/`` via ``src.flat_eval.derive_leaf_set``)
  3. Step descriptions (note + tips) as biological hints
  4. Top HVP markers list (optional, controlled by ablation)
  5. Per-cluster centroid + cell-sentence
  6. Output schema instructions — multi-label leaf SET per cluster

Ablations: ``full`` / ``hvp10`` / ``nohvp``.

Usage:
    python -m src.flowsom_c2s.c2s_prompt --batch benchmark_flat/<ds>/<sample>/ \
        --top-hvp-n 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ── Ablation slug ───────────────────────────────────────────────────────────

def ablation_slug(use_top_hvp: bool, top_hvp_n: int | None) -> str:
    if not use_top_hvp:
        return "nohvp"
    if top_hvp_n is None:
        return "full"
    return f"hvp{top_hvp_n}"


def ablation_dir(sample_dir: Path, slug: str) -> Path:
    return sample_dir / f"flowsom_{slug}"


# ── Prompt rendering ────────────────────────────────────────────────────────

def _format_centroid(centroid: dict, hvp_order: list[str], top_n: int | None) -> str:
    if top_n is not None:
        keep = [m for m in hvp_order if m in centroid][:top_n]
    else:
        keep = list(centroid.keys())
    return ", ".join(f"{m}({centroid[m]:.2f})" for m in keep)


def _sort_parts_by_value(parts: list[str]) -> list[str]:
    """Sort `Marker(value)` tokens by value descending (high → low)."""
    def _val(token: str) -> float:
        try:
            return float(token.split("(", 1)[1].rstrip(")"))
        except (IndexError, ValueError):
            return 0.0
    return sorted(parts, key=lambda t: -_val(t))


def _truncate_sentence(sentence: str, top_n: int | None) -> str:
    """Truncate to top-N HVP markers (variance-ranked in c2s.json), then
    re-sort the kept markers by value descending so the rendered ``>``
    chain genuinely reads high-to-low for this cluster."""
    parts = [p.strip() for p in sentence.split(">")]
    if top_n is not None:
        parts = parts[:top_n]
    parts = _sort_parts_by_value(parts)
    return " > ".join(parts)


def format_prompt(
    task: dict,
    c2s: dict,
    use_top_hvp: bool,
    top_hvp_n: int | None,
) -> str:
    leaves: list[str] = task["leaf_categories"]
    hvp_order = c2s.get("top_hvp_markers", [])
    truncate_n = top_hvp_n if (use_top_hvp and top_hvp_n is not None) else None

    out: list[str] = []

    out.append("[Context]")
    out.append(f"Modality: {task.get('modality', 'unknown')} (cofactor {task.get('cofactor')})")
    out.append(
        f"Sample: {task['sample']} (dataset: {task['dataset']}) -- "
        f"n_cells={task['n_cells']:,}, n_clusters={c2s['n_clusters']}"
    )
    out.append("")

    out.append("[Leaf categories]")
    for leaf in leaves:
        out.append(leaf)
    out.append("Discard")
    out.append("")

    if use_top_hvp and hvp_order:
        ranked = hvp_order if top_hvp_n is None else hvp_order[:top_hvp_n]
        out.append("[Top HVP markers (high -> low variance)]")
        out.append(", ".join(ranked))
        out.append("")

    out.append("---")
    out.append("")

    cluster_keys = sorted(c2s["clusters"].keys(), key=lambda k: int(k))
    for k in cluster_keys:
        cl = c2s["clusters"][k]
        n = cl["n_cells"]
        frac = cl["fraction"] * 100
        out.append(f"## cluster_{k} (n={n:,}, {frac:.2f}%)")
        if use_top_hvp:
            # HVP-ranked sentence (descending variance), optionally truncated.
            sent = _truncate_sentence(cl["cell_sentence"], truncate_n)
            out.append(sent)
        else:
            # No HVP ordering — list all markers in panel order (centroid keys).
            out.append(_format_centroid(cl["centroid"], list(cl["centroid"].keys()), None))
        out.append("")

    out.append("---")
    out.append("")
    out.append("## Output Format")
    out.append(
        "Return a single JSON object keyed by cluster id. For each cluster,\n"
        "`leaves` MUST be a JSON list whose elements come from [Leaf categories]\n"
        '(verbatim) — or the singleton ["Discard"]. `rationale` is one short\n'
        "sentence."
    )
    out.append("")
    out.append("```json")
    out.append("{")
    sample_keys = cluster_keys[:2] if len(cluster_keys) >= 2 else cluster_keys
    sample_lines = []
    for k in sample_keys:
        sample_lines.append(
            f'  "cluster_{k}": {{"leaves": ["<leaf_a>", "<leaf_b>"], "rationale": "..."}}'
        )
    out.append(",\n".join(sample_lines))
    out.append("}")
    out.append("```")

    return "\n".join(out)


def render_for_sample(
    sample_dir: Path,
    plan_slug: str,
    use_top_hvp: bool,
    top_hvp_n: int | None,
) -> Path | None:
    """Render cells.md.

    Layout:
      sample_dir/c2s.json                      (plan-independent)
      sample_dir/<plan_slug>/task.json         (plan-specific)
      sample_dir/<plan_slug>/flowsom_<slug>/cells.md  (output)
    """
    plan_dir = sample_dir / plan_slug
    task_path = plan_dir / "task.json"
    c2s_path = sample_dir / "c2s.json"
    if not task_path.exists():
        print(f"[SKIP] {sample_dir.name} [{plan_slug}]: no task.json (run preprocess first)")
        return None
    if not c2s_path.exists():
        print(f"[SKIP] {sample_dir.name}: no c2s.json (run c2s clustering first)")
        return None
    with open(task_path) as f:
        task = json.load(f)
    with open(c2s_path) as f:
        c2s = json.load(f)

    text = format_prompt(task, c2s, use_top_hvp=use_top_hvp, top_hvp_n=top_hvp_n)
    slug = ablation_slug(use_top_hvp, top_hvp_n)
    out_dir = ablation_dir(plan_dir, slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cells.md"
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def _resolve_plan_slug(plan_slug_arg: str | None, gating_plan: Path | None) -> str:
    from src.flat_eval import gating_plan_slug as _slug
    if plan_slug_arg:
        return plan_slug_arg
    if gating_plan is not None:
        return _slug(gating_plan)
    return "default"


def main() -> None:
    parser = argparse.ArgumentParser(description="flowsom_c2s prompt renderer")
    parser.add_argument("--batch", type=Path, required=True,
                        help="Sample dir: benchmark_flat/<dataset>/<sample>/")
    parser.add_argument("--gating-plan", type=Path, default=None,
                        help="Used to derive plan-slug subdir (e.g. L1_coarse)")
    parser.add_argument("--plan-slug", type=str, default=None,
                        help="Override plan slug (default: auto from --gating-plan, else 'default')")
    parser.add_argument("--no-hvp", action="store_true")
    parser.add_argument("--top-hvp-n", type=int, default=None)
    args = parser.parse_args()

    plan_slug = _resolve_plan_slug(args.plan_slug, args.gating_plan)
    use_top_hvp = not args.no_hvp
    out = render_for_sample(
        args.batch, plan_slug=plan_slug,
        use_top_hvp=use_top_hvp, top_hvp_n=args.top_hvp_n,
    )
    if out:
        slug = ablation_slug(use_top_hvp, args.top_hvp_n)
        print(f"  {args.batch.name}/{plan_slug} [{slug}] -> {out}")


if __name__ == "__main__":
    main()
