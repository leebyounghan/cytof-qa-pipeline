"""Parse the LLM's whole-cell JSON response and assemble pred.json.

The LLM picks a SET of leaves per cluster (multi-label), drawn from the
shared leaf vocabulary (``src.flat_eval.derive_leaf_set``) plus the
sentinel ``"Discard"``.
"""

from __future__ import annotations

import json
import re

DISCARD = "Discard"


_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_cluster_answers(
    text: str, leaf_categories: set[str],
) -> tuple[dict, int]:
    """Parse the LLM JSON object keyed by cluster_<N>.

    Returns (parsed, n_invalid).
      parsed[cluster_id] = {"leaves": list[str], "rationale": str|None}
      n_invalid          = clusters whose JSON was malformed or whose leaves
                           list contained no valid entries (yielding []).
    """
    obj = None
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        m2 = _BARE_OBJ_RE.search(text)
        if m2:
            try:
                obj = json.loads(m2.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return {}, -1

    valid_set = set(leaf_categories) | {DISCARD}
    out: dict = {}
    n_invalid = 0
    for k, v in obj.items():
        if not isinstance(v, dict):
            out[k] = {"leaves": [], "rationale": None}
            n_invalid += 1
            continue
        raw = v.get("leaves")
        if isinstance(raw, str):
            raw = [raw]  # tolerate ungrammatical singleton
        if not isinstance(raw, list):
            out[k] = {"leaves": [], "rationale": v.get("rationale") if isinstance(v.get("rationale"), str) else None}
            n_invalid += 1
            continue
        cleaned: list[str] = []
        for item in raw:
            if isinstance(item, str) and item in valid_set:
                if item not in cleaned:  # dedupe within cluster
                    cleaned.append(item)
        rationale = v.get("rationale")
        out[k] = {
            "leaves": cleaned,
            "rationale": rationale if isinstance(rationale, str) else None,
        }
        if not cleaned:
            n_invalid += 1
    return out, n_invalid


def assemble_pred(
    model: str,
    ablation: str,
    benchmark_dir: str,
    per_sample: dict,
    perturbation: str | None = None,
) -> dict:
    n_samples = len(per_sample)
    n_parse_fail_total = sum(s.get("n_invalid", 0) for s in per_sample.values() if s.get("n_invalid", 0) > 0)
    payload = {
        "meta": {
            "model": model,
            "ablation": ablation,
            "benchmark": benchmark_dir,
            "n_samples": n_samples,
            "n_parse_fail_clusters": n_parse_fail_total,
        },
        "samples": {
            name: {"clusters": s["clusters"]}
            for name, s in per_sample.items()
        },
    }
    if perturbation is not None:
        payload["meta"]["perturbation"] = perturbation
    return payload
