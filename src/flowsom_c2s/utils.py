"""Helpers for the flowsom_c2s whole-cell pipeline.

- ``compute_top_hvp`` — fork of ``src.llm.utils.features.compute_top_hvp``
  so this package has no dependency on ``src/llm/``.
- ``build_leaf_to_step_col`` — given the gating plan's ``steps`` list and
  the leaf-vocabulary set (computed via ``src.flat_eval.derive_leaf_set``),
  return a mapping ``{leaf_name -> annotation_column}``. Used by postprocess
  to know which Step* column each predicted leaf populates.

Leaf vocabulary is **shared** with ``src/llm_gate_flat/`` via
``src.flat_eval.derive_leaf_set`` — both pipelines treat a leaf as a
category that never appears in any later step's parent expression.
"""

from __future__ import annotations

import numpy as np


# ── Top HVP (forked from src/llm/utils/features.py) ─────────────────────────

def compute_top_hvp(
    X: np.ndarray,
    marker_names: list[str],
    top_n: int | None = None,
    exclude: tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    """Marker names ranked by variance over all cells in X (high → low)."""
    if X.size == 0 or not marker_names:
        return []
    excluded = set(exclude) if exclude else set()
    variances = np.var(X, axis=0)
    order = np.argsort(variances)[::-1]
    ranked = [marker_names[i] for i in order if marker_names[i] not in excluded]
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


# ── Leaf -> step annotation_column mapping ──────────────────────────────────

def build_leaf_to_step_col(steps: list[dict], leaf_set: set[str]) -> dict[str, str]:
    """Return ``{leaf_name -> annotation_column}``.

    Each leaf must appear in exactly one step's ``annotation categories``;
    that step's ``annotation column name`` is the parquet column we'd
    populate with the leaf to make ``src.flat_eval`` pick it up.

    Raises ValueError if a leaf is missing from every step (gating plan
    inconsistent) or appears in more than one step (ambiguous).
    """
    found: dict[str, str] = {}
    for step in steps:
        col = step["annotation column name"]
        for cat in step.get("annotation categories", []) or []:
            if cat in leaf_set:
                if cat in found and found[cat] != col:
                    raise ValueError(
                        f"Leaf {cat!r} appears in two steps' annotation categories: "
                        f"{found[cat]!r} and {col!r}"
                    )
                found[cat] = col
    missing = leaf_set - set(found.keys())
    if missing:
        raise ValueError(
            f"Leaves not found in any step's annotation categories: {sorted(missing)}"
        )
    return found
