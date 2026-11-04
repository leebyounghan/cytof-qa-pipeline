"""Helpers for the flowsom_c2s whole-cell pipeline.

- ``compute_top_hvp`` — fork of ``src.llm.utils.features.compute_top_hvp``
  so this package has no dependency on ``src/llm/``.

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
