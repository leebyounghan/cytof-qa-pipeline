"""Feature-engineering helpers for the c2s two-stage clustering pipeline.

- ``compute_top_hvp`` — protein names ranked by variance over all parent
  cells (high → low), used to truncate the cell-sentence vocabulary.
"""

from __future__ import annotations

import numpy as np


# ── Top HVP (highly variable proteins, global over all parent cells) ─────────

def compute_top_hvp(
    X: np.ndarray,
    marker_names: list[str],
    top_n: int | None = None,
    exclude: tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    """Marker names ranked by variance over ALL cells in X (high → low).

    X: (n_cells, n_markers). marker_names must align with columns of X.
    If `top_n` is None, returns ALL markers in rank order; otherwise truncates.
    `exclude` removes the given marker names from the ranking *before*
    truncation, so `top_n` HVPs are still returned when excluded markers
    happened to be in the top ranks (assuming enough remain).
    """
    if X.size == 0 or not marker_names:
        return []
    excluded = set(exclude) if exclude else set()
    variances = np.var(X, axis=0)
    order = np.argsort(variances)[::-1]
    ranked = [marker_names[i] for i in order if marker_names[i] not in excluded]
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked
