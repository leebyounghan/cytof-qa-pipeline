"""Adapters that expose a ``TaskInput`` (from ``src.bench``) in the shape
the LLM-gate pipeline expects — independent of ``src.llm``.

``_TaskInputDataset`` is keyed by *channel* names (collision-safe under
bead-style duplicates) and exposes *display* (marker) names so prompts
and plots stay human-readable.

``_step_from_task`` reshapes a ``TaskInput`` into the gating-plan-style
``step`` dict the LLM-gate generator consumes.

``_add_unassigned`` ensures the option list shown to the LLM always
includes the ``Unassigned`` fallback category.
"""

from __future__ import annotations

import numpy as np

from src.bench import TaskInput


def _add_unassigned(categories: list) -> list:
    cats = list(categories)
    if 'Unassigned' not in cats:
        cats.append('Unassigned')
    return cats


class _TaskInputDataset:
    """Adapter exposing a minimal dataset API over a channel-space TaskInput.

    Indexes by channel for correctness under bead-style collisions; surfaces
    marker display names for downstream text/plots.
    """

    def __init__(self, ti: TaskInput):
        self._ti = ti
        self._kinds = ti.marker_kinds or {}
        self._ch_to_marker = ti.channel_marker_map or {}
        df = ti.expression
        self.feat_cols = list(df.columns)
        self.X = df.values.astype(np.float32)
        self.cofactor = ti.cofactor

    def get_parent_mask(self, step: dict) -> np.ndarray:
        return np.ones(len(self.X), dtype=bool)

    def get_xy_vals(self, step: dict):
        xi = self.feat_cols.index(step['x_marker'])
        yi = self.feat_cols.index(step['y_marker'])
        return self.X[:, xi], self.X[:, yi]

    def get_gt_labels(self, step: dict) -> np.ndarray:
        return self._ti.y

    def get_display_name(self, channel: str) -> str:
        return self._ch_to_marker.get(channel, channel)


def _step_from_task(ti: TaskInput) -> dict:
    """Build a gating_plan-style step dict from a TaskInput."""
    cats = [c for c in ti.categories if c != 'Unassigned']
    return {
        'step':                    ti.step,
        'parent':                  ti.parent,
        'x_marker':                ti.x_marker,
        'y_marker':                ti.y_marker,
        'annotation column name': ti.task['annotation_column'],
        'annotation categories':  cats,
        'note':                   ti.task.get('note', ''),
        'tips':                   ti.task.get('tips', ''),
    }
