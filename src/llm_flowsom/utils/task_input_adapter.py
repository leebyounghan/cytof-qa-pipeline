"""Adapters that expose a ``TaskInput`` (from ``src.bench``) in the shape the
LLM-side c2s pipeline expects.

``_TaskInputDataset`` mirrors the legacy ``ParquetDataset`` API but is
keyed by *channel* names (collision-safe under bead-style duplicates) and
exposes *display* (marker) names so downstream prompts and plots stay
human-readable.

``_step_from_task`` reshapes a ``TaskInput`` into the gating-plan-style
``step`` dict the c2s generator consumes.

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
    """Adapter exposing the ParquetDataset API over a channel-space TaskInput.

    TaskInput provides a parent-masked, perturbation-applied df keyed by
    **channel names** plus `channel_marker_map` / `marker_kinds` from meta.
    This adapter indexes by channel (for correctness under bead-style
    collisions) but exposes marker **display names** for protein list / zone
    tags / cell sentence so downstream text/plots stay human-readable.

    Duplicate display names (e.g., two protein channels mapping to the same
    marker — rare) are disambiguated by falling back to the channel name.
    """

    def __init__(self, ti: TaskInput):
        self._ti = ti
        self._kinds = ti.marker_kinds or {}
        self._ch_to_marker = ti.channel_marker_map or {}
        df = ti.expression
        self.feat_cols = list(df.columns)  # channels
        self.X = df.values.astype(np.float32)

        # Protein channels → display names (collision-safe).
        protein_idx: list[int] = []
        protein_channels: list[str] = []
        for i, ch in enumerate(self.feat_cols):
            if self._kinds.get(ch) == 'protein':
                protein_idx.append(i)
                protein_channels.append(ch)

        raw_display = [self._ch_to_marker.get(ch, ch) for ch in protein_channels]
        seen: dict[str, int] = {}
        protein_names: list[str] = []
        for ch, name in zip(protein_channels, raw_display):
            seen[name] = seen.get(name, 0) + 1
        for ch, name in zip(protein_channels, raw_display):
            protein_names.append(ch if seen[name] > 1 else name)

        self._protein_idx = protein_idx
        self.protein_names = protein_names
        self.cofactor = ti.cofactor

    def get_marker_type(self, channel: str) -> str:
        return self._kinds.get(channel, 'unknown')

    def get_feat_vals(self, channel: str) -> np.ndarray:
        return self.X[:, self.feat_cols.index(channel)]

    def get_parent_mask(self, step: dict) -> np.ndarray:
        return np.ones(len(self.X), dtype=bool)

    def get_xy_vals(self, step: dict):
        xi = self.feat_cols.index(step['x_marker'])
        yi = self.feat_cols.index(step['y_marker'])
        return self.X[:, xi], self.X[:, yi]

    def get_protein_expr(self):
        if not self._protein_idx:
            return np.empty((len(self.X), 0), dtype=np.float32), []
        return self.X[:, self._protein_idx], self.protein_names

    def get_gt_labels(self, step: dict) -> np.ndarray:
        return self._ti.y

    def get_display_name(self, channel: str) -> str:
        return self._ch_to_marker.get(channel, channel)


def _step_from_task(ti: TaskInput) -> dict:
    """Build a gating_plan-style step dict from a TaskInput (in marker-space)."""
    cats = [c for c in ti.categories if c != 'Unassigned']
    return {
        'step': ti.step,
        'parent': ti.parent,
        'x_marker': ti.x_marker,
        'y_marker': ti.y_marker,
        'annotation column name': ti.task['annotation_column'],
        'annotation categories': cats,
        'note': ti.task.get('note', ''),
        'tips': ti.task.get('tips', ''),
    }
