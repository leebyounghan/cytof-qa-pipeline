"""Per-step pipeline for a sequential cascading gating walk.

For one ``gating_plan.json`` step, given the cascade state up to here:

  1. Resolve x/y channels against the parquet column space.
  2. Compute the predicted-parent mask over the sample (cascade-aware).
  3. Build the LLM task dict via ``src.llm_gate.task.generate_gate_task``.
  4. Render the prompt via ``src.llm_gate.prompt.format_prompt``.
  5. Call the LLM, parse gates, propagate to per-cell labels.
  6. Update the cascade state with the new Step column (length = N_full,
     ``Unassigned`` outside the predicted parent).

Returns ``None`` when the predicted parent is empty — the caller treats
this as an abort signal for the sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bench import BenchmarkLoader
from src.llm_gate.inference.output_schema import extract_gates, is_absent_gate
from src.llm_gate.postprocess.propagate import _gate_contains, _ordered_gates
from src.llm_gate.prompt import format_prompt
from src.llm_gate.task import generate_gate_task

from .cascade_state import CascadeState


@dataclass
class StepResult:
    annotation_col: str
    predicted_subset: list           # length == n_parent
    n_parent: int
    info: dict
    raw_response: str
    prompt_text: str
    gates: dict                      # {category: gate-dict | list | None}
    options: list                    # categories incl. "Unassigned"
    x_arr: np.ndarray                # parent-subset x values, len == n_parent
    y_arr: np.ndarray                # parent-subset y values, len == n_parent
    x_marker_display: str
    y_marker_display: str
    parent_expr: str
    step_num: int


class _CascadeDataset:
    """Adapter exposing the (parent_mask, x_vals, y_vals, gt) shape that
    ``generate_gate_task`` expects, where the parent mask comes from the
    cascade's own Step predictions rather than GT."""

    def __init__(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        parent_mask: np.ndarray,
        gt_labels: np.ndarray | None,
        channel_marker_map: dict,
    ):
        self._x = x_vals
        self._y = y_vals
        self._mask = parent_mask
        self._gt = gt_labels
        self._cmap = channel_marker_map or {}

    def get_parent_mask(self, step):
        return self._mask

    def get_xy_vals(self, step):
        return self._x, self._y

    def get_gt_labels(self, step):
        return self._gt

    def get_display_name(self, channel):
        return self._cmap.get(channel, channel)


def _resolve_channel(channel_marker_map: dict, name: str, df_cols: list[str]) -> str | None:
    if name in df_cols:
        return name
    for ch, marker in channel_marker_map.items():
        if marker == name and ch in df_cols:
            return ch
    return None


def _propagate_gates(
    gates: dict,
    options: list[str],
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    tiebreak: str,
) -> tuple[list[str], dict]:
    n = int(len(x_arr))
    targets = [c for c in options if c != "Unassigned"]
    ordered = _ordered_gates(gates, options, tiebreak)
    cats_with_gate = sorted({c for c, _ in ordered})
    cats_absent = sorted(c for c in targets if is_absent_gate(gates.get(c)))

    info = {
        "method": f"gate_rect_{tiebreak}",
        "n_parent_cells": n,
        "n_gates_total": len(targets),
        "n_gates_valid": len(cats_with_gate),
        "n_gates_absent": len(cats_absent),
        "n_rectangles": len(ordered),
        "categories_with_gate": cats_with_gate,
        "categories_absent": cats_absent,
        "tiebreak": tiebreak,
    }

    pred = np.array(["Unassigned"] * n, dtype=object)
    if n == 0 or not ordered:
        return pred.tolist(), info

    for cat, g in ordered:
        inside = _gate_contains(g, x_arr, y_arr)
        if inside.any():
            pred[inside] = cat
    return pred.tolist(), info


def execute_step(
    state: CascadeState,
    step: dict,
    *,
    channel_marker_map: dict,
    cofactor: float | None,
    modality: str,
    cohort: str,
    llm_call,                # callable(system_prompt, user_prompt) -> str
    system_prompt: str,
    tiebreak: str,
) -> StepResult | None:
    feature_cols = [c for c in state._df_full.columns if not c.startswith("Step")]  # noqa: SLF001
    x_ch = _resolve_channel(channel_marker_map, step["x_marker"], feature_cols)
    y_ch = _resolve_channel(channel_marker_map, step["y_marker"], feature_cols)
    if x_ch is None or y_ch is None:
        raise RuntimeError(
            f"Step {step['step']}: cannot resolve markers "
            f"x={step['x_marker']!r}, y={step['y_marker']!r} against parquet columns"
        )

    syn = state.synthetic_df()
    parent_mask = BenchmarkLoader._parent_mask(syn, step.get("parent", "ALL"))
    n_parent = int(parent_mask.sum())
    if n_parent == 0:
        return None

    annotation_col = step["annotation column name"]
    gt_labels = state.gt_labels(annotation_col)

    adapter = _CascadeDataset(
        x_vals=state.feature_values(x_ch),
        y_vals=state.feature_values(y_ch),
        parent_mask=parent_mask,
        gt_labels=gt_labels,
        channel_marker_map=channel_marker_map,
    )

    step_for_gen = dict(step)
    step_for_gen["x_marker"] = x_ch
    step_for_gen["y_marker"] = y_ch

    task_dict = generate_gate_task(
        adapter, step_for_gen,
        modality=modality, cofactor=cofactor, cohort=cohort,
    )
    if task_dict is None:
        return None

    prompt_text = format_prompt(
        task_dict, use_note=True, use_joint_2d=False, use_notes_yaml=True,
    )

    raw_text = llm_call(system_prompt, prompt_text)

    gates = extract_gates(raw_text, task_dict["options"])

    parent_idx = np.where(parent_mask)[0]
    x_arr = state.feature_values(x_ch)[parent_idx]
    y_arr = state.feature_values(y_ch)[parent_idx]
    pred_subset, info = _propagate_gates(
        gates, task_dict["options"], x_arr, y_arr, tiebreak,
    )

    info["llm_model"] = getattr(llm_call, "model", None)
    info["x_channel"] = x_ch
    info["y_channel"] = y_ch
    info["raw_response_chars"] = len(raw_text)

    full_col = np.array(["Unassigned"] * state.n, dtype=object)
    full_col[parent_idx] = pred_subset
    state.set(annotation_col, full_col)

    return StepResult(
        annotation_col=annotation_col,
        predicted_subset=pred_subset,
        n_parent=n_parent,
        info=info,
        raw_response=raw_text,
        prompt_text=prompt_text,
        gates=gates,
        options=list(task_dict["options"]),
        x_arr=x_arr,
        y_arr=y_arr,
        x_marker_display=task_dict["x_marker"],
        y_marker_display=task_dict["y_marker"],
        parent_expr=step.get("parent", "ALL"),
        step_num=int(step["step"]),
    )
