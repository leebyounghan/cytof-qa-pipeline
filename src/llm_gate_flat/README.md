# `src/llm_gate_flat/` — Sequential Cascading LLM-Gate

Walks `gating_plan.json` step by step **per sample**, where step *k*'s parent
population is computed from the LLM's own predictions for steps `1..k-1`
(not from the GT `Step*` columns in the parquet). Errors compound — this
mirrors a real-world annotation pipeline.

Contrast with `src/llm_gate/`: that module loads each step's parent
**from GT** and runs every (sample, step) prompt independently in parallel.
Here, every sample is its own sequential cascade.

## Output

```
{output_dir}/{dataset}/{sample}/
  step_01/
    prediction.json           # length = predicted-parent at that step; includes parsed gates
    plot.png                  # 2-D scatter + LLM gate overlay
  step_02/
    ...
  cascade_meta.json           # aborted, abort_step, n_steps_completed, model, ...
  cascade_summary.png         # all-steps grid for the sample
  predicted_steps.parquet     # full-N table of all predicted Step columns
  eval_flat.json              # (after eval) flat-label metrics + per-class
  confusion_matrix.png        # (after eval) flat-label confusion matrix heatmap
{output_dir}/eval_summary.json  # cross-sample aggregate metrics
```

`prediction.json` schema mirrors `TaskInput.save_prediction`, with a
`predicted_labels` length equal to the **predicted-parent** cell count
(not GT-parent), plus a `gates` field containing the parsed LLM gate
dict (rectangles or polygons). Final evaluation runs end-to-end on the
flat per-cell terminal label derived from `predicted_steps.parquet`.

## Behavior

- **Empty predicted parent at step *k***: abort the cascade for that
  sample. `cascade_meta.json` records `abort_step` and reason.
- **Sample order**: sequential.
- **Step order within a sample**: sequential (must be — that's the cascade).
- **Reused machinery**: `BenchmarkLoader._parent_mask`, `generate_gate_task`,
  `format_prompt`, `extract_gates`, `_ordered_gates`, `_gate_contains`,
  `get_system_prompt`. The cascade-specific layer is just `CascadeState`
  (synthetic-df builder) and `step_executor.execute_step`.

## CLI

```bash
export OPENAI_API_KEY=sk-...
python -m src.llm_gate_flat.runner \
    --benchmark   benchmark/ \
    --data-dir    data/ \
    --datasets    Acute2020 \
    --gating-plan data/Acute2020/gating_plan_L1_coarse.json \
    --output-dir  results_flat/llm_gate_flat/ \
    --model       gpt-5.4 \
    --tiebreak    smallest
```

`--gating-plan` accepts any of `gating_plan.json` (L3),
`gating_plan_L2_identity.json`, `gating_plan_L1_coarse.json`. The cascade
length and difficulty scale accordingly.

`--samples`, `--split`, `--perturbation`, `--hard-depletion`, `--hard-shift`
are inherited from `BenchmarkLoader.add_cli_args` and behave identically
to the rest of the pipeline.

Disable plotting with `--no-plot`; disable the per-sample full-N parquet
with `--no-save-cascade-parquet`.

## Evaluation (leaf-set ∪ Discard, multi-label)

`eval_flat.py` walks the **same** Step columns on both sides
(`predicted_steps.parquet` and the original GT parquet) and reduces each
cell to its **flat label set** over the vocabulary `leaves ∪ {Discard}`.

A *leaf* is automatically derived from the gating plan: any category
that never appears in any later step's parent expression. Non-leaf
categories (`Cleanup1..6`, `TotalCD45pos`, `Mononuclear`, `abT`, ...)
are intermediate way-points and do **not** count as flat labels — only
the terminal classes do (`CD4`, `CD8`, `Granulocyte`, `Bead`,
`TotalNK`, ...).

`Discard` is a synthetic class for cells that **never reach a leaf**
(rejected somewhere in the cleanup chain or the cascade aborted). It is
added because pred-rejected-at-step-1 and GT-rejected-at-step-5 are
functionally equivalent — both ended up *not* being a real cell type.
Only the binary outcome matters (reached a leaf vs not), not where in
the cleanup chain rejection happened. With `Discard` as a real class,
cleanup-agreement contributes positive TPs to F1 instead of being
neutral zero-vectors.

In plans with orthogonal axes (L2/L3 helper subtype × memory subtype) a
single cell can legitimately carry multiple leaf labels — multi-label
framework handles both cases uniformly.

Multi-label is the natural framing because in deeper plans (L2/L3) a
single cell can carry independent leaves on orthogonal axes — e.g., a
CD4 cell with both a helper subtype (Th1) and a memory subtype (TEM).
For L1 most cells produce a single-element set; the framework handles
both uniformly.

```bash
python -m src.flat_eval \
    --output-dir  results_flat/llm_gate_flat/ \
    --gating-plan data/Acute2020/gating_plan_L1_coarse.json \
    --data-dir    data/ \
    --benchmark   benchmark/
```

Pass the **same** `--gating-plan` that was used for the cascade so both
sides walk the same Step columns and induce the same leaf vocabulary.
`--datasets`/`--samples` filters work the same way as in `runner.py`.

### Metrics

**Multi-label** (primary view, computed via `MultiLabelBinarizer` over
the discovered leaf set):
- `subset_accuracy`, `hamming_loss`
- `f1_micro`, `f1_macro`, `f1_weighted`
- `jaccard_micro`, `jaccard_macro`
- per-leaf precision / recall / F1 / support

**Primary-leaf** (single-label view for intuition + heatmap): the
deepest leaf in each cell's path, or `Unassigned` when no leaf was hit.
Reports `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, and
the standard NxN confusion matrix.

### Per-sample artifacts

- `eval_flat.json` — full multi-label + primary-leaf metrics, cell
  counts (`n_cells_no_gt_leaf`, `n_cells_no_pred_leaf`,
  `n_cells_multi_gt`, `n_cells_multi_pred`), per-leaf stats, raw CM.
- `confusion_matrix.png` — row-normalised heatmap of the primary-leaf
  view, cell text shows raw counts.
- `per_leaf_metrics.png` — bar chart of multi-label
  precision/recall/F1 with a log-scale GT-support row underneath.

At the top of `--output-dir`, `eval_summary.json` aggregates means
across samples for both the multi-label and primary-leaf views.
