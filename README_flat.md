# Flat-Annotation Experiments

Two methods, same task, same evaluator — meant to be compared head-to-head.

| Method | Module | Approach |
|---|---|---|
| **LLM-Gate (cascading)** | `src/llm_gate_flat/` | Walk `gating_plan.json` step-by-step per sample. At step *k* the parent population comes from the LLM's own predictions for steps `1..k-1` — errors compound. |
| **FlowSom + LLM (whole-cell)** | `src/flowsom_c2s/` | Cluster the whole sample once with FlowSom on all protein channels. The LLM then labels each cluster with a leaf set in a single shot — no cascade, no compounding. |

Both methods emit `predicted_steps.parquet` with the same Step* schema as
the GT parquet, so the **shared evaluator** `src/flat_eval.py` scores them
identically and the metrics are directly comparable.

## Task

For each cell in a sample, the gating plan ultimately assigns a SET of leaf
cell-type labels (multi-label — a single CD4 T cell can carry both a memory
leaf like `CD4TCM` and an activation leaf like `CD4TCM/activated` from
parallel-axis steps).

The leaf vocabulary is **derived from the gating plan**:
> A category is a leaf iff it never appears as a parent in any later step.

Way-points (`Cleanup1..6`, `Mononuclear`, `CD45`, `abT`, ...) are filtered
out — only terminal classes count. Cells that never reach a leaf
(rejected at some cleanup step, or where the cascade aborted) collapse
to the synthetic class `Discard`.

## Plan depths (L1 / L2 / L3)

Most cohorts ship three plan variants of increasing depth:

```
benchmark_flat/<dataset>/
  gating_plan_L1_coarse.json     # major lineages only (e.g. 16 leaves for Acute2020)
  gating_plan_L2_identity.json   # +identity-level subsetting
  gating_plan.json               # L3 — full canonical plan (e.g. 62 steps for Acute2020)
```

The plan slug is auto-derived from the filename and propagates through
both pipelines so L1 / L2 / L3 outputs land in distinct subdirectories:

| Filename | Slug |
|---|---|
| `gating_plan.json` | `default` |
| `gating_plan_L1_coarse.json` | `L1_coarse` |
| `gating_plan_L2_identity.json` | `L2_identity` |

The same `--gating-plan` MUST be passed to every stage of a run (preprocess,
prompt, inference, postprocess, eval) so all stages key on the matching slug.

**`benchmark_flat/<dataset>/`** is the canonical home for gating plans —
copy the `data/<dataset>/gating_plan*.json` files there once, then point
all flat tools to them.

## Directory layout

```
benchmark_flat/<dataset>/
  gating_plan*.json                       # canonical plan home
  <sample>/
    c2s.json, cell2cluster.npz            # FlowSom output (plan-independent)
    c2s_scatter.png                       # diagnostic
    <plan_slug>/
      task.json                           # leaf vocabulary + leaf->step_col map
      flowsom_<ablation>/cells.md         # LLM prompt (full / hvp10 / nohvp)

results_flat/
  flowsom_<run_name>/<plan_slug>/<dataset>/
    pred.json                             # cluster-level predictions
    <sample>/predicted_steps.parquet      # cell-level Step* table (eval input)
    <sample>/eval_flat.json               # per-sample metrics (after eval)
    <sample>/confusion_matrix.png
    <sample>/per_leaf_metrics.png
    eval_summary.json                     # cross-sample aggregate (after eval)

  llm_gate_flat_<run_name>/<plan_slug>/<dataset>/
    <sample>/step_<NN>/prediction.json    # per-step LLM call output + parsed gates
    <sample>/predicted_steps.parquet      # cell-level Step* table (eval input)
    <sample>/cascade_meta.json
    <sample>/cascade_summary.png
    eval_summary.json
```

Both methods write under `results_flat/<method>/<plan_slug>/...` so all
three depths can co-exist for the same method on the same dataset.

## Pipeline — `src/llm_gate_flat/`

Sequential cascade — one LLM call per (sample, step):

```bash
DATASET=Acute2020
PLAN=benchmark_flat/$DATASET/gating_plan_L1_coarse.json   # or _L2_identity / .json (L3)

python -m src.llm_gate_flat.runner \
    --benchmark   benchmark/ --data-dir data/ \
    --datasets    $DATASET \
    --gating-plan $PLAN \
    --output-dir  results_flat/llm_gate_flat/ \
    --model       gpt-5.4 \
    --tiebreak    smallest \
    --split       benchmark/splits.json --split-set test
# auto-writes to: results_flat/llm_gate_flat/<plan_slug>/<dataset>/<sample>/...
```

Per (sample, step):
- Build the parent mask from prior LLM predictions.
- If empty → abort cascade for this sample (recorded in `cascade_meta.json`).
- Render biaxial-plot prompt → LLM → parse rectangle/polygon gates.
- Apply gates to assign per-cell labels at this step.

## Pipeline — `src/flowsom_c2s/`

Whole-cell — one LLM call per sample:

```bash
DATASET=Acute2020
PLAN=benchmark_flat/$DATASET/gating_plan_L1_coarse.json

# 1) Per-sample task.json (leaf vocabulary, leaf->step_col map)
python -m src.flowsom_c2s.preprocess \
    --data-dir data/ --output-dir benchmark_flat/ \
    --datasets $DATASET --gating-plan $PLAN

# 2) FlowSom clustering (plan-independent — shared across L1/L2/L3)
python -m src.flowsom_c2s.c2s \
    --benchmark benchmark_flat/ --data-dir data/ \
    --datasets $DATASET --n-meta-clusters 60

# 3) Render per-sample prompt
for s in benchmark_flat/$DATASET/*/; do
    python -m src.flowsom_c2s.c2s_prompt --batch "$s" \
        --gating-plan $PLAN --top-hvp-n 10
done

# 4) Inference — one LLM call per sample
python -m src.flowsom_c2s.inference.run_openai \
    --benchmark    benchmark_flat/$DATASET \
    --gating-plan  $PLAN \
    --model        gpt-5.4 --ablation_slug hvp10 \
    --output_path  results_flat/flowsom_gpt5_hvp10/L1_coarse/$DATASET/pred.json

# 5) Decompose cluster leaves -> predicted_steps.parquet (auto-appends plan_slug)
python -m src.flowsom_c2s.postprocess \
    --pred           results_flat/flowsom_gpt5_hvp10/L1_coarse/$DATASET/pred.json \
    --benchmark-flat benchmark_flat/ \
    --gating-plan    $PLAN \
    --output-dir     results_flat/flowsom_gpt5_hvp10/
```

Pipeline detail:
- **FlowSom**: SOM grid (10×10 default) trained on protein channels, then
  Ward hierarchical clustering on the SOM codebook → K_meta meta-clusters
  (default 40, raised for deeper plans). Implementation uses `minisom` +
  `scipy.cluster.hierarchy` because the saeyslab `flowsom` PyPI package
  pulls in `numba`/`llvmlite`, which fails to build on common macOS
  setups.
- **C2S prompt**: per-cluster `Marker(value)` cell sentence ordered by
  descending expression, top-N HVP markers, `[Leaf categories]` flat list.
  The Task instructions live in the system prompt
  (`src/flowsom_c2s/inference/system_prompts.py`).
- **Multi-label**: the LLM may pick multiple leaves per cluster when the
  plan has parallel axes. Each picked leaf populates a different Step*
  column in `predicted_steps.parquet`.

## Evaluation — shared `src.flat_eval`

Same call regardless of method:

```bash
python -m src.flat_eval \
    --output-dir  results_flat/<method>/<plan_slug>/ \
    --gating-plan $PLAN \
    --data-dir    data/ --benchmark benchmark/ \
    --datasets    $DATASET
```

For every cell, both sides walk the gating plan's Step* columns and
reduce to a multi-label leaf set (∈ leaves ∪ {Discard}).

### Metrics

**Multi-label** (primary, computed via `MultiLabelBinarizer`):
- `subset_accuracy`, `hamming_loss`
- `f1_micro`, `f1_macro`, `f1_weighted`
- `jaccard_micro`, `jaccard_macro`
- per-leaf precision / recall / F1 / support

**Primary-leaf** (single-label view for intuition + heatmap):
- the deepest leaf in each cell's path, or `Discard` if none
- `accuracy`, `balanced_accuracy`, `f1_macro`, `f1_weighted`, NxN
  confusion matrix

### Per-sample artifacts

- `eval_flat.json` — all metrics, cell counts (`n_cells_gt_discard`,
  `n_cells_pred_discard`, `n_cells_multi_gt`, `n_cells_multi_pred`),
  per-leaf stats, raw confusion matrix.
- `confusion_matrix.png` — row-normalised heatmap (primary-leaf view).
- `per_leaf_metrics.png` — bar chart of per-leaf precision / recall / F1
  over a log-scale GT-support row.

`eval_summary.json` at the `--output-dir` level aggregates means across
samples for both views.

## Sanity baselines

We use **GT-oracle** majority-vote pred.json files to bound each pipeline:

- For each FlowSom cluster, the oracle picks the leaves that ≥50% of the
  cluster's cells carry in the GT.
- This is the ceiling that the FlowSom + LLM pipeline could possibly hit
  if the LLM's leaf assignment were perfect.

Ceiling readings on Acute2020 / 994570_Normalized / L1_coarse / K=60:
| Metric | Oracle ceiling |
|---|---|
| multi-label F1_micro | 0.868 |
| multi-label F1_macro | 0.429 |
| primary-leaf accuracy | 0.864 |

The macro-vs-micro gap reveals which leaves are too rare to land in any
single FlowSom cluster at this K — bumping K_meta is the lever.

## Comparing the two methods

To compare LLM-Gate-Flat vs FlowSom+LLM at a fixed depth:

1. Run both methods on the **same** `--gating-plan`, `--datasets`,
   `--samples` (test split is the convention).
2. `src.flat_eval` on each method's output dir.
3. Compare `eval_summary.json` aggregates side-by-side.

Per-(scenario × depth) tradeoffs to expect:
- **L1 (coarse)** — both methods should hit primary-leaf accuracy in the
  0.85+ range. Cascading rarely aborts; clustering finds the major
  lineages cleanly.
- **L2 / L3 (deep)** — LLM-Gate-Flat starts losing samples to cascade
  abort at deep cleanup steps; FlowSom + LLM is robust to that but its
  K_meta becomes a bottleneck for rare leaves. Multi-label F1_macro is
  the metric that separates them most.

## Plan-slug naming conventions

When you fork a method (e.g. different K_meta, different model, different
ablation), encode the variation in the *method* segment, not the plan
segment:

```
results_flat/flowsom_gpt5_hvp10_K60/L1_coarse/<dataset>/...
results_flat/flowsom_gpt5_hvp10_K120/L1_coarse/<dataset>/...
results_flat/llm_gate_flat_gpt5/L1_coarse/<dataset>/...
results_flat/llm_gate_flat_haiku/L1_coarse/<dataset>/...
```

Plan slugs (`L1_coarse` / `L2_identity` / `default`) are reserved for the
gating plan filename — keep that axis clean for direct depth-vs-depth
comparison.
