# C2S LLM Pipeline (`src.llm`)

Cell-to-Sentence pipeline: each gating step's parent population is clustered on the (x, y) plane into K clusters, and each cluster's representative cell HVP profile becomes the "cell sentence" the LLM sees. Once the LLM assigns a label per cluster, `cell2cluster.npz` propagates it back to every parent cell at O(1) per cell.

```
parquet → c2s.json → c2s_{ablation}/cells.md → pred.json → prediction.json → eval_summary.json
   (loader)  (Step 1)        (Step 2)              (Step 3)      (Step 4)        (Step 5)
```

## Step 1 — `src.llm.c2s`: parquet → `c2s.json`

Single-stage clustering on the (x, y) plane:

1. **Cluster** parent cells in the step's `(x, y)` plane into K clusters. Recommended: `--cluster-method flowsom --flowsom-grid 10 --flowsom-k 20` (SOM on (x,y) → consensus metaclustering of the codebook). For deterministic kmeans-style alternatives use `--cluster-method {mbkm,agglo,gmm} --k 16` (these fit on a uniform `--subsample-n 10_000` subsample and propagate via 1-NN; flowsom skips subsampling — SOM cost is fixed in iterations).
2. **Rep cell** — per cluster, pick the parent cell that is 1-NN (Euclidean, full protein) to the cluster's protein centroid. Its `(x, y)`, HVP cell sentence (sorted high→low), and GT label are recorded in the c2s slot. The cluster mean is never surfaced.

```bash
# CyTOF (recommended: flowsom, grid=10, k=20)
python -m src.llm.c2s --benchmark benchmark/ --data-dir data/ \
    --modality cytof --datasets Acute2020 Acute2021 Vaccine \
    --cluster-method flowsom --flowsom-grid 10 --flowsom-k 20

# Flow (recommended: flowsom as well)
python -m src.llm.c2s --benchmark benchmark/ --data-dir data/ \
    --modality flow --datasets FR-FCM-Z74D_hc Lyoplate_tcell \
    --cluster-method flowsom --flowsom-grid 10 --flowsom-k 20

# Single-stage agglomerative (deterministic, faster on small parents)
python -m src.llm.c2s --benchmark benchmark/ --data-dir data/ \
    --datasets Acute2020 --cluster-method agglo --k 16

# Larger K with kmeans-style methods
python -m src.llm.c2s --benchmark benchmark/ --data-dir data/ \
    --datasets Acute2020 --cluster-method mbkm --k 36
```

**Key flags** (full list via `--help`):
- `--cluster-method {mbkm, agglo, gmm, flowsom}` — `flowsom` is the recommended default. Code default is `agglo` for backward compatibility; pass `--cluster-method flowsom` explicitly.
- `--flowsom-grid N` (default 10) — SOM grid size; total SOM neurons = `N²`.
- `--flowsom-k N` (default 20) — FlowSOM consensus metacluster count (final K). The SOM codebook is hierarchically clustered (average linkage) into `N` final clusters.
- `--k N` (default 16) — target cluster count for `mbkm` / `agglo` / `gmm`. Ignored for `flowsom` (use `--flowsom-k`).
- `--subsample-n N` (default 10_000)
- `--base-seed 42` — seeds the fit subsample + stochastic init. Identical inputs + same seed → bit-identical outputs (for the deterministic methods).
- `--n-workers N`, `--no-plots`, `--samples`, `--split` / `--split-set test`

`flowsom` is lazy-imported. Install on first use:

```bash
uv add minisom       # for --cluster-method flowsom
```

### `c2s.json` structure

```json
{
  "step": 9,
  "x_marker": "CD66b", "y_marker": "CD45",
  "parent": "Step08_... == CD45",
  "options": ["Mononuclear", "Granulocyte", "...", "Unassigned"],
  "n_parent_cells": 18007,
  "top_hvp_markers": ["CD14", "CD38", "..."],
  "n_clusters": 20,
  "cluster_method": "flowsom",
  "k": 16, "k_fit": 16, "subsample_n": 18007, "base_seed": 42,
  "flowsom_grid": 10, "flowsom_k": 20,
  "modality": "cytof", "cofactor": 5.0,
  "cells": {
    "1": {"cluster_id": 0, "n_cells": 4356, "x": 0.78, "y": 3.11,
          "cell_index": 699, "parquet_row": 9305,
          "cell_sentence": "CD45RA(4.13) > CD3(4.03) > CD4(3.80) > ..."},
    "...": "..."
  },
  "gt_answers": {"699": "T cell CD4 Naive", "...": "..."}
}
```

For `cluster_method=flowsom`, extra `flowsom_grid` and `flowsom_k` fields are recorded (as shown above). `n_clusters` may be less than `flowsom_k` if some metaclusters end up empty.

### `cell2cluster.npz`

Three int32 arrays aligned with parent cells:
- `parent_indices` (n_parent,) — parquet row per parent cell.
- `cluster_labels` (n_parent,) — cluster slot (`0..K-1`). Propagation is `slot_to_label[cluster_labels[i]]`.
- `rep_parent_local` (K,) — parent-local index of each slot's rep cell (for plots / sim graphs).

### Artifacts

- `c2s.json`, `cell2cluster.npz`
- `c2s_scatter.png` — 2 panels: GT · clusters with rep cells overlaid as white stars

Under `--perturbation`, all artifacts are suffixed: `c2s__{slug}.json` / `cell2cluster__{slug}.npz` / `c2s__{slug}_scatter.png`.

## Step 2 — `src.llm.c2s_prompt`: `c2s.json` → `c2s_{ablation}/cells.md`

Renders one cell-level prompt per step into a per-ablation sibling directory. `--n-cells N` picks the first N clusters from `c2s.json`; omit to prompt every cluster.

```bash
# Default (full): c2s_full/cells.md
python -m src.llm.c2s_prompt --batch benchmark/Acute2020/994567_Normalized --save_md

# HVP truncated: c2s_hvp10/cells.md
python -m src.llm.c2s_prompt --batch benchmark/Acute2020/994567_Normalized \
    --save_md --top-hvp-n 10

# Drop HVP entirely: c2s_nohvp/cells.md
python -m src.llm.c2s_prompt --batch benchmark/Acute2020/994567_Normalized \
    --save_md --no-top-hvp

# All ablation variants at once
python -m src.llm.c2s_prompt --batch benchmark/Acute2020/994567_Normalized --ablation-all
```

**Ablation slugs**:
- *(none)* → `c2s_full/`
- `--no-top-hvp` → `c2s_nohvp/` — drop the `[Top HVP]` section + strip HVP tokens from each cell sentence (axis values only)
- `--top-hvp-n N` → `c2s_hvp{N}/` — truncate HVP list + per-cell tokens to the first N markers
- `--n-cells N` — orthogonal to ablation. Use only the first N clusters.

The `[Context]` block opens with `Modality: {CyTOF | Flow Cytometry} (arcsinh cofactor: N)` so the LLM knows which modality and cofactor produced the values (read from `c2s.json`'s `modality` / `cofactor`).

Batch across a cohort:
```bash
for s in benchmark/Acute2020/*/; do
    python -m src.llm.c2s_prompt --batch "$s" --save_md --top-hvp-n 10
done
```

## Step 3 — `src.llm.inference.run_openai`: prompts → `pred.json`

Sends prompts to the model and writes raw cluster-level predictions. **No metrics, no plots, no propagation.**

```bash
# OPENAI_API_KEY auto-loaded from ./.env (python-dotenv)
python -m src.llm.inference.run_openai \
    --dataset_path   benchmark/Acute2020 \
    --model          gpt-5.4 \
    --ablation_slug  hvp10 \
    --concurrency    20 \
    --output_path    results/gpt-5.4_hvp10/Acute2020/pred.json
```

`--ablation_slug` selects the prompt directory (`full` / `nohvp` / `hvp10`). `--model` accepts any ID your OpenAI-compatible endpoint exposes (fine-tune ID, Azure deployment name). `OPENAI_BASE_URL` is also respected.

`pred.json` (no metric fields):
```
{
  "meta":    {model, ablation, dataset_path, n_samples, n_steps, perturbation?},
  "samples": {<sample>: {steps: {<step>: {x_marker, y_marker, options, cells: [...]}}}}
}
```

The vLLM counterpart (`src.llm.inference.run_vllm`) shares the same loader / output schema.

## Step 4 — `src.llm.postprocess.propagate`: cluster preds → per-cell `prediction.json`

O(1) lookup via `cell2cluster.npz`: each parent cell's `cluster_labels[i]` indexes into a slot→LLM-label table. No 1-NN search. Parse-failure clusters (`prediction=None`) leave their parent cells with `None`, which `src.eval` counts as wrong.

```bash
python -m src.llm.postprocess.propagate \
    --eval_path  results/gpt-5.4_hvp10/Acute2020/pred.json \
    --dataset    Acute2020 \
    --benchmark  benchmark/ --data-dir data/ \
    --method     gpt-5.4_hvp10 \
    --output_dir results/gpt-5.4_hvp10
```

`--eval_path` / `--dataset` are repeatable for multi-dataset runs in one call. Output: `{output_dir}/{dataset}/{sample}/step_NN/prediction.json` (`src.eval` compatible).

## Step 5 — `src.eval`

Method-agnostic. See [Evaluation Protocol in the top-level README](../../README.md#evaluation-protocol).

```bash
python -m src.eval --predictions results/gpt-5.4_hvp10 \
    --benchmark benchmark/ --data-dir data/
```

## Optional cluster-level diagnostics

Not required for the core pipeline.

```bash
# Cluster-level metric (label accuracy on K clusters before propagation) → pred_anchor_metrics.json
python -m src.llm.postprocess.anchor_metric \
    --eval_path results/gpt-5.4_hvp10/Acute2020/pred.json

# Cluster GT-vs-Pred plot per sample (all steps stacked as rows × 3 cols)
python -m src.llm.plot.anchor_pred_vs_gt \
    --eval_path results/gpt-5.4_hvp10/Acute2020/pred.json \
    --dataset Acute2020 \
    --benchmark benchmark/ --data-dir data/ \
    --output_dir results/gpt-5.4_hvp10/Acute2020

# Sim-disagreement edge graph per step
#   Edges = top-3 (x,y)-NN clusters; weights = rep cell raw-protein cosine sim.
#   Same-label edges gray, cross-label edges orange.
python -m src.llm.plot.disagreement_graph \
    --eval_path results/gpt-5.4_hvp10/Acute2020/pred.json \
    --dataset Acute2020 \
    --benchmark benchmark/ --data-dir data/ \
    --output_dir results/gpt-5.4_hvp10
```

### Cluster smoothing (`src.llm.postprocess.smooth`)

Optional pre-propagation step. For the clusters the LLM actually saw (`--n-cells` truncates excluded):

1. Edge graph = top-3 (x, y)-NN rep cells.
2. Edge weight = cosine similarity on **raw** protein expression of each cluster's rep cell (captures orthogonal biological similarity, not the gating-axis structure).
3. Greedy: while some cluster has `sim_weighted_diff_label_degree / total_degree ≥ 0.5`, pick the worst offender and flip it to the plurality label among its orange neighbors. Safety-capped at `max_iterations=500`.

```bash
python -m src.llm.postprocess.smooth \
    --eval_path results/gpt-5.4_hvp10/Acute2020/pred.json \
    --dataset Acute2020 \
    --benchmark benchmark/ --data-dir data/

# Before/after visualization
python -m src.llm.plot.disagreement_graph \
    --eval_path     results/gpt-5.4_hvp10/Acute2020/pred.json \
    --smoothed_path results/gpt-5.4_hvp10/Acute2020/pred_smoothed.json \
    --dataset Acute2020 \
    --benchmark benchmark/ --data-dir data/ \
    --output_dir results/gpt-5.4_hvp10
```

Output: `pred_smoothed.json`. Re-run propagate → `src.eval` with a distinct `--method` to compare with vs without smoothing.

## Setting 2 (In-Panel Hard)

See [Setting 2 in the top-level README](../../README.md#setting-2-in-panel-hard). Pass the same `--perturbation` (or `--hard-depletion <HID>` / `--hard-shift <HID>`) spec to every stage of the pipeline; the slug propagates consistently through `c2s__{slug}.json` / `cell2cluster__{slug}.npz` / `c2s_{ablation}__{slug}/` / `pred.json.meta.perturbation` / `prediction.json`. Mismatches hard-fail at the next stage.
