# `src/flowsom_c2s/` — Whole-cell C2S + LLM (flat-leaf)

Whole-sample alternative to `src/llm/`: cluster the whole sample once with
**FlowSom** over all protein markers, then ask the LLM to assign each cluster
**one final leaf cell-type label** picked from a flat enumerated list. The
leaf is decomposed back into per-step `prediction.json` files that
`src.eval` scores unchanged.

This package is independent of `src/llm/` — it only uses `src.bench` /
`src.eval` / a few helpers from `src.preprocess`.

## End-to-end

```bash
DATASET=Acute2020
PLAN=data/$DATASET/gating_plan_L1_coarse.json   # L1 / L2 / L3 — same flag everywhere

# 1) Preprocess: emit benchmark_flat/<ds>/<sample>/<plan_slug>/task.json
python -m src.flowsom_c2s.preprocess \
    --data-dir data/ --output-dir benchmark_flat/ \
    --datasets $DATASET --gating-plan $PLAN

# 2) FlowSom whole-sample clustering (plan-independent — shared across L1/L2/L3)
#    Output: benchmark_flat/<ds>/<sample>/{c2s.json, cell2cluster.npz}
python -m src.flowsom_c2s.c2s \
    --benchmark benchmark_flat/ --data-dir data/ \
    --datasets $DATASET --n-meta-clusters 40

# 3) Render per-sample prompt under benchmark_flat/<ds>/<sample>/<plan_slug>/flowsom_<ablation>/cells.md
for s in benchmark_flat/$DATASET/*/; do
    python -m src.flowsom_c2s.c2s_prompt --batch "$s" \
        --gating-plan $PLAN --top-hvp-n 10
done

# 4) Inference -> results_flat/flowsom_gpt5_hvp10/<plan_slug>/<ds>/pred.json
python -m src.flowsom_c2s.inference.run_openai \
    --benchmark benchmark_flat/$DATASET \
    --gating-plan $PLAN \
    --model gpt-5.4 --ablation_slug hvp10 \
    --output_path results_flat/flowsom_gpt5_hvp10/L1_coarse/$DATASET/pred.json

# 5) Decompose leaves -> predicted_steps.parquet (output_dir auto-appends plan_slug)
python -m src.flowsom_c2s.postprocess \
    --pred results_flat/flowsom_gpt5_hvp10/L1_coarse/$DATASET/pred.json \
    --benchmark-flat benchmark_flat/ \
    --gating-plan $PLAN \
    --output-dir results_flat/flowsom_gpt5_hvp10/

# 6) Flat eval (shared with src/llm_gate_flat/)
python -m src.flat_eval \
    --output-dir  results_flat/flowsom_gpt5_hvp10/L1_coarse/ \
    --gating-plan $PLAN \
    --data-dir    data/ --benchmark benchmark/ \
    --datasets    $DATASET
```

Plan slug is auto-derived from the gating-plan filename:
`gating_plan.json` → `default`, `gating_plan_L1_coarse.json` → `L1_coarse`,
`gating_plan_L2_identity.json` → `L2_identity`. L1 / L2 / L3 outputs are
kept in distinct subdirs so all three depths can co-exist.

`src.flat_eval` is the **shared** evaluator across `src/flowsom_c2s/` and
`src/llm_gate_flat/` — both methods emit `predicted_steps.parquet` with the
same Step* schema as the GT parquet, so flat-leaf metrics are directly
comparable.

## What's different vs. `src/llm/`

|  | `src/llm/` (per-step) | `src/flowsom_c2s/` (whole-cell) |
|---|---|---|
| Clustering | per gating step, per parent population | once per sample, all markers |
| Algorithm | Stage1 KMeans on (x,y) + Stage2 KMeans on protein centroids | FlowSom (SOM grid + consensus hierarchical) |
| Prompt unit | 1 prompt per (sample, step) | 1 prompt per sample |
| LLM answer | category per cluster *for this step* | one **leaf cell-type** per cluster |
| Mapping back to steps | per-step cluster→category propagation | leaf decomposition via `leaf_to_step_labels` |

## Leaf vocabulary (shared with `src/llm_gate_flat/`)

The leaf set is computed by `src.flat_eval.derive_leaf_set(gating_plan)` —
exactly the categories that **never appear as a parent** in any later
step. Way-points like `CD3pos` / `CD4` / `CD8` (used as parents) are
filtered out; only terminal classes (`DNT`, `DPT`, `CD4_Naive`,
`CD4_Activated`, `CD8_38negDRneg`, ...) qualify.

A cluster is **multi-label**: in plans with parallel axes (e.g.
`Lyoplate_tcell` Step3 differentiation × Step4 activation under the CD4
parent) one cluster can carry leaves on both axes. The LLM picks a JSON
list per cluster; postprocess populates each leaf's matching Step* column
in `predicted_steps.parquet`. Cells whose cluster picked
`["Discard"]` or an empty list collapse to the synthetic `Discard` class
in `src.flat_eval`.

Plan variants (L1_coarse / L2_identity / L3) are supported transparently:
just pass the matching `--gating-plan` to both `preprocess` and `flat_eval`.

## Files

| File | Role |
|---|---|
| `utils.py` | leaf enumeration, parent-clause evaluator, HVP ranking (forked) |
| `preprocess.py` | parquet + gating_plan → `benchmark_flat/<ds>/<sample>/task.json` |
| `c2s.py` | FlowSom + rep-cell selection → `c2s.json`, `cell2cluster.npz`, scatter |
| `c2s_prompt.py` | per-sample `cells.md` for ablations `full` / `hvp10` / `nohvp` |
| `inference/run_openai.py` | async OpenAI runner — 1 call per sample |
| `inference/output_schema.py` | parse + validate LLM JSON, assemble `pred.json` |
| `postprocess.py` | leaf → per-step `prediction.json` via `cell2cluster.npz` |

## Dependencies

- `minisom>=2.3` + `scipy` (faithful FlowSom: SOM grid + Ward hierarchical metaclustering)
- `openai>=1.30`, `python-dotenv` (inference only)

We use `minisom` rather than the saeyslab `flowsom` package because the
latter transitively requires `numba` / `llvmlite`, which fails to build
from source on common macOS x86_64 + Python 3.11 setups. The algorithm is
the same: SOM on protein channels → hierarchical clustering on the SOM
codebook → cell-to-metacluster via BMU lookup.

The pipeline auto-discovers protein channels from `meta.json` (`kind == "protein"`),
so scatter / Time / bead channels are excluded from clustering.

## Out of scope (v1)

- `--hard-depletion` / `--hard-shift` perturbations
- vLLM runner
- Anchor / smoothing post-processing
