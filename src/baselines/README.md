# Non-LLM Baselines (`src.baselines`)

Baselines that write `prediction.json` per step directly via `ti.save_prediction(...)`. The runner is its own propagation step — no separate `propagate` CLI. Score with the same `src.eval` as every other method. (The UNITO trained-baseline recipe is documented in [`UNITO.md`](UNITO.md).)

```bash
# Run → results/{method}/{dataset}/{sample}/step_{NN}/prediction.json
python -m src.baselines.flowsom_baseline \
    --benchmark benchmark/ --data-dir data/ --output-dir results/flowsom/

# Score → results/flowsom/{dataset}/eval_summary.json
python -m src.eval --predictions results/flowsom/ \
    --benchmark benchmark/ --data-dir data/
```

`--plot` (~1 min/sample; e.g. 33-sample Acute2021 ≈ 50 min wall-clock) emits GT-vs-pred 4-panel PNGs in addition to metrics. Omit for metrics-only runs.

## FlowSOM — per-step clustering reference

`flowsom_baseline` trains a square SOM on the `(x, y)` plane, hierarchically clusters the codebook into K metaclusters (average linkage, the canonical FlowSOM choice), assigns each cell via its winning neuron, and maps metacluster → GT label by majority vote. Steps larger than 15 000 parent cells are fit on a uniform subsample and propagated via 1-NN. `--flowsom-grid` / `--flowsom-k` control granularity.

## flowDensity — parallel R stage

The R gating stage (`flowdensity/flowdensity_gate.R`) uses `data.table::fread` for CSV parsing and `parallel::mclapply` to process step directories in parallel. Defaults to `detectCores() - 1`; override with `--cores N`. `--overwrite` re-runs even when `thresholds.json` already exists.

```bash
python -m src.baselines.flowdensity.pre_flowdensity \
    --benchmark benchmark/ --data-dir data/ --output-dir results/flowdensity/

Rscript src/baselines/flowdensity/flowdensity_gate.R \
    --results-dir results/flowdensity/ --cores 8

python -m src.baselines.flowdensity.post_flowdensity \
    --results-dir results/flowdensity/
```

`post_flowdensity --cleanup` deletes each step's `input.csv` after `prediction.json` is written (skipped steps too, as long as `prediction.json` already exists). Useful when disk usage from large parent populations matters. `meta.json` and `thresholds.json` are kept so post-only re-runs with `--overwrite` still work; re-running `pre` regenerates `input.csv` from parquet.

Parallelism falls back to serial `lapply` on Windows (no `fork`).

## cyMAE — per-cohort encoder + per-step head

Trained baseline that follows the [cyMAE](https://github.com/JaesikKim/cyMAE) recipe
(MAE pretrain → supervised classification head), per cohort. The encoder consumes
the full panel (every channel except `tech` / `background`); each gating step gets
its own MLP head on top of the frozen mean-pooled cell embedding.

Three stages — pretrain, head training, inference — each with its own CLI. All cells
in pretrain and head training are random-subsampled at 10 000 per parquet (configurable).
Inference processes every cell of the test parent population.

```bash
# Per-cohort pretrain → encoder.pth + marker_order.json + channel_stats.npz
python -m src.baselines.cyMAE.pretrain --cohort Acute2020 \
    --benchmark benchmark/ --data-dir data/ --output-dir artifacts/cymae/

# Per-step heads (MLP on frozen embeddings) → heads/step_NN.pth
python -m src.baselines.cyMAE.train_heads --cohort Acute2020 \
    --benchmark benchmark/ --data-dir data/ \
    --encoder-dir artifacts/cymae/Acute2020/ \
    --output-dir  artifacts/cymae/Acute2020/heads/

# Inference (matches the canonical baseline CLI shape)
python -m src.baselines.cyMAE.inference \
    --benchmark benchmark/ --data-dir data/ \
    --encoder-root artifacts/cymae/ \
    --output-dir results/cymae/ \
    --split benchmark/splits.json --split-set test
```

Full recipe details, hyperparameter table, and vendoring notes live in
[`cyMAE/README.md`](cyMAE/README.md).

## Setting 2 (In-Panel Hard)

See [Setting 2 in the top-level README](../../README.md#setting-2-in-panel-hard). Pass the same `--perturbation` (or `--hard-depletion <HID>` / `--hard-shift <HID>`) spec to both the baseline runner and `src.eval` (or `src.eval_hard`).
