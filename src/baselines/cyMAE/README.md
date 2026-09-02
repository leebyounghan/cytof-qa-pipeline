# cyMAE — per-cohort encoder + per-step head

Trained baseline based on the Cytometry Masked Autoencoder
([Kim et al., bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.02.13.580114v2);
upstream code at <https://github.com/JaesikKim/cyMAE>, Apache-2.0).

Pipeline matches cyMAE's native recipe (per-cohort MAE pretrain → supervised head),
adapted to our per-step task and per-cohort marker panels:

| Stage | What it does | Granularity |
|---|---|---|
| 1. Pretrain | MAE reconstruction on cohort train cells | one encoder per cohort |
| 2. Heads | MLP on frozen mean-pooled embedding + GT labels | one head per (cohort, internal step) |
| 3. Inference | Encoder + head → per-cell label, write `prediction.json` | per task |
| 4. Score | Standard `src.eval` | per cohort |

## Layout

```
src/baselines/cyMAE/
  vendor/                   # Apache-2.0 vendored upstream (see vendor/NOTICE)
    LICENSE NOTICE
    modeling_pretrain.py modeling_finetune.py masking_generator.py
  data.py                   # parquet → tensor adapter (channel selection, z-score)
  models.py                 # MAE + encoder factories (parameterized num_features) + StepHead
  pretrain.py               # per-cohort MAE pretrain CLI
  train_heads.py            # per-(cohort, step) MLP head CLI
  inference.py              # BenchmarkLoader-style runner
  README.md                 # this file
```

After pretrain + train_heads:

```
artifacts/cymae/{cohort}/
  encoder.pth               # encoder state dict + num_features
  marker_order.json         # frozen channel order, display names, kinds
  channel_stats.npz         # per-channel mean/std (z-score)
  pretrain_args.json        # CLI args + per-epoch loss
  heads/
    step_01.pth             # MLP head + categories + best val F1
    step_02.pth
    ...
    _summary.json
```

## Quick start

```bash
COHORT=Acute2020

# 1) Pretrain encoder (~minutes-to-hours per cohort depending on cell count)
#    Defaults: bf16 autocast, full cell tensor kept on GPU, no DataLoader workers.
#    Knobs: --amp-dtype {bf16,fp16,off}, --compile, --no-cells-on-gpu (if OOM).
python -m src.baselines.cyMAE.pretrain \
    --cohort $COHORT \
    --benchmark benchmark/ --data-dir data/ \
    --output-dir artifacts/cymae/ \
    --batch-size 16384 --compile

# 2) Train per-step heads
python -m src.baselines.cyMAE.train_heads \
    --cohort $COHORT \
    --benchmark benchmark/ --data-dir data/ \
    --encoder-dir artifacts/cymae/$COHORT/ \
    --output-dir  artifacts/cymae/$COHORT/heads/

# 3) Inference on test split
python -m src.baselines.cyMAE.inference \
    --benchmark benchmark/ --data-dir data/ \
    --encoder-root artifacts/cymae/ \
    --output-dir results/cymae/ \
    --split benchmark/splits.json --split-set test \
    --datasets $COHORT

# 4) Score (no method-specific code)
python -m src.eval \
    --predictions results/cymae/ \
    --benchmark benchmark/ --data-dir data/
```

## Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| `embed_dim` | 30 | cyMAE released `pretrain_mae_30D_6L` |
| `depth` / `heads` | 6 / 6 | same |
| `mlp_ratio` | 4 | same |
| `mask_ratio` | 0.25 | matches released checkpoint `_0.25R` |
| `lr_pretrain` | 1.5e-5 | cyMAE README default |
| `lr_head` | 1e-3 | standard MLP head |
| `weight_decay` | 0.05 (pretrain) / 1e-4 (head) | cyMAE / standard |
| Optimizer | AdamW, betas (0.9, 0.95) for pretrain | cyMAE |
| Pretrain epochs | 100 | fixed; train-set is capped, so cost is bounded |
| Head epochs | 30, early stopping (patience 5 on val F1-macro) | small head, val available |
| Batch | 4096 default / 16384+ recommended (pretrain), 16384 (head) | cyMAE README defaults; the small ViT under-utilizes the GPU at 4096 |
| `--max-cells-per-sample` | 10 000 | uniform random per parquet, fixed seed |
| Class weights (head) | inverse frequency | rare populations otherwise vanish |

The cell cap applies to **pretrain and head training only**. Inference embeds every
cell of every parent population — the per-cell metrics in `src.eval` require it.

## Performance flags

The pretrain/embed paths run mixed-precision and avoid host↔device roundtrips by
default. CLI knobs that affect speed:

| Flag | Default | Effect |
|---|---|---|
| `--amp-dtype` | `bf16` | Autocast dtype. `fp16` enables GradScaler; `off` runs fp32. Use `fp16` on pre-Ampere GPUs (e.g. V100). |
| `--compile` | off | `torch.compile(model)` — adds a one-time graph warmup, then ~1.2–1.5× steady-state. |
| `--cells-on-gpu` / `--no-cells-on-gpu` | on | Keep the full normalized cell tensor on GPU and shuffle indices (no DataLoader). Disable if VRAM is tight; the loop falls back to a pinned host tensor. |
| `--batch-size` | 4096 | Bumping to 16384–32768 improves utilization of the small ViT. Scale `--lr` if you go much higher. |
| `--num-workers` | 0 | No-op; cells live in memory. Kept for back-compat. |

`train_heads.py` and `inference.py` keep the encoder embedding on GPU end-to-end —
no extra flags. Inference auto-uses bf16 autocast when running on CUDA.

## Marker / channel selection

The encoder consumes channels with kind ∈ `{protein, livedead, scatter, dna, bead, gaussian}`.
`tech` (Time, Event_length) and `background` (empty CyTOF mass channels) are excluded —
the former is uncorrelated with biology, the latter is pure noise that destabilizes
MAE reconstruction.

Channel order is frozen at pretrain time (`marker_order.json`):
1. Tier order: protein → livedead → scatter → dna → bead → gaussian
2. Alphabetical by channel key inside each tier

Inference enforces an exact channel-key match against the training order — any
missing channel raises an error rather than silently dropping it.

## Per-channel z-score

Channel kinds have wildly different native scales (protein is `arcsinh(raw / 150)`,
scatter is roughly raw, bead/gaussian are CyTOF-specific). Per-channel z-score is
fitted on the capped pretrain set (saved to `channel_stats.npz`) and reused at every
later stage. σ is floored at 1e-6 to avoid divide-by-zero on constant channels.

## Decoupling `embed_dim` from `num_features`

`vendor/modeling_pretrain.py` defines `feature_embed = nn.Parameter[num_features, embed_dim-1]`
— the two are independent in the architecture. The upstream factories
(`pretrain_mae_30D_6L` etc.) hardcode `encoder_num_features=30` (MDIPA panel size),
so we don't use them; `models.build_mae(num_features=M_c)` in `models.py` instantiates
`PretrainVisionTransformer` directly with the cohort's `M_c`. Every other hyperparameter
matches the released `_30D_6L` variant.

## What is *not* included

- **Setting 2 (hard depletion / hard shift) eval.** Encoder + heads are trained on
  Setting 1 only; perturbed evaluation would be a separate experiment.
- **Cross-panel transfer.** cyMAE uses positional marker indexing — cross-panel use
  is non-trivial. This baseline is in-panel only by design.
- **Architecture sweeps.** Recipe matches the released `_30D_6L` checkpoint verbatim
  (with `num_features` swapped per cohort).

## Vendoring notes

`vendor/` keeps only the files we exercise:

- `modeling_pretrain.py`, `modeling_finetune.py`, `masking_generator.py`, `LICENSE`

We dropped `engine_for_pretraining.py`, `engine_for_finetuning.py`, `optim_factory.py`,
and `utils.py` — they pull in `tensorboardX` and distributed-training utilities we
don't use. Our training loops live in `pretrain.py` / `train_heads.py` and use plain
AdamW. Patches applied in `modeling_pretrain.py`:

- import switched to relative form
- non-learnable `pos_embed` registered as a non-persistent buffer in both the
  encoder and the MAE wrapper, so `.to(device)` moves it once and forward passes
  stop reallocating it via `.clone().detach()` each step (existing checkpoints
  remain compatible because the buffer is `persistent=False`)

See `vendor/NOTICE` for the full list of modifications.
