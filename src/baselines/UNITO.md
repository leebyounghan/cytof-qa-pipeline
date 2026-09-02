# UNITO baseline — how it was run

The UNITO rows of the paper come from a reimplementation of
[UNITO](https://github.com/KyleeCJ/UNITO) (Chen et al., 2025). We do not
ship the training harness in this repository; this note records the exact
recipe so the baseline can be reproduced against the released benchmark.

## Model

Per (cohort, step): a U-Net (features 64/128/256/512, `torch`) that
segments the gating region on a **101×101 density image** of the step's
biaxial plot, following the upstream design.

## Data preparation (per gating step)

1. Load the step's parent-population cells from the parquet
   (`BenchmarkLoader` semantics: parent condition from `task.json`).
2. Min–max normalize the `(x_marker, y_marker)` plane to the 101×101 grid.
3. Density image = 2-D histogram of parent cells (log-scaled).
4. Training mask per category = convex-hull fill of the GT-labeled cells
   on the same grid (upstream's mask construction).

## Training

- One model per (cohort, step, category-set), fit on the **train + val**
  samples of `benchmark/splits.json` (the paper's trained-baseline regime).
- Adam, lr 1e-4, batch 8, up to 1000 epochs, early stopping patience 50.

## Evaluation

- Predict the mask on each **test** sample's density image, map each
  parent cell to predicted-inside/outside per category, resolve overlaps
  by predicted-probability, remainder → `Unassigned`.
- Write the standard per-step `prediction.json` and score with the shared
  evaluator: `python -m src.eval --predictions results/unito/ --benchmark
  benchmark/ --data-dir data/` — identical metrics (F1 macro, balanced
  accuracy, Hull IoU) to every other method.

## Setting 2

Apply the same perturbations at load time via the shared loaders
(`--hard-depletion <HID>` / `--hard-shift <HID>` semantics of
`src.bench`), re-run mask prediction on the perturbed parent population
(no re-training), and score with `src.eval` / `src.eval_hard`.
