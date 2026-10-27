# flowDensity + notes.yaml rule baseline (no LLM)

A fully rule-based gating baseline. For every gating step, flowDensity finds
density valleys on the two axes, and the step's `rule` in
`benchmark/notes.yaml` maps the resulting grid cells to category labels.
No language model is involved, and ground-truth labels are never used to
place or name a gate. They are used only by `src.eval` for scoring.

Results go to `results/flowdensity_rule/{dataset}/{sample}/step_NN/prediction.json`
(method name `flowdensity_rule`), with per-cohort `post_summary.json` and
`eval_summary.json`.

## Method

### Inputs

`pre_rule.py` iterates the test split through `BenchmarkLoader` and, per step, writes:

- `input.csv` holds the `(x, y)` values of the step's parent population (the
  ground-truth parent gate, as for every other method in the benchmark). No
  labels are written.
- `meta.json` holds the task id, categories, parent size and the validated rule.

The rule is checked against `task.json` before anything is written. The run
stops with an error if any of the following is wrong:

- `rule.x_marker` and `rule.y_marker` must match the task's markers exactly.
- `regions` must be a rectangular table (or the string `unsupported`).
- Every region label must be one of the step's categories.

### The rule

```yaml
rule:
  x_marker: CD66_Gd158Di
  y_marker: CD45_In115Di
  regions:            # regions[y_bin][x_bin]; rows/columns run low → high
  - [Unassigned, Granulocyte]
  - [CD45hiCD66lo, Granulocyte]
```

An `R × C` table needs `C − 1` valleys on x and `R − 1` valleys on y. A
cell's bin index on an axis is the number of valleys strictly below its
value; a value equal to a valley stays in the lower bin. Repeated labels
merge regions. `notes.yaml` gives no rule for choosing valleys; the valley
selection below is this baseline's own.

Across the 263 cohort-steps in `notes.yaml`:

| Valleys needed (x, y) | Cohort-steps |
|---|---|
| (1, 1) | 241 |
| (1, 2) | 3 (FRDR_covid19 s10, FR-FCM-Z74D_hc s5, FR-FCM-Z74D_tissue s5) |
| (2, 2) | 3 (Acute2020 / Acute2021 / Vaccine s18) |
| `unsupported` | 16 (Acute2020 / Acute2021 / Vaccine s2–5, FRDR_covid19 s1–4) |

### Valley selection (`flowdensity_valleys.R`)

flowDensity 1.38 (`deGate`) on R 4.4, applied to each axis of the parent
population separately. With `k` = the number of valleys the rule needs on
that axis:

| Case | Action | `{axis}_method` |
|---|---|---|
| `k = 1` | `deGate` default single cut | `default` |
| `k ≥ 2`, `all.cuts` returns exactly `k` cuts | use them | `all_cuts` |
| `k ≥ 2`, more than `k` cuts | keep the `k` deepest cuts | `top_depth` |
| `k = 2`, exactly 1 cut | re-run `deGate` on each side of the cut and keep the deeper valid valley | `split_rescue` |
| `k = 2`, 1 cut, neither side gives a valid valley | keep the single cut and drop the middle label (see below) | `middle_dropped` |
| `deGate` raises or returns no finite cut | record `{axis}_error`; no substitute threshold | `error` |

**Depth** of a cut is measured on `density(x, adjust = 1)`, the same
smoothing `deGate` uses by default. It is the lower of the two neighbouring
peak heights minus the density at the cut. Each peak is the highest density
between the cut and the adjacent cut (or the data edge) on that side.

**Split rescue.** `deGate` (default single cut) is re-run on the cells below
the cut and on the cells above it. A side's cut counts as a valley only if it
lies strictly inside that side and has positive depth on the full-axis
density with the original cut as its neighbour. If both sides qualify, the
deeper one is kept.

**Middle dropped.** If no second valley is found, the axis keeps one valley:
cells below it take the lowest region row/column, cells above it take the
highest, and the middle label is never predicted on that axis.

### Prediction (`post_rule.py`)

| Step status | Prediction | Counted as |
|---|---|---|
| `ok` | `regions[y_bin][x_bin]` per cell | valid |
| `ok_middle_dropped_{x,y}` | as above, outer labels only on that axis | valid |
| `unsupported_all_target` | every parent cell gets the step's single target category (all 16 unsupported steps have exactly one; more than one is an error) | fixed prediction, not a gate |
| `degate_error_{x,y}` / `valley_count_{x,y}` | all `None` | wrong (scored, not dropped) |

The unsupported steps are acquisition-QC / scatter / singlet cleanup gates
(e.g. `Time × Residual → Cleanup2`, `FSC-A × SSC-A → Cells`). Predicting
"keep everything" scores high accuracy but about 0.5 macro-F1, because
`Unassigned` recall is 0. Report them as a fixed prediction, not a
flowDensity gate.

Failed steps are still scored because `src.eval` only scores steps that have a
`prediction.json`. An all-`None` prediction keeps a failed step in the
denominator instead of silently dropping it.

## Running

Per cohort (`DS`), on an LSF compute node:

```bash
python -m src.baselines.flowdensity_rule.pre_rule --output-dir results/flowdensity_rule \
    --datasets $DS --split benchmark/splits.json --split-set test
module load R/4.4
Rscript src/baselines/flowdensity_rule/flowdensity_valleys.R \
    --results-dir results/flowdensity_rule/$DS --cores 8
python -m src.baselines.flowdensity_rule.post_rule --results-dir results/flowdensity_rule/$DS
python -m src.eval --predictions results/flowdensity_rule --benchmark benchmark --data-dir data \
    --datasets $DS --split benchmark/splits.json --split-set test
```

`post_rule` deletes each `input.csv` after writing `prediction.json`.

The following flags rebuild only part of a finished run:

- `pre_rule --only-multi-valley` rewrites only steps that need two or more
  valleys.
- `post_rule --reuse-existing` re-posts only steps that still have
  `input.csv`; every other status is read back from its `prediction.json`.

R packages: `flowCore`, `flowDensity`, `jsonlite`, `data.table`.

## Results (test split, 2026-10-20)

386 test samples, 11 cohorts, 11,671 steps. Each step is scored on its
ground-truth parent population. Means are over steps within a cohort; the
last row averages the 11 cohort means with equal weight. Counts remain totals.

| Cohort | Test samples | Steps | `ok` | Middle dropped | Unsupported → target | deGate error | F1 macro | F1 weighted | Balanced acc | Pop. prop. err |
|---|---|---|---|---|---|---|---|---|---|---|
| Acute2020 | 12 | 742 | 635 | 5 | 48 | 54 | 0.476 | 0.639 | 0.591 | 0.207 |
| Acute2021 | 33 | 2035 | 1753 | 25 | 132 | 125 | 0.500 | 0.694 | 0.634 | 0.192 |
| Bjornson | 32 | 256 | 256 | 0 | 0 | 0 | 0.731 | 0.826 | 0.772 | 0.134 |
| Vaccine | 85 | 5270 | 4754 | 24 | 340 | 152 | 0.486 | 0.653 | 0.618 | 0.213 |
| FR-FCM-Z74D_hc | 2 | 24 | 22 | 2 | 0 | 0 | 0.589 | 0.841 | 0.633 | 0.086 |
| FR-FCM-Z74D_tissue | 15 | 180 | 159 | 15 | 0 | 6 | 0.467 | 0.720 | 0.559 | 0.144 |
| FRDR_covid19 | 111 | 2660 | 2024 | 103 | 444 | 89 | 0.594 | 0.775 | 0.681 | 0.195 |
| Lyoplate_DC | 24 | 120 | 120 | 0 | 0 | 0 | 0.750 | 0.879 | 0.861 | 0.058 |
| Lyoplate_bcell | 24 | 120 | 120 | 0 | 0 | 0 | 0.612 | 0.755 | 0.715 | 0.169 |
| Lyoplate_tcell | 24 | 144 | 144 | 0 | 0 | 0 | 0.619 | 0.787 | 0.782 | 0.074 |
| Lyoplate_treg | 24 | 120 | 119 | 0 | 0 | 1 | 0.535 | 0.690 | 0.629 | 0.159 |
| **All (equal-cohort mean)** | **386** | **11671** | **10106** | **174** | **964** | **427** | **0.578** | **0.751** | **0.679** | **0.148** |

The 427 `deGate` errors (304 on x, 123 on y) are raised inside flowDensity:
"no finite cut", `wrong sign in 'by' argument`, `argument "sd.threshold" is
missing`. No valley-count failures remain after the depth-based selection.

### Axes that need two valleys

388 axis instances (6 rules × test samples). `deGate(all.cuts = TRUE)`
returned exactly 2 cuts on 57 of them (15%), 1 cut on 231 and 3–7 cuts on
100.

| Cohort step | Axis | n | `all_cuts` (=2) | `top_depth` | `split_rescue` | `middle_dropped` |
|---|---|---|---|---|---|---|
| FRDR_covid19 s10 | y (BV421) | 111 | 5 | 1 | 2 | 103 |
| Acute2020 s18 | x (CD38) | 12 | 4 | 1 | 4 | 3 |
| Acute2020 s18 | y (CD14) | 12 | 2 | 1 | 5 | 4 |
| Acute2021 s18 | x (CD38) | 33 | 5 | 1 | 10 | 17 |
| Acute2021 s18 | y (CD14) | 33 | 1 | 4 | 10 | 18 |
| Vaccine s18 | x (CD38) | 85 | 14 | 51 | 11 | 9 |
| Vaccine s18 | y (CD14) | 85 | 26 | 41 | 1 | 17 |
| FR-FCM-Z74D_hc s5 | y (PD-1) | 2 | 0 | 0 | 0 | 2 |
| FR-FCM-Z74D_tissue s5 | y (PD-1) | 15 | 0 | 0 | 0 | 15 |
| **Total** | | **388** | **57** | **100** | **43** | **188** |

Split rescue recovered a second valley on 43 of the 231 single-cut axes.
The middle label is dropped on the other 188, which covers almost every FRDR
s10 and Z74D s5 sample. Step-level F1 macro on these steps: Vaccine s18 0.505,
FRDR s10 0.431, Acute2020 s18 0.430, Z74D_hc s5 0.301, Z74D_tissue s5 0.256,
Acute2021 s18 0.202.
