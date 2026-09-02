# CytoGate-Bench

Benchmark for cross-panel hierarchical cell gating in cytometry.

## Overview

CytoGate-Bench formalizes **cross-panel hierarchical gating** as a benchmark task. Unlike flat cell type annotation, this benchmark requires sequential gating decisions on biaxial plots — mirroring the reasoning of expert immunologists.

### Task Definition

Each instance is a single **gating step**:

- **Input**: parent population (2D: x_marker vs y_marker) + marker pair + category list
- **Output**: per-cell label (one of the given categories, or `null` for unclassifiable cells)
- **Evaluation**: step-level F1, balanced accuracy, population-proportion error, and 2D convex-hull IoU — computed on all parent-gate cells, treating `Unassigned` as a regular category (methods that mislabel background cells get penalized).

### What is a gating step?

Experts analyze cells two markers at a time on a 2D scatter plot (a **biaxial plot**), drawing boundaries to separate populations (e.g. CD4+ T cells from CD8+ T cells). The process is hierarchical: separate broad groups first, then progressively refine subtypes through a **gating tree**. Each step is one benchmark task. Your method receives the parent population (cells that passed all previous gates) and classifies them into the given categories using the specified marker pair.

## Data Format

### Raw data: parquet

Each sample lives at `data/{dataset}/parquet/{sample}.parquet`. Rows are cells; columns are either expression channels or gating-label columns (`Step*`).

A cell's value in a `Step*` column is one of:
1. **A category label** (e.g. `"CD4"`, `"Naive_CD8"`)
2. **`NaN` / empty string** — either outside the parent population, or `Unassigned` within it

To distinguish the two, check the **parent condition** in `gating_plan.json`. If a cell satisfies the parent condition but has `NaN`, treat it as `Unassigned`.

Example T cell panel columns:
```
Step01_CD4|CD8          # "CD4", "CD8", "CD4p_CD8p", "CD4n_CD8n", or NaN
Step02_CCR7|CD45RA_CD8  # "Naive_CD8", "CM_CD8", "EM_CD8", "EMRA_CD8", or NaN
Step03_Foxp3|CD127      # "tot_Treg", "non_Treg", or NaN
```

### `meta.json`

`data/{dataset}/parquet/meta.json` defines the marker panel, channel metadata, and label normalization:

```json
{
  "cohort": "covid19_panel8_myeloid",
  "cofactor": 150.0,
  "channels": {
    "Comp-FITC-A":  ["CD303"],
    "Comp-BV421-A": ["CD14"],
    "FSC-A":        ["FSC-A"],
    "Time":         ["Time"]
  },
  "category_map": {"old_label": "new_label"}
}
```

- **`channels`**: instrument channel → `[marker_name, ...]`. Used to build `channel_marker_map` and identify expression columns.
- **`cofactor`**: arcsinh transform cofactor (informational — parquet is already transformed).
- **`category_map`**: optional cross-cohort label normalization.

### `gating_plan.json`

Defines the hierarchical gating tree — step sequence, marker pairs, parent conditions, categories:

```json
{
  "steps": [
    {
      "step": 1,
      "parent": "ALL",
      "x_marker": "CD4",
      "y_marker": "CD8",
      "annotation column name": "Step01_CD4|CD8",
      "annotation categories": ["CD4", "CD8", "CD4p_CD8p", "CD4n_CD8n"],
      "note": "CD4: helper T cells — lower-right (CD4+, CD8-). CD8: ..."
    },
    {
      "step": 2,
      "parent": "Step01_CD4|CD8 == CD8",
      "x_marker": "CCR7", "y_marker": "CD45RA",
      "annotation column name": "Step02_CCR7|CD45RA_CD8",
      "annotation categories": ["Naive_CD8", "CM_CD8", "EM_CD8", "EMRA_CD8"],
      "note": "..."
    }
  ]
}
```

Key fields:
- **`parent`**: which cells enter this step. `"ALL"` = all cells. Supports `==`, `!=`, `in`. Combine with `&` (e.g. `"Step01 == CD8 & Step02 != Unassigned"`).
- **`x_marker` / `y_marker`**: the two markers to examine (channel names resolved via `meta.json`).
- **`annotation categories`**: possible labels at this step (excluding `Unassigned`).
- **`note`**: describes the **spatial shape of each population** in the GT biaxial plot — where each cluster sits, what the gate boundaries follow (valley between bimodal peaks, unimodal peak position, density-anchored cluster edges), and any strategy needed to identify each population. Calibrated against `biaxial_plot_gt.png` density structure across multiple samples — use this style when authoring or improving notes.
- **`tips`**: optional. Marker-level / domain hints surfaced only when the prompt is rendered with tips enabled.

## Benchmark Structure

After preprocessing:

```
benchmark/
  {dataset}/
    panel_meta.yaml
    {sample}/step_{NN}/
      task.json              # Task metadata + GT summary
      biaxial_plot.png       # Density-colored scatter (viridis)
      biaxial_plot_gt.png    # GT category overlay
  splits.json                # Train/val/test split definitions
  hard_depletion.yaml        # Curated In-Panel Hard spec (43 hard_ids, 9 scenarios)
  hard_shift.yaml            # Curated channel-shift spec (52 hard_ids, 9 scenarios)
```

### `task.json`

```json
{
  "task_id": "FR-FCM-Z74D_hc__sample_name__step_01",
  "dataset": "FR-FCM-Z74D_hc",
  "sample": "sample_name",
  "step": 1,
  "parent": "ALL",
  "x_marker": "CD4", "y_marker": "CD8",
  "annotation_column": "Step01_CD4|CD8",
  "categories": ["CD4", "CD8", "CD4p_CD8p", "CD4n_CD8n", "Unassigned"],
  "note": "...",
  "tips": "...",
  "gating_context": "Root",
  "gt_summary": {
    "n_parent_cells": 139372,
    "n_cells_per_category": {"CD4": 66768, "CD8": 63891, "...": "..."},
    "population_proportions": {"CD4": 0.4791, "CD8": 0.4584, "...": "..."}
  }
}
```

### `splits.json`

Per-cohort **train / val / test** definitions. Target ratio **25 / 25 / 50** (1 : 1 : 2) is applied within cohort-specific strata, then concatenated — `test` is the largest split because LLM / VLM zero-shot evaluation only scores on `test`, and we want every meaningful stratum well-represented there. `blacklist` entries are always excluded (e.g. QC-failed samples).

```json
{
  "in_panel": {
    "Acute2020": {
      "train": ["sample_A", "..."],
      "val":   ["sample_B", "..."],
      "test":  ["sample_C", "..."],
      "blacklist": ["sample_X"]
    }
  }
}
```

- **train / val**: clustering / vision method fitting + hyperparameter tuning.
- **test**: held out for final evaluation. LLM / VLM zero-shot also scores on `test` only.
- **blacklist**: never loaded, regardless of `--split-set`.

## How to Add a New Method

`src.bench.BenchmarkLoader` handles parquet loading, parent-population masking, GT label extraction, split filtering, and Setting 2 perturbations. You only write the method.

**Column naming**: `TaskInput.expression` columns are **channel names** (the same keys as the parquet file and `meta.json["channels"]`). Use `ti.display_name(col)` — or `ti.channel_marker_map` — to get human-readable marker names for plots/logs. We keep channel-space internally because CyTOF panels can map multiple channels to the same marker (e.g. several bead channels → marker `"Bead"`); renaming would silently collide.

### Recommended pattern (CLI helpers)

```python
import argparse
from pathlib import Path
from src.bench import BenchmarkLoader

def main():
    parser = argparse.ArgumentParser(description="My gating method")
    parser.add_argument("--output-dir", type=Path, required=True)
    BenchmarkLoader.add_cli_args(parser)
    args = parser.parse_args()

    loader = BenchmarkLoader.from_cli(args)
    for ti in loader.iter_tasks_from_cli(args):
        # ti.expression : pd.DataFrame (N, M)  — parent-pop cells, channel-named columns
        # ti.markers    : list[str]            — channel keys present in expression (ordered)
        # ti.y          : np.ndarray (N,)      — GT labels (NaN → "Unassigned")
        # ti.categories : list[str]            — valid categories for this task
        # ti.X_2d       : np.ndarray (N, 2)    — shortcut: [x_marker, y_marker] plane
        # ti.x_marker, ti.y_marker             — channel keys for the 2D plane
        # ti.x_marker_display, ti.y_marker_display — marker display names (for plots)
        # ti.channel_marker_map                — channel → marker display name
        # ti.task                              — raw task.json
        # ti.task_id, ti.dataset, ti.sample, ti.step, ti.parent, ti.step_dir_name

        pred = my_method(ti.expression[ti.markers].values, ti.categories)

        ti.save_prediction(
            pred,                             # list[str|None] or np.ndarray, length == N
            output_dir=args.output_dir,
            method="my_method",
            info={"any": "metadata"},         # optional
        )
```

`save_prediction` writes `{output_dir}/{dataset}/{sample}/{step_dir}/prediction.json` in the expected format. When a perturbation was passed via `--perturbation`, the same slug is recorded into the prediction file and `src.eval` cross-checks it.

### Direct API

```python
from src.bench import BenchmarkLoader, parse_perturbations

name, perturb = parse_perturbations(["channel_shift CD3=0.5"])  # optional
loader = BenchmarkLoader("benchmark/", "data/", perturbation=perturb, perturbation_name=name)

for ti in loader.iter_tasks(datasets=["Lyoplate_tcell"], split="benchmark/splits.json"):
    ...

ti = loader.load_task("benchmark/Lyoplate_tcell/1228-1_A1_A01/step_01/task.json")
```

### Prediction format

```json
{
  "task_id": "FR-FCM-Z74D_hc__sample_name__step_01",
  "method": "your_method_name",
  "predicted_labels": ["CD4", "CD8", null, "CD4", "..."]
}
```

`predicted_labels` must have exactly the same length as the parent population (`len(ti.expression)`). `null` entries count as wrong predictions.

## Setting 2: In-Panel Hard

Population-shifted / batch-effect scenarios. Pass **identical** `--perturbation` specs to both the baseline runner and the evaluator — each `prediction.json` records the derived slug and eval cross-checks it to catch mismatches.

### Inline syntax

| Type | Syntax | Example |
|---|---|---|
| `deplete` | `deplete <column> <value> [<value> ...]` | `'deplete Step01_CD3\|SSC-A CD3pos'` |
| `ratio_shift` | `ratio_shift <column> <cat>=<w> ... [seed=N]` | `'ratio_shift Step02_CD4\|CD8 CD4=10 CD8=1'` |
| `channel_shift` | `channel_shift <name>=<delta> ...` | `'channel_shift CD3=0.5 145Nd_CD4=-0.3'` |
| `channel_scale` | `channel_scale <name>=<factor> ...` | `'channel_scale CD3=1.5 CD4=0.7'` |

Semantics:
- **`deplete`** — remove all cells whose GT label at `<column>` matches any of the given values. Downstream gating steps whose parent becomes empty are auto-skipped by `iter_tasks`.
- **`ratio_shift`** — downsample per-category cells at `<column>` so the target ratios (weights) are satisfied. Uses the smallest common scale; never upsamples. Cells not named in the ratios are untouched. Optional `seed=N` (default 42).
- **`channel_shift`** — add a constant offset to the listed columns. `<name>` may be a **channel** key (e.g. `145Nd_CD4`) or a **marker display name** (e.g. `CD4`); a marker is broadcast to every channel that maps to it (e.g. `Bead=0.3` shifts every bead channel at once). Simulates per-channel calibration / stain-intensity batch effects.
- **`channel_scale`** — multiplicative factor on **raw (pre-arcsinh) signal**: `raw' = raw × factor`, equivalently `arcsinh' = arcsinh(sinh(arcsinh) × factor)`. `factor=1.0` is no-op. Same name-broadcast rules as `channel_shift`. More physically realistic than `channel_shift` for modeling lot/antibody/instrument drift — preserves dim cells near zero (additive arcsinh shifts push them to unphysical negative raw counts).

### Composing perturbations

Pass `--perturbation` repeatedly; specs apply left-to-right:

```bash
python -m src.baselines.flowsom_baseline \
    --benchmark benchmark/ --data-dir data/ \
    --output-dir results/flowsom_hard/ \
    --perturbation 'deplete Step01_CD3|SSC-A CD3pos' \
    --perturbation 'channel_shift CD3=0.5'

python -m src.eval \
    --predictions results/flowsom_hard/ \
    --benchmark benchmark/ --data-dir data/ \
    --perturbation 'deplete Step01_CD3|SSC-A CD3pos' \
    --perturbation 'channel_shift CD3=0.5'
```

The slug is auto-derived from the specs (e.g. `deplete-Step01_CD3_SSC-A-CD3pos+channel_shift-CD3+0.5`). Override with `--perturbation-name my_hard_a` for a fixed tag.

Perturbations operate on the **loaded, normalized** sample df in channel-space (Step* labels category-mapped, channel columns unrenamed). No parquet is regenerated — pure runtime transforms.

### Curated hard depletions (`--hard-depletion`)

Inline specs are flexible but ad-hoc. The standard In-Panel Hard benchmark pins 43 (dataset, step, depletion) triples covering 9 clinically grounded scenarios in `benchmark/hard_depletion.yaml`:

| # | Scenario | Depleted | Clinical rationale |
|---|---|---|---|
| S1 | `hiv` | CD4 T cells | HIV late-stage / anti-CD4 therapy |
| S2 | `rituximab` | B cells (CD20+) | anti-CD20 (rituximab / ocrelizumab) |
| S3 | `b_cell_aplasia` | B cells (pan-B) | CAR-T-induced B cell aplasia |
| S4 | `immunosenescence` | Naive T cells | Elderly thymic involution |
| S5 | `neutropenia` | Granulocytes / PMN | Post-chemo / idiopathic |
| S6 | `pdc_depletion` | Plasmacytoid DCs | HIV, severe COVID-19, measles |
| S7 | `nonclassical_mono_loss` | Patrolling (CD14dim CD16+) mono | Severe COVID-19 / sepsis |
| S8 | `th17_axis_collapse` | Th17 cells | Secukinumab / ustekinumab |
| S9 | `treg_depletion_daclizumab` | Tregs | Daclizumab / IPEX |

Each task reduces to a standard `deplete` perturbation; the hard_id is used verbatim as the perturbation slug. **Depletion is applied only at the target step's parent population** (not sample-wide). Other steps of the same sample are untouched; step-to-step connectivity is intentionally ignored. The category list is kept intact — this tests whether the model hallucinates an absent population vs correctly labels the populations that remain.

Every stage accepts `--hard-depletion <hard_id>` in lieu of `--perturbation` / `--perturbation-name` / `--datasets` (auto-resolved from the yaml). Default split = in-panel `test`. `--hard-depletion` is mutually exclusive with `--perturbation` / `--perturbation-name` / `--datasets` / `--hard-shift`.

`--hard-depletion-fraction <F>` (default 1.0) sweeps the depletion magnitude — 0.0 = no removal (baseline), 0.5 = half. The slug is suffixed with `_frac{NN}` only when ≠ 1.0.

```bash
HARD_IDS=$(python -c "from src.hard_depletions import load_hard_depletions; \
    print(' '.join(t.hard_id for t in load_hard_depletions()[1]))")
for HID in $HARD_IDS; do
    python -m src.llm_gate.task --hard-depletion $HID
    # ... rest of pipeline ...
done

# YAML self-test
python -m src.hard_depletions
```

### Curated channel-scale tasks (`--hard-shift`)

A second axis of Setting 2: instead of removing populations, perturb the **expression-channel calibration** to mimic real-world batch effects (antibody lot drift, freeze-thaw, bead-normalization residuals). 52 (dataset, step, channel_scale) triples covering 9 technically grounded scenarios are pinned in `benchmark/hard_shift.yaml`:

| # | Scenario | Markers scaled | Technical rationale |
|---|---|---|---|
| C1 | `lineage_dim` | CD3 / CD4 / CD8 / CD19 / CD20 (dim) | Aged antibody / partial conjugation — dim lineage staining |
| C2 | `ccr7_internalization` | CCR7 (dim) | Surface CCR7 lost from freeze-thaw or in-tube activation |
| C3 | `cd45ra_drift` | CD45RA (bright) | Lot-to-lot brightness increase — over-counts Naive / TEMRA |
| C4 | `treg_marker_collapse` | Foxp3 / CD25 (dim), CD127 (bright) | Intracellular Treg-marker dim — collapses the Treg gate |
| C5 | `mono_subset_shift` | CD16 (bright) | CD16 spillover — moves Classical/Intermediate/Patrolling mono and Neutrophil/Eosinophil borders |
| C6 | `dc_marker_drift` | CD11c / CD123 / CD303 (dim) | Lot variability on narrow-dynamic-range DC discriminators |
| C7 | `bead_residual` | Bead (dim, broadcast to 4 mass channels) | Post-EQ4-bead-normalization residual — drifts the step 1 cleanup gate |
| C8 | `viability_drift` | Live_Dead (bright) | Rh staining intensified by storage / freeze-thaw — more cells look dead at step 6 |
| C9 | `panel_global` | All `protein` channels (bright) | Whole-panel calibration drift — instrument-level batch effect |

Each task reduces to a `ChannelScale` perturbation — **multiplicative on raw (pre-arcsinh) signal** rather than additive in arcsinh space. This matches how lot/antibody/instrument calibration drift behaves physically: `raw' = raw × factor`, equivalently `arcsinh' = arcsinh(sinh(arcsinh) × factor)`. Dim populations near zero are preserved (an additive arcsinh shift would push them into unphysical negative raw counts); bright populations rescale proportionally. `factor=1.0` is no-op; `<1` dims, `>1` brightens. The hard_id is used verbatim as the perturbation slug.

**Magnitudes are empirically calibrated per (dataset, step)** as `factor = exp(delta)` where:

```
delta = clip(0.30 × |median(marker | high_classes) − median(marker | low_classes)|, ±1.0)
```

The 0.30 fraction targets ~10–30% of cells crossing the gate boundary; the ±1.0 delta cap (so factor ∈ [0.37, 2.72]) keeps the perturbation in the "lot variability" regime rather than effectively ablating the marker. `panel_global` uses `clip(0.20 × median(protein-channel IQR), 1.0)`. GT labels are unchanged — the test is whether the model is robust to calibration drift without re-fitting.

Markers in `scales:` may be either channel keys (e.g. `145Nd_CD4`) or marker display names (e.g. `CD4`); names broadcast to every channel that maps to the marker. The sentinel `__all_protein_markers__: <factor>` in `panel_global` tasks is expanded by the loader to every channel with `kind == "protein"` in `meta.json` — `bead`, `livedead`, `dna`, `tech`, `scatter`, `background` channels are deliberately excluded.

```bash
HARD_SHIFT_IDS=$(python -c "from src.hard_shifts import load_hard_shifts; \
    print(' '.join(t.hard_id for t in load_hard_shifts()[1]))")
for HID in $HARD_SHIFT_IDS; do
    python -m src.llm_gate.task --hard-shift $HID
    # ... rest of pipeline ...
done

# YAML self-test
python -m src.hard_shifts
```

`--hard-shift` is mutually exclusive with `--hard-depletion` / `--perturbation` / `--perturbation-name` / `--datasets`. The loader (`src.hard_shifts`) provides `add_hard_shift_args` / `resolve_hard_shift_args` helpers that mirror the `hard_depletions` API.

`--hard-shift-scale <S>` (default 1.0) is the magnitude knob — linearly interpolates each calibrated factor toward 1.0: `effective_factor = 1 + (calibrated_factor − 1) × S`. So `S=0.0` → all factors = 1.0 (baseline, no perturbation); `S=0.5` → halfway. The slug is suffixed with `_scale{NN}` only when ≠ 1.0.

### Hard evaluation (`src.eval_hard`)

Walks a result tree organised as

```
{root}/{scenario}/{dataset}_step{NN}/{magnitude_dir}/{dataset}/{sample}/
    step_{NN}/prediction.json
                          /pred.json     # raw LLM output (gates) — optional
```

`{magnitude_dir}` is `frac{NN}` (depletion mode) or `scale{NN}` (shift mode); a single root contains exactly one mode (auto-detected). For every prediction:
- Look up the matching `HardDepletion` / `HardShift` by (scenario, dataset, step).
- Re-load GT with the corresponding perturbation at the parsed magnitude so cells align with the prediction array.
- Compute the standard metrics and aggregate per (hard_id, magnitude), (scenario, magnitude), (dataset, magnitude).
- Optionally render a 1×K side-by-side scatter panel per (hard_id, sample), GT-coloured, with LLM gates as dashed boxes.

```bash
# Depletion tree
python -m src.eval_hard \
    --root results/gate_gpt-5.4_depletion \
    --plot-dir results/gate_gpt-5.4_depletion/plots

# Shift tree (mode auto-detected from scale{NN} subdirs)
python -m src.eval_hard \
    --root results/gate_gpt-5.4_shift \
    --plot-dir results/gate_gpt-5.4_shift/plots
```

Optional filters: `--scenarios neutropenia hiv`, `--magnitudes 0 0.5 1.0`. Output: `{root}/eval_hard_summary.json` and per-(hard_id, sample) PNGs under `{plot-dir}/{scenario}/`.

## Evaluation Protocol

### Metrics

Computed on **all parent-gate cells**, with `Unassigned` treated as a regular category. `null` predictions always count as wrong (mapped internally to `__noise__`).

Label-based:
- **F1 (macro)**: unweighted average F1 across all categories (including `Unassigned`)
- **F1 (weighted)**: weighted by category prevalence
- **Balanced accuracy**: average per-category recall
- **Population proportion error**: mean \|pred % − GT %\| across categories

Spatial (real cell types only, `Unassigned` excluded):
- **Hull IoU (per category)**: IoU of 2D convex hulls of GT-vs-pred cells in the `(x_marker, y_marker)` plane. 1–99% per-axis quantile trimming reduces outlier sensitivity.
- **Hull IoU (macro)**: mean across categories with a valid GT hull (≥3 non-collinear cells after trimming). `iou_hull_n` records contributing categories.
- **Interpretation**: IoU is scale-invariant within a task (area ratio in the same coord system). Cross-task means are simple averages of ratios. Convex hull is a coarse boundary — dense label-level disagreement inside a shared region still scores high IoU; use F1 for cell-level accuracy.

Plots are `--plot` opt-in (~1 min/sample). Per step: 4-panel figure — Input density · GT categories · Predicted categories · GT (solid+fill) vs Pred (dashed+fill) hulls overlaid on a scatter, macro Hull IoU in the title. Hull polygons match the IoU in `eval_summary.json` exactly (same trim, same cell basis).

### In-panel split

Per-cohort `train / val / test` at 25 / 25 / 50. Allocated within cohort-specific strata (1:1:2 within stratum, then concatenated) so every meaningful stratum is well-represented in `test`; small strata drift by ±1–2 samples. Stratum definitions are authoritative in `splits.json`; preprocess only reads the sample lists.

- **Clustering / Vision**: fit on `train`, tune on `val`, score on `test` (`--split benchmark/splits.json --split-set {train,val,test}`)
- **VLM / LLM**: zero-shot on `test` only
- **Blacklist**: always excluded regardless of `--split-set`

### Cross-panel protocol

The benchmark is *cross-panel* at the cohort level: methods face 8 different
marker panels across the 11 cohorts, and LLM-class methods run zero-shot on
every panel with no panel-specific fitting (trained baselines are fit per
cohort). `splits.json` defines only the within-cohort `in_panel`
train/val/test partition; there is no separate machine-readable cross-panel
split.

### Running `src.eval`

```bash
python -m src.eval \
    --predictions results/your_method/ \
    --benchmark benchmark/ --data-dir data/
```

Iterates datasets one at a time and writes one `{predictions}/{dataset}/eval_summary.json` per dataset. The per-sample parquet cache is released between datasets, so peak memory scales with the largest dataset rather than the sum. Restrict with `--datasets Acute2020 Vaccine`; omit to auto-discover every dataset under `--predictions`.

## Datasets

| Dataset | Type | Panel | Samples |
|---|---|---|---|
| Acute2020 | CyTOF | PBMC (MDIPA, 30-marker) | 25 |
| Acute2021 | CyTOF | PBMC (MDIPA, 30-marker) | 65 |
| Bjornson | CyTOF | Phospho-signaling (Nolan) | 64 |
| FR-FCM-Z74D_hc | Flow | T cell (HC) | 5 |
| FR-FCM-Z74D_tissue | Flow | T cell (Tissue) | 32 |
| FRDR_covid19 | Flow | Myeloid (Panel 8) | 223 |
| Lyoplate_bcell | Flow | B cell | 63 |
| Lyoplate_DC | Flow | DC/Mono/NK | 63 |
| Lyoplate_tcell | Flow | T cell | 63 |
| Lyoplate_treg | Flow | T reg | 63 |
| Vaccine | CyTOF | PBMC (MDIPA, 30-marker) | 169 |

## Data

The benchmark data (arcsinh-transformed float16 parquet channel values plus
per-event gating-step labels; no raw FCS files) is distributed via Harvard
Dataverse:

> CytoGate-Bench: Evaluating LLMs on Cross-Panel Cell Gating in Cytometry.
> Harvard Dataverse, [doi:10.7910/DVN/RDW4UL](https://doi.org/10.7910/DVN/RDW4UL)

The deposit bundles cohorts obtained from several upstream sources that carry
different licenses, so the dataset terms of use are **segmented per cohort** —
see the Terms tab on the Dataverse record before redistributing any subset.
The code in this repository is MIT-licensed (see `LICENSE`); the data license
is governed solely by the Dataverse terms.

After downloading, place each cohort under `data/{dataset}/parquet/` as
described in **Data Format** above.

**Lyoplate cohorts** (`Lyoplate_tcell/treg/bcell/DC`) are *not* included in
the deposit — they are distributed through ImmPort/ImmuneSpace under the
NIAID-DAIT Data Use Agreement. Obtain the source FCS files there and rebuild
the harmonized per-step parquet with the extraction scripts in this
repository; their gating plans, per-step annotations, and splits are already
in `benchmark/`.

## Citation

```bibtex
@article{kim2026cytogatebench,
  title   = {CytoGate-Bench: an LLM benchmark for cross-panel cell gating in cytometry},
  author  = {Kim, Jaesik and Lee, Byounghan and Ahn, Namhyuk and Ionita, Matei and
             McKeague, Michelle L. and Lee, Matthew E. and Jeong, Chang-Uk and
             Apostolidis, Sokratis A. and Baxter, Amy and Shwetank and
             Greenplate, Allison R. and Wherry, E. John and Sohn, Kyung-Ah and Kim, Dokyoon},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.08.24.746336}
}
```

Please also cite the dataset deposit (doi:10.7910/DVN/RDW4UL) and, when using
individual cohorts, the upstream sources listed in its terms of use.

## Reproducing the paper

Every experiment assumes `data/` (Dataverse download) and the tracked
`benchmark/` tree, and runs from the repository root. Open-weight backbones
use the paper configuration = the `*_tb8096` YAMLs (thinking-token budget
8096 with per-model `max_tokens` as set in each YAML, and each backbone's
recommended sampling); GPT-5.4 is greedy via `gpt_5_4.yaml`. The launchers
serve models with `vllm serve`, but every inference stage is a plain
OpenAI-compatible client — any endpoint exposing that API works if you
drive the stages manually (`OPENAI_BASE_URL`/`OPENAI_API_KEY`). Note that
`thinking_token_budget` is a vLLM extension; other servers may accept but
not enforce it.

**0 — Prompt prerequisites** (one-time, CPU only). The gate runners consume
pre-rendered `gate_task.json` + `gate_{ablation}/cells.md` per step:

```bash
python -m src.llm_gate.task --benchmark benchmark --data-dir data --n-workers 8
for s in benchmark/*/*/; do
    python -m src.llm_gate.prompt --batch "$s" --save_md --ablation full
done
```

LLM-C2S prerequisites (`c2s.json`, cluster prompts) are built per module —
see `src/llm/README.md` (density-grid) and `src/llm_flowsom/README.md`
(clustering). Setting-2 prompt prereqs are built by `build_all_prereqs.py`.

Prompts are rendered for **every** sample; the split is enforced at
inference time. The launchers do this via `SPLIT=test`; if you drive
`src.llm_gate.inference.run_openai` manually, pass the test samples
explicitly (`--sample $(python -m src.llm.scripts.list_split_samples
--splits benchmark/splits.json --dataset <DS> --split test)`) or you will
silently score train/val samples too.

**1 — Zero-shot panel evaluation (main table).**

```bash
# LLM-Gate, all open-weight backbones (local vLLM):
bash run_setting1_all_models.sh
bash run_propagate_all.sh && bash run_eval_all.sh

# LLM-Gate, GPT-5.4 (OpenAI API):
OPENAI_API_KEY=... bash run_setting1_openai.sh --config src/llm_gate/configs/models/gpt_5_4.yaml

# LLM-C2S density-grid / clustering (per model):
bash src/llm/scripts/run_vllm_grid.sh      --config src/llm/configs/models/qwen3_6_27b_tb8096.yaml hvp10 output/c2s_grid
bash src/llm_flowsom/scripts/run_vllm_flowsom.sh --config src/llm/configs/models/qwen3_6_27b_tb8096.yaml hvp10 output/c2s_clu

# Trained baselines, fit per cohort:
# UNITO: recipe in src/baselines/UNITO.md (reimplementation of KyleeCJ/UNITO)
# cyMAE: see src/baselines/README.md (pretrain → train_heads → inference)
```

**2 — Seed stability** (open-weight rows are 10-seed means; seeds 1–7, 42, 123, 456):

```bash
bash scripts/run_gate_seed_sweep.sh
bash scripts/run_c2s_seed_sweep.sh          # PARADIGMS="density_grid clustering" for both
python scripts/aggregate_seed_sweep.py --root results/seed_sweep --latex
```

**3 — Visual modality (VLM-Gate / Agent-Gate).**

```bash
bash src/vlm_gate/scripts/run_vlm_gate.sh     --config src/llm_gate/configs/models/gpt_5_4.yaml full results/vlm_gpt54
bash src/agent_gate/scripts/run_agent_gate.sh --config src/llm_gate/configs/models/gpt_5_4.yaml full results/agent_gpt54
```

**4 — LLM-Gate input ablation** (Qwen3.6-27B; −P = `no_pv`, −D = `no_desc`, −H = `no_hist`):

```bash
for abl in no_pv no_desc no_hist; do
    for s in benchmark/*/*/; do
        python -m src.llm_gate.prompt --batch "$s" --save_md --ablation $abl
    done
    SPLIT=test bash src/llm_gate/scripts/run_vllm_gate.sh \
        --config src/llm_gate/configs/models/qwen3_6_27b_tb8096.yaml \
        $abl "output/ablation_$abl"
done
```

**5 — Robustness under distribution shift (Setting 2).** 9 depletion + 9
calibration-drift scenarios, 43 + 52 curated tasks:

```bash
python build_all_prereqs.py                       # Setting-2 prompt prereqs
OPENAI_API_KEY=... bash run_hard_openai.sh --config src/llm_gate/configs/models/gpt_5_4.yaml --mode all
# open-weight equivalent: HARD_DEPLETION=<hid> (or HARD_SHIFT=<hid>) with run_vllm_gate.sh
# depletion magnitude sweep (ρ ∈ {1.0, 0.1, 0.01} of baseline prevalence):
python build_depletion_magnitude_prereqs.py
bash run_hard_openai.sh --config src/llm_gate/configs/models/gpt_5_4.yaml --mode depletion-magnitude
# UNITO / cyMAE under the same perturbations: same --hard-* flags at load time
# (UNITO recipe: src/baselines/UNITO.md)
```

**6 — Flat vs. hierarchical annotation** — see [`README_flat.md`](README_flat.md)
(FlowSOM K = 16 / 38 / 83 per plan depth, GPT-5.4 backbone).

## Quick start

```bash
DATASET=Acute2020

# 1) Preprocess: parquet → benchmark/{dataset}/{sample}/step_NN/task.json
python -m src.preprocess --data-dir data/ --output-dir benchmark/ --datasets $DATASET

# 2) C2S: parquet → c2s.json (see method-specific README)
python -m src.llm.c2s --benchmark benchmark/ --data-dir data/ --datasets $DATASET \
    --cluster-method flowsom --flowsom-grid 10 --flowsom-k 20

# 3) Prompt
for s in benchmark/$DATASET/*/; do
    python -m src.llm.c2s_prompt --batch "$s" --save_md --top-hvp-n 10
done

# 4) Inference
python -m src.llm.inference.run_openai \
    --dataset_path benchmark/$DATASET \
    --model gpt-5.4 --ablation_slug hvp10 \
    --output_path results/gpt-5.4_hvp10/$DATASET/pred.json

# 5) Propagate cluster preds → per-cell prediction.json
python -m src.llm.postprocess.propagate \
    --eval_path  results/gpt-5.4_hvp10/$DATASET/pred.json \
    --dataset    $DATASET \
    --benchmark  benchmark/ --data-dir data/ \
    --method     gpt-5.4_hvp10 \
    --output_dir results/gpt-5.4_hvp10

# 6) Score
python -m src.eval --predictions results/gpt-5.4_hvp10 \
    --benchmark benchmark/ --data-dir data/
```

Setting 2 (In-Panel Hard) end-to-end: pass the same `--hard-depletion <HID>` / `--hard-shift <HID>` (or `--perturbation` spec) to **every** stage. Write outputs to a separate results tree so perturbed runs don't overwrite the baseline.

### Preprocess details

`src.preprocess` reads `splits.json` (path overridable with `--splits`) and processes only the samples listed in each cohort's `train + val + test`. Parquet files not referenced by any split are silently skipped; `blacklist` is a second filter on top of the allow-list. If a cohort has no entry in `splits.json`, all parquet samples are processed (legacy behavior). Sample IDs must match parquet stems exactly — including spaces (e.g. `exp_390C_BM_027_T cells`); a `[WARN]` line is printed for any split entry without a matching parquet.

## Sweep launchers

Top-level shell scripts wrap the per-stage commands above into resumable
multi-cohort / multi-HID / multi-model sweeps. Pick by backend (local vLLM
vs OpenAI API) and setting (1 vs 2):

| Script | Backend | Setting | Notes |
|---|---|---|---|
| [`run_setting1_all_models.sh`](run_setting1_all_models.sh) | vLLM | 1 | Multi-model sweep on 11 cohorts × test split. Brings up vLLM per model, runs cohorts in parallel, tears down. **Stage 1 only** — propagate/eval deferred. Resumable per cohort. Override the model list via `MODELS_OVERRIDE` env. |
| [`run_setting1_openai.sh`](run_setting1_openai.sh) | OpenAI API | 1 | Same shape as above but hits an OpenAI-compatible endpoint via the model yaml's `openai_id`. Resumable per cohort. `.llm_done` marker per model. `--dry-run` validates orchestration without an API call. |
| [`run_hard_openai.sh`](run_hard_openai.sh) | OpenAI API | 2 | Hard sweep over `depletion` / `shift` / `depletion-magnitude`. Resumable per (HID, fraction, cohort). Per-spot prereq auto-skip avoids re-running `task.py` / `prompt.py` when `cells.md` already exists. `--dry-run` available. |
| [`build_all_prereqs.py`](build_all_prereqs.py) | — | 0 | 96-way parallel `cells.md` prebuild for every HID. Run once; subsequent sweeps skip prereq generation. |
| [`build_depletion_magnitude_prereqs.py`](build_depletion_magnitude_prereqs.py) | — | 0 | Same idea, scoped to the 138 `(HID, fraction)` magnitude combos. |
| [`run_propagate_all.sh`](run_propagate_all.sh) | — | 1 + 2 | Stage 2: walks `results/sweep/` and runs `src.llm_gate.postprocess.propagate` per cohort / HID. CPU-bound, no GPU needed. |
| [`run_eval_all.sh`](run_eval_all.sh) | — | 1 + 2 | Stage 3: per-step metrics. Handles the `_fracNN` slug split for hard-depletion magnitude. |

### Pipeline split (multi-stage architecture)

Each stage is a **separate script** so the slowest stage never starves the
others — important when LLM gen is the dominant cost (vLLM saturates GPU;
OpenAI saturates rate limits, and any inline CPU stage is API throughput
you never get back):

```
Stage 0  build_all_prereqs.py            → benchmark/<DS>/<sample>/step_NN/
                                            {gate_task.json, gate_<ABL>/cells.md}
Stage 1  run_setting1_*.sh /             → results/sweep/<setting>/<served>/.../pred.json
         run_hard_openai.sh
Stage 2  run_propagate_all.sh            → predictions/<DS>/<sample>/step_NN/<method>/prediction.json
Stage 3  run_eval_all.sh                 → predictions/eval_summary.json
```

### vLLM vs OpenAI

| You want | Use |
|---|---|
| Open-source models on local GPUs | `run_setting1_all_models.sh` (Setting 1) or `src/llm_gate/scripts/run_vllm_gate.sh` with `HARD_DEPLETION`/`HARD_SHIFT` (Setting 2) |
| GPT-5.4 / other API models | `run_setting1_openai.sh` (Setting 1) or `run_hard_openai.sh` (Setting 2) |

Setting 1 is identical work in both cases — only the inference call swaps.

### Resume + dry-run

All four sweep launchers are **resumable**: a re-run after a crash skips
every job whose `pred.json` already exists. Setting 1 launchers also drop a
per-model `.llm_done` marker so a wrapping multi-model loop can skip the
whole model, not just individual cohorts.

The OpenAI variants additionally support `--dry-run`: orchestration runs
end-to-end (config load, job list, prereq audit, sample resolution, output
path planning) but `run_openai` is **never invoked**, so no HTTP request
reaches OpenAI. Use this to validate a sweep config before paying for real
API calls.

```bash
# vLLM Setting 1 on the paper's open-weight roster
bash run_setting1_all_models.sh

# Same, only one model
MODELS_OVERRIDE="src/llm_gate/configs/models/qwen3_5_4b_tb8096.yaml|Qwen3.5-4B-tb8096" \
    bash run_setting1_all_models.sh

# OpenAI Setting 1 on GPT-5.4 (11 cohorts × test)
OPENAI_API_KEY=sk-... \
    bash run_setting1_openai.sh \
        --config src/llm_gate/configs/models/gpt_5_4.yaml

# Smoke test: 2 small cohorts only, dry-run first
bash run_setting1_openai.sh \
    --config src/llm_gate/configs/models/gpt_5_4.yaml \
    --cohorts FR-FCM-Z74D_hc Lyoplate_DC --dry-run

# OpenAI hard sweep — depletion + shift (95 HIDs)
OPENAI_API_KEY=sk-... \
    bash run_hard_openai.sh \
        --config src/llm_gate/configs/models/gpt_5_4.yaml --mode all

# Hard magnitude sweep (138 jobs), prereqs already built
PARALLEL_HIDS=12 CONCURRENCY=24 \
    bash run_hard_openai.sh \
        --config src/llm_gate/configs/models/gpt_5_4.yaml \
        --mode depletion-magnitude --no-prereq

# Stage 2 + 3 (run after the launcher finishes)
bash run_propagate_all.sh
bash run_eval_all.sh
```

The header of each script lists every env var (`PARALLEL_DS`, `CONCURRENCY`,
`SPLIT`, `ABLATION`, `OPENAI_BASE_URL`, ...) and explains the bottleneck it
mitigates.

## Pipeline-specific READMEs

Method-specific details live in their directories:

- **`src/llm/README.md`** — LLM-C2S pipeline (cluster cell-sentence → label; density-grid via `c2s_grid`)
- **`src/llm_flowsom/README.md`** — LLM-C2S clustering paradigm (FlowSOM cell-sentences; the "clustering" rows of the paper)
- **`src/llm_gate/README.md`** — LLM-Gate pipeline (axis-aligned gate boxes, text-only single-shot)
- **`src/vlm_gate/README.md`** — VLM-Gate pipeline (text + 2-D scatter image, single-shot)
- **`src/agent_gate/README.md`** — Agent-Gate pipeline (text + image + iterative `render_gate_overlay` tool loop)
- **`src/baselines/README.md`** — flowDensity (C2S density-grid backend), FlowSOM reference, cyMAE trained baseline
- **`src/baselines/UNITO.md`** — UNITO trained baseline recipe (per-cohort U-Net)
- **`README_flat.md`** — flat vs. hierarchical annotation pilot (`src/llm_gate_flat`, `src/flowsom_c2s`)

### LLM gating methods at a glance

The three gating modules — `llm_gate`, `vlm_gate`, `agent_gate` — emit the **byte-identical `pred.json` schema** and propagate to the same per-cell `prediction.json`, so all three are scored by the same `src.eval` and produce directly comparable F1 / Hull IoU numbers. They differ only in what the model sees per `(sample, step)`:

| | `src.llm_gate` | `src.vlm_gate` | `src.agent_gate` |
|---|---|---|---|
| Inference | one chat turn | one chat turn | tool-call loop |
| Model sees | text axis distribution + categories | text + decorated scatter image | text + decorated scatter + iterative overlay tool |
| Tools | — | — | `render_gate_overlay` (re-renders proposed gates on the bg) |
| Per-step latency / tokens | lowest | medium (one image) | highest (multi-turn + per-call image) |
| Use case | text-only baseline | "does the image alone help?" | "does iterative visual verify help?" |

Shared intermediate artefacts (paths and content):
- `cells.md` — text prompt, used by all three; produced by any of the three `prompt.py`.
- `gate_task.json` — task metadata; `vlm_gate` / `agent_gate` add `scatter_bg` + `scatter_extent` fields (a superset of `llm_gate`'s).
- `scatter_bg.png` — decoration-free background scatter (only needed for `vlm_gate` / `agent_gate`); produced by either of their `task.py`.

So if you've already run `agent_gate.task` for a sample, `vlm_gate` re-uses those artefacts as-is — no re-rendering. End-to-end recipes per module are in their own READMEs; the unified runners (`src/llm_gate/scripts/run_vllm_gate.sh` / `run_openai_gate.sh`, `src/vlm_gate/scripts/run_vlm_gate.sh`, `src/agent_gate/scripts/run_agent_gate.sh`) chain `task → prompt → inference → propagate → eval`.

For the **flat-annotation experiment setup** — `src/llm_gate_flat/`
(sequential cascading) and `src/flowsom_c2s/` (whole-cell FlowSom + LLM),
both scored by the shared `src/flat_eval.py`, with L1 / L2 / L3 plan
depths and `predicted_steps.parquet` as the common eval input — see
**[`README_flat.md`](README_flat.md)**.

## Project Structure

```
data/                              # Raw data (not tracked in git)
  {dataset}/
    parquet/
      meta.json
      {sample}.parquet
    gating_plan.json

src/
  preprocess.py                    # parquet → task.json + plots
  bench.py                         # BenchmarkLoader / TaskInput
  eval.py                          # method-agnostic evaluator (parquet GT)
  eval_hard.py                     # Setting-2 / depletion-tree evaluator
                                   #   walks {root}/{scenario}/{ds}_step{NN}/frac{NN}/...
                                   #   → eval_hard_summary.json + side-by-side plots
  hard_depletions.py               # Curated In-Panel Hard depletion loader
                                   #   HardDepletion dataclass + add_hard_depletion_args /
                                   #   resolve_hard_depletion_args helpers
  hard_shifts.py                   # Curated In-Panel Hard channel-shift loader
                                   #   HardShift dataclass + add_hard_shift_args /
                                   #   resolve_hard_shift_args helpers

  llm/                             # LLM-C2S pipeline      — see src/llm/README.md
  llm_flowsom/                     # LLM-C2S clustering paradigm (FlowSOM cell-sentences)
  flowsom_c2s/                     # whole-cell flat C2S   — see README_flat.md
  llm_gate_flat/                   # flat LLM-Gate cascade — see README_flat.md
  llm_gate/                        # LLM-Gate pipeline     — see src/llm_gate/README.md
  vlm_gate/                        # VLM-Gate pipeline     — see src/vlm_gate/README.md
                                   #   text + decorated scatter image, single-shot (no tools)
  agent_gate/                      # Agent-Gate pipeline   — see src/agent_gate/README.md
                                   #   text + image + iterative render_gate_overlay tool loop
  baselines/                       # flowDensity / FlowSOM / cyMAE — see src/baselines/README.md

scripts/                           # seed-sweep launchers + aggregation
case_study/                        # paper-figure scripts (need your own run outputs)

benchmark/                         # Generated tasks (tracked in git, PNGs excluded)
  {dataset}/{sample}/step_{NN}/
    task.json
    biaxial_plot.png
    biaxial_plot_gt.png
    c2s.json                       # C2S cluster cell sentences
    cell2cluster.npz
    c2s_scatter.png
    c2s_full/ c2s_hvp10/ ...       # per-ablation prompts
    gate_task.json                 # LLM-/VLM-/Agent-Gate task
                                   #   superset for vlm_gate / agent_gate adds
                                   #   scatter_bg + scatter_extent fields
    scatter_bg.png                 # vlm_gate / agent_gate: decoration-free
                                   #   bg used by render_gate_overlay (extent-aligned)
    gate_full/                     # gate prompt (cells.md) — shared by all 3 gate modules

    # Setting 2: every artifact suffixed with the perturbation slug
    c2s__{pslug}.json
    cell2cluster__{pslug}.npz
    c2s__{pslug}_scatter.png
    c2s_{ablation}__{pslug}/
  splits.json
  hard_depletion.yaml
  hard_shift.yaml

results/                           # Predictions + eval (not tracked in git)
  {method}/
    {dataset}/pred.json            # cluster-level (LLM pipelines only)
    {dataset}/{sample}/step_{NN}/
      prediction.json              # cell-level (n_parent labels)
    {dataset}/eval_summary.json    # src.eval output (per dataset)
    plots/{dataset}__{sample}.png  # src.eval --plot output (4 panels/step)

# Setting 2 (depletion fraction sweep) layout — consumed by src.eval_hard:
{depletion-root}/{scenario}/{dataset}_step{NN}/frac{NN}/{dataset}/
  pred.json                        # raw LLM gates (one per run)
  {sample}/step_{NN}/prediction.json
{depletion-root}/eval_hard_summary.json
{depletion-root}/plots/{scenario}/{dataset}_step{NN}__{sample}.png

logs/
```

## Requirements

```
numpy   pandas   pyarrow   scipy   scikit-learn   matplotlib
pyyaml   tqdm   openai   python-dotenv   minisom   torch   Pillow
```

Serving open-weight backbones additionally needs `vllm` (GPU box); the
LLM-C2S density-grid backend needs R with the `flowDensity` and
`data.table` packages. See [`requirements.txt`](requirements.txt).
