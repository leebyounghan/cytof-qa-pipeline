# Lyoplate Pipeline — GatingSetList → Parquet

> **Why this exists.** The four HIPC Lyoplate cohorts are distributed through
> ImmPort/ImmuneSpace under the NIAID-DAIT Data Use Agreement and are therefore
> *not* included in the CytoGate-Bench Dataverse deposit. This pipeline rebuilds
> their harmonized per-step parquet from the GatingSetList exports you obtain
> under the DUA. The `gating_plan.json` files here are byte-identical to the ones
> used to build the released benchmark, so the rebuilt cohorts slot straight
> into `benchmark/Lyoplate_*` (whose per-step `task.json`, `splits.json`, and
> `notes.yaml` entries are already tracked in this repository).

## Placing the output in this repository

After `run_pipeline.sh` finishes, copy each panel into the benchmark's `data/`
layout (panel → cohort name mapping: `bcell → Lyoplate_bcell`, `tcell →
Lyoplate_tcell`, `DC → Lyoplate_DC`, `treg → Lyoplate_treg`):

```bash
for p in bcell tcell DC treg; do
    mkdir -p data/Lyoplate_$p
    cp -r pipeline/$p/parquet        data/Lyoplate_$p/parquet
    cp    pipeline/$p/gating_plan.json data/Lyoplate_$p/gating_plan.json
done
```

`benchmark/Lyoplate_*/` is already built; the LLM-Gate / LLM-C2S / eval stages
only need `data/Lyoplate_*/parquet/` and `gating_plan.json` to be present.

Runnable pipeline that turns the HIPC Lyoplate GatingSetList exports into per-sample
Parquet tables (compensated expression + per-step gating labels), one set per panel.

This package contains **only the pipeline code, configs, and docs**. The raw input
data (`gslist-<panel>/`) is not included — see [Layout](#layout) for where it must sit.

## Layout

The scripts use relative paths and **must run from the Lyoplate data directory** — the
directory that holds the `gslist-<panel>/` folders. Drop this `pipeline/` folder in next
to them:

```
Lyoplate/                     <- run from here
├── gslist-bcell/             <- GatingSetList input (7 sites), REQUIRED
├── gslist-tcell/
├── gslist-DC/
├── gslist-treg/
└── pipeline/                 <- this package
    ├── run_pipeline.sh
    ├── extract_from_gs.R
    ├── convert_to_parquet.py
    ├── requirements.txt
    ├── install.R
    └── <panel>/              <- bcell, tcell, DC, treg
        ├── dataset_config.yaml
        ├── gating_plan.json
        ├── csv_tmp/          <- created by step 1 (intermediate CSVs)
        └── parquet/          <- created by step 2 (final output)
```

Panel names are **case-sensitive**: `bcell`, `tcell`, `DC`, `treg` (note `DC` is uppercase).

## Dependencies

Two runtimes are required: **R** drives step 1 (GatingSet → CSV) and **Python** drives
step 2 (CSV → Parquet). Versions below are what the pipeline was tested against; the
floors are conservative and newer releases work.

### System

| Runtime | Tested | Minimum | Notes |
|---------|--------|---------|-------|
| R | 4.4.2 | 4.4 | Bioconductor packages are tied to the R version (see below) |
| Python | 3.9.6 | 3.8 | standard CPython; no compiled extensions to build yourself |

### R packages (step 1 — `extract_from_gs.R`)

| Package | Source | Tested | Purpose |
|---------|--------|--------|---------|
| `flowWorkspace` | Bioconductor | 4.18.1 | load GatingSetList, legacy-format conversion (`convert_legacy_gs`) |
| `flowCore` | Bioconductor | 2.18.0 | flowFrame expression access (`exprs`, `parameters`) |
| `data.table` | CRAN | 1.16.2 | fast CSV read/write (`fwrite`) |

Install:
```bash
Rscript pipeline/install.R
# or manually inside R:
#   if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
#   BiocManager::install(c("flowWorkspace", "flowCore"))
#   install.packages("data.table")
```

> **Bioconductor ↔ R coupling**: `flowWorkspace` / `flowCore` are pinned to a Bioconductor
> release, which is pinned to an R minor version (R 4.4 → Bioc 3.20). If you run an older R,
> `BiocManager` installs the matching older package versions automatically — just use an R
> that BiocManager still supports.

### Python packages (step 2 — `convert_to_parquet.py`)

| Package | Tested | Floor | Purpose |
|---------|--------|-------|---------|
| `numpy` | 1.26.4 | 1.21 | array math, arcsinh / min-max transforms |
| `pandas` | 1.5.3 | 1.3 | DataFrame, `Categorical`, `read_parquet` / `to_parquet` |
| `pyarrow` | 21.0.0 | 6.0 | Parquet engine + `zstd` compression |
| `pyyaml` | 6.0.3 | 5.1 | parses `dataset_config.yaml` |

Install (a virtualenv is recommended):
```bash
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r pipeline/requirements.txt
```

## Run

```bash
cd /path/to/Lyoplate          # the dir containing gslist-bcell/ etc.

# Smoke test first — single panel, 2 samples
bash pipeline/run_pipeline.sh bcell --test 2

# All four panels, all samples
bash pipeline/run_pipeline.sh

# Single panel
bash pipeline/run_pipeline.sh tcell

# All panels, capped at N samples each
bash pipeline/run_pipeline.sh --test 5
```

### Individual steps

```bash
# Step 1: GatingSetList -> per-sample CSV (Live-filtered)
Rscript pipeline/extract_from_gs.R --panel bcell [--max_samples 2]

# Step 2: CSV -> Parquet
python3 pipeline/convert_to_parquet.py \
    --csv_dir     pipeline/bcell/csv_tmp \
    --out_dir     pipeline/bcell/parquet \
    --gating_plan pipeline/bcell/gating_plan.json \
    --config      pipeline/bcell/dataset_config.yaml \
    [--max_samples 2]
```

## Output (`pipeline/<panel>/parquet/`)

| File | Content |
|------|---------|
| `<sample>.parquet` | Wide table: marker columns (`float16`, transformed) + one categorical column per gating step |
| `meta.json` | Cohort metadata: cofactor, transform spec, channel→(marker, type) map, category map |
| `samples.csv` | Per-sample rows: `parquet_file`, `source_csv`, `sample_name`, `site`, `n_cells` |

Channel transforms (applied in step 2, per `dataset_config.yaml` marker types):
- **protein / livedead / dna / bead** — `arcsinh(x / cofactor)` (cofactor = 150)
- **scatter / tech** — clip to [p1, p99] then min-max to [0, 10]

Per-step label columns (e.g. `Step01_CD3|CD19`) are categorical; cells in a parent gate
that match no child gate are left as `NaN` (no derived negative category).

```python
import pandas as pd, json
df = pd.read_parquet("pipeline/tcell/parquet/<sample>.parquet")
meta = json.load(open("pipeline/tcell/parquet/meta.json"))
cd4 = df[df["Step02_CD4|CD8"] == "CD4"]     # CD4+ T cells
```

## Dataset Overview

HIPC (Human Immunology Project Consortium) Lyoplate standardized reference dataset.
SeraCare lyophilized PBMC controls measured across 7 sites with 4 staining panels.

| Item | Value |
|------|-------|
| Sites | 7 (anonymized) |
| Lots | 3 (12828, 1349, 1369) × 3 replicates per site |
| Panels | 4 (Bcell, Tcell, DC, Treg), 63 samples each |
| Gating | Automated via openCyto, stored as GatingSetList |

### Panels

| Panel | Markers | Gating Steps |
|-------|---------|-------------|
| **Bcell** | CD3, CD19, CD20, IgD, CD27, CD38, CD24 | 5 |
| **Tcell** | CD3, CD4, CD8, CD45RA, CCR7, HLA-DR, CD38 | 6 |
| **DC** | CD14, Lineage, CD11c, CD123, CD16, CD56, HLA-DR | 5 |
| **Treg** | CD3, CD4, CD25, CD127, CCR4, CD45RO, HLA-DR | 5 |

All panels include FSC-A, SSC-A scatter parameters. The viability (Live) channel is
included in expression but is not a gating step.

### Pre-filtering

The pipeline extracts **live cells only**. During step 1, cells are filtered to the
GatingSet "Live" population (`Lymphocytes/singlets/Live` or equivalent), removing debris,
doublets, and dead cells before any phenotypic annotation. Scatter and viability gates are
applied as pre-filters, not as annotation steps.

### Gating Hierarchies

**Bcell** (5 steps):
```
Live lymphocytes (pre-filtered)
└── Step 1: CD19+ / CD3+ / Other (CD3 vs CD19)
    └── CD19+
        └── Step 2: CD19+CD20+ / CD19+CD20- (CD3 vs CD20)
            ├── CD19+CD20+
            │   ├── Step 3: Naive / MemIgD+ / MemIgD- / DN (IgD vs CD27)
            │   └── Step 4: Transitional / Non-Transitional (CD38 vs CD24, independent of Step 3)
            └── CD19+CD20-
                └── Step 5: Plasmablasts (CD38 vs CD27)
```

**Tcell** (6 steps):
```
Live lymphocytes (pre-filtered)
└── Step 1: CD3+ / CD3- (CD3 vs SSC-A)
    └── CD3+
        └── Step 2: CD4 / CD8 / DNT / DPT (CD4 vs CD8)
            ├── CD4
            │   ├── Step 3: Naive / CM / EM / Effector (CCR7 vs CD45RA)
            │   └── Step 4: Activated / 38-DR+ / 38+DR- / 38-DR- (CD38 vs HLA-DR)
            └── CD8
                ├── Step 5: Naive / CM / EM / Effector (CCR7 vs CD45RA)
                └── Step 6: Activated / 38-DR+ / 38+DR- / 38-DR- (CD38 vs HLA-DR)
```

**DC** (5 steps):
```
Live monocytes (pre-filtered)
└── Step 1: Lin-CD14+ / Lin-CD14- (CD14 vs Lineage)
    ├── Lin-CD14+
    │   └── Step 2: CD14+CD16+ / CD14+CD16- (CD11c vs CD16)
    └── Lin-CD14-
        └── Step 3: CD16/CD56 quadrants (CD16 vs CD56)
            └── CD16-CD56-
                └── Step 4: HLA-DR+ / HLA-DR- (SSC-A vs HLA-DR)
                    └── HLA-DR+
                        └── Step 5: pDC / mDC / DP / DN (CD11c vs CD123)
```

**Treg** (5 steps):
```
Live lymphocytes (pre-filtered)
└── Step 1: CD3+ / CD3- (CD3 vs SSC-A)
    └── CD3+
        └── Step 2: CD4+ / CD4- (CD3 vs CD4)
            └── CD4+
                └── Step 3: Treg (CD127lo CD25hi) / Non-Treg (CD25 vs CD127)
                    └── Treg
                        ├── Step 4: CCR4/HLA-DR quadrants (CCR4 vs HLA-DR)
                        └── Step 5: CCR4/CD45RO quadrants (CCR4 vs CD45RO)
```

## Notes

- **Annotation convention**: `annotation categories` in `gating_plan.json` only include
  categories with an explicit gate in the source GatingSet. Cells in a parent gate that
  match no child gate are left as `NaN`, not labeled with a derived negative category.
- **GatingSetList, not WSP**: This dataset uses R/Bioconductor GatingSet objects (legacy
  format), not FlowJo WSP files. `extract_from_gs.R` handles legacy conversion via
  `convert_legacy_gs()` (cached under `/tmp/gs_lyoplate_<panel>_<site>`).
- **Automated gating**: Gates were generated by openCyto (flowClust, mindensity, cytokine
  methods), not manual FlowJo polygon gates.
- **Bcell Transitional**: The Transitional gate (CD38hi/CD24hi, Step 4) is on different
  axes than the IgD/CD27 subsets (Step 3), so the two overlap by design.
- **Tcell CD38/HLA-DR disambiguation**: The `38- DR+` / `38+ DR-` / `38- DR-` gate leaf
  names exist under both CD4 and CD8 branches; the converter resolves the correct column
  via parent hints (`__CD4__` / `__CD8__`).
- **Treg boolean gates**: 4 boolean reference gates (Total Treg, Memory, Naive, Activated)
  have broken node references in the legacy GatingSet. They are combinations of polygon
  gates that ARE extracted, so no annotation is lost.
- **Live path variation**: One treg site uses `/Lymphocytes/Live` instead of
  `/Lymphocytes/singlets/Live`; the R script handles both patterns.
