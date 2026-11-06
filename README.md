# CytoGate-Bench

CytoGate-Bench evaluates cytometry annotation at the expert-defined gating step. The final manuscript uses 11 flow- and mass-cytometry cohorts, a donor-stratified train/validation/test split (target 1:1:2, seed 42), and 11,671 held-out per-step test instances. Each step supplies its parent cells, two markers, candidate categories, and definition-only phenotype descriptions. A prediction assigns each parent cell to a candidate or `Unassigned`; evaluation compares it with the expert annotation on the same parent population.

This README describes the experiments in `../cytogate-bench-paper/patterns/word/main_revision.docx` and `../cytogate-bench-paper/patterns/word/supplemental_revision.docx`. The repository also retains implementation and run artifacts for development, but those are not part of the final protocol below.

## Data and task contract

- `data/<cohort>/parquet/<sample>.parquet`: transformed cell-by-channel matrix and expert `Step*` labels. The canonical row order is retained during prediction and evaluation.
- `data/<cohort>/parquet/meta.json`: channel-to-marker map, arcsinh cofactor, and category normalization.
- `benchmark/<cohort>/gating_plan.json`: expert tree, step order, marker pairs, parent expressions, and candidate categories.
- `benchmark/<cohort>/<sample>/step_<NN>/task.json`: one per-step task. The parent expression distinguishes cells outside the step from parent cells labeled `Unassigned`.
- `benchmark/splits.json`: donor-stratified split; only `test` is used for reported zero-shot scores.
- `benchmark/notes.yaml`: phenotype text. The reported LLM and VLM runs use the `definition` field only. GT-informed spatial `oracle` notes are excluded from these runs and appear only in the separate oracle-instruction reference comparison.

A missing phenotype definition is an error. The reported prompt does not take `note` or `tips` from a gating-plan JSON. Full prompt examples and output schemas are in the manuscript supplement.

## Installation

For the installation-tested Linux/Python 3.12.3 environments, use the [hash-pinned inference and FCS-conversion profiles](reproducibility/README.md#install-the-validated-environments). These profiles separate incompatible NumPy requirements and include clean-install and smoke-check results.

## Per-step methods in the paper

| Experiment | Implementation | Reported comparison |
|---|---|---|
| LLM-Gate | `src/llm_gate/` | One axis-aligned rectangle per candidate from marker histograms and phenotype definitions |
| LLM-C2S | `src/llm_flowsom/`, `src/llm/` | Cluster the current parent population and label clusters from cell sentences or density grids |
| VLM-Gate | `src/vlm_gate/` | Annotate from scatter, contour, or density images |
| Agent-Gate | `src/agent_gate/` | Inspect a rendered gate overlay and optionally revise it |
| UNITO and cyMAE | `src/baselines/` | Fit per cohort on train and validation labels, then score on test |
| Deterministic rule controls | `src/baselines/flowdensity/`, `src/baselines/flowdensity_rule/`, `src/baselines/rect_oracle/` | GT-informed region-to-label controls reported separately from zero-shot models |

The main per-step metrics are macro-F1, balanced accuracy, and convex-hull IoU. The evaluator treats `Unassigned` as a category and retains failed or unparsed steps in the denominator according to the method's output rule. The aggregate reported in the paper first averages over test steps within each cohort, then equally over 11 cohorts. The visual comparisons use matched valid rectangles for gate-area summaries.

## Robustness and protocol analyses

- Nine curated population-depletion scenarios at nominal, 10%, and 1% prevalence: `benchmark/hard_depletion.yaml`, `src/eval_hard.py`.
- Nine curated raw-signal calibration-drift scenarios: `benchmark/hard_shift.yaml`, `src/eval_hard.py`.
- GPT-6 Luna medium definition-only prompt-component and prompt-resolution analyses, GT-informed rule/proxy references, and three-seed open-weight stability are reported in the supplement. The source/panel/tree family comparison groups the 11 cohorts into five dependent families and is descriptive.
- The separate two-expert MDIPA comparison measures reference variation; it is not part of the 11-cohort test aggregate.

`src/bench.py` supplies the same parent population and GT extraction to methods. `src/eval.py` evaluates per-step predictions. Prediction files use `prediction.json.gz` (legacy uncompressed JSON remains readable) and must preserve the parent-cell row order.

## Flat versus cascade leaf annotation

The final Figure 5 compares Flat C2S with an LLM-Gate cascade using GPT-6 Sol medium and Qwen3.6-27B. The supplement also reports GPT-6 Luna medium. The three MDIPA/shared-tree cohorts (Acute2020, Acute2021, Vaccine) use L1/L2/L3 plans with 16/38/83 leaves. Eight other cohorts use their full native trees. There are 17 cohort-depth settings and 646 test sample-plan pairs.

- Plans live in `benchmark_flat/<cohort>/gating_plan_L1_coarse.json`, `gating_plan_L2_identity.json`, or `gating_plan.json`. Both methods use the same plan, test samples, and derived leaf vocabulary within a setting.
- Flat C2S (`src/flowsom_c2s/`) trains a 10×10 MiniSom on all arcsinh-transformed protein channels of a sample, using 30,000 randomly sampled training steps and no z-scoring. Average-linkage clustering collapses the 100 codebook nodes to **K equal to that plan's leaf count**. The LLM labels all metaclusters in one whole-sample call, returning one or more leaves per cluster. Every cell inherits its cluster's full leaf set.
- The cascade (`src/llm_gate_flat/`) walks the supplied tree, computing each later parent from its own earlier predictions. Empty predicted-parent branches are skipped. Each successful step records inference duration, token usage, and estimated API cost.
- `src/flowsom_c2s/eval.py` scores Flat C2S leaf sets directly. `src/flat_eval.py` derives cascade leaf sets from predicted Step columns. Both use the same GT leaf extraction and `MultiLabelBinarizer`; the primary Figure 5 metrics are multi-label F1 macro and micro. A single primary leaf is selected only for the supplemental per-cell visualizations.

See [README_flat.md](README_flat.md) for the artifact layout and exact final flat-run commands. The final Sol outputs are `results/flat_cascade_gpt-6-sol-medium/` and `results/flat_c2s_flowsom_gpt-6-sol-medium/`; Luna outputs use analogous names.

## Reproducibility

The [reproduction reference environment](reproducibility/README.md) provides hash-pinned inference and FCS-conversion environments based on the software captured on 6 November 2026. Both installed successfully in a fresh validation pod and passed dependency and basic runtime checks. The record also gives the GPU configuration and verified serving settings for the revised Qwen visual experiments, preserves the original package inventory, and identifies records that still need verification.

The revised Qwen visual runs were executed in a Kubernetes GPU environment. Preserved run manifests record model identity, generation settings, dates, and execution-code fingerprints. Some retained job templates contain LSF directives; these templates do not by themselves establish where a particular result was generated. Cluster-specific scripts, internal notes, and derived outputs remain local. The reproducibility record distinguishes the preserved execution code from the current public checkout and documents the work needed to assign a final immutable manuscript revision.
