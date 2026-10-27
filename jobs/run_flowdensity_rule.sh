#!/usr/bin/env bash
# flowDensity valleys + notes.yaml rule (no LLM), one cohort per array index.
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
DS="${DS:-${DATASETS[$((LSB_JOBINDEX-1))]}}"
SAMPLE_ARGS=()
[[ -n "${SAMPLES:-}" ]] && SAMPLE_ARGS=(--samples "$SAMPLES")
OUT=results/flowdensity_rule
PY=.venv/bin/python
CORES="${LSB_DJOB_NUMPROC:-1}"
echo "[cohort] $DS  cores=$CORES"

$PY -m src.baselines.flowdensity_rule.pre_rule --output-dir "$OUT" \
    --datasets "$DS" --split benchmark/splits.json --split-set test "${SAMPLE_ARGS[@]}"
module load R/4.4
Rscript src/baselines/flowdensity_rule/flowdensity_valleys.R --results-dir "$OUT/$DS" --cores "$CORES"
$PY -m src.baselines.flowdensity_rule.post_rule --results-dir "$OUT/$DS"
$PY -m src.eval --predictions "$OUT" --benchmark benchmark --data-dir data \
    --datasets "$DS" --split benchmark/splits.json --split-set test "${SAMPLE_ARGS[@]}"
echo DONE
