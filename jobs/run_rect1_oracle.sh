#!/usr/bin/env bash
# rect-1 oracle fit + eval, one cohort per array index (or DS=..., SAMPLES=...).
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
DS="${DS:-${DATASETS[$((LSB_JOBINDEX-1))]}}"
OUT="${OUT:-results/rect1_oracle}"
SAMPLE_ARGS=()
[[ -n "${SAMPLES:-}" ]] && SAMPLE_ARGS=(--samples "$SAMPLES")
echo "[cohort] $DS → $OUT"
.venv/bin/python -m src.baselines.rect_oracle.fit_rect1 --output-dir "$OUT" \
    --datasets "$DS" --split benchmark/splits.json --split-set test "${SAMPLE_ARGS[@]}"
.venv/bin/python -m src.eval --predictions "$OUT" --benchmark benchmark --data-dir data \
    --datasets "$DS" --split benchmark/splits.json --split-set test "${SAMPLE_ARGS[@]}"
echo DONE
