#!/usr/bin/env bash
# Rebuild only the steps whose rule needs >=2 valleys, re-run valley selection
# (top_depth / split_rescue / middle_dropped), re-post and re-evaluate.
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
DATASETS=(Acute2020 Acute2021 Vaccine FRDR_covid19 FR-FCM-Z74D_hc FR-FCM-Z74D_tissue)
DS="${DATASETS[$((LSB_JOBINDEX-1))]}"
OUT=results/flowdensity_rule
echo "[cohort] $DS"
.venv/bin/python -m src.baselines.flowdensity_rule.pre_rule --output-dir "$OUT" --only-multi-valley \
    --datasets "$DS" --split benchmark/splits.json --split-set test
module load R/4.4
Rscript src/baselines/flowdensity_rule/flowdensity_valleys.R --results-dir "$OUT/$DS" --cores "${LSB_DJOB_NUMPROC:-1}"
.venv/bin/python -m src.baselines.flowdensity_rule.post_rule --results-dir "$OUT/$DS" --reuse-existing
.venv/bin/python -m src.eval --predictions "$OUT" --benchmark benchmark --data-dir data \
    --datasets "$DS" --split benchmark/splits.json --split-set test
echo DONE
