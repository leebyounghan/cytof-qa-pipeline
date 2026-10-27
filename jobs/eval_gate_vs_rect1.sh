#!/usr/bin/env bash
# Score existing LLM-Gate predictions against the rect-1 oracle labels (part B).
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
SAMPLES=()
for ds in "${DATASETS[@]}"; do
    SAMPLES+=("$(.venv/bin/python -c "import json,sys; print(json.load(open('benchmark/splits.json'))['in_panel'][sys.argv[1]]['test'][0])" "$ds")")
done
# sanity: the oracle against itself must score 1.0
.venv/bin/python -m src.eval --predictions results/rect1_oracle --gt-labels results/rect1_oracle \
    --benchmark benchmark --data-dir data --datasets Bjornson --samples "${SAMPLES[2]}"
for m in gpt-5.4 gpt-6-luna; do
    .venv/bin/python -m src.eval --predictions "results/gate_${m}_full/predictions" --gt-labels results/rect1_oracle \
        --benchmark benchmark --data-dir data --datasets "${DATASETS[@]}" --samples "${SAMPLES[@]}"
done
echo DONE
