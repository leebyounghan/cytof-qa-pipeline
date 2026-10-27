#!/usr/bin/env bash
# Score result folders against the rect-1 oracle labels ("rect GT"), full test split.
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
DIRS=(gate_gpt-6-luna-none_full gate_gpt-6-luna-low_full gate_gpt-6-luna-medium_full gate_gpt-6-luna-high_full
      gate_gpt-6-sol-medium_full c2s_flowdensity_gpt-5.4 c2s_flowdensity_gpt-6-luna c2s_flowsom_gpt-5.4 c2s_flowsom_gpt-6-luna)
pids=()
for d in "${DIRS[@]}"; do
    .venv/bin/python -m src.eval --predictions "results/$d/predictions" --gt-labels results/rect1_oracle \
        --benchmark benchmark --data-dir data --datasets "${DATASETS[@]}" \
        --split benchmark/splits.json --split-set test > "results/$d/logs/eval_vs_rect1.log" 2>&1 &
    pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
(( fail == 0 )) || { echo "ERROR: $fail eval runs failed" >&2; exit 1; }
echo DONE
