#!/usr/bin/env bash
# Diagnosis follow-up: GT dump + propagate/eval of the variant runs.
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
SAMPLES=(); for ds in "${DATASETS[@]}"; do
    SAMPLES+=("$($PY -c "import json,sys; print(json.load(open('benchmark/splits.json'))['in_panel'][sys.argv[1]]['test'][0])" "$ds")")
done
$PY diagnosis/scripts/dump_gt.py > diagnosis/runs/dump_gt.log 2>&1 &
gt_pid=$!
for out in diagnosis/runs/*__diag_*; do
    name=$(basename "$out")
    preds=(); for ds in "${DATASETS[@]}"; do preds+=("$out/${ds}.json"); done
    $PY -m src.llm_gate.postprocess.propagate --eval_path "${preds[@]}" --dataset "${DATASETS[@]}" \
        --benchmark benchmark --data-dir data --method "$name" --output_dir "$out/predictions" \
        --tiebreak smallest > "$out/logs/propagate.log" 2>&1
    $PY -m src.eval --predictions "$out/predictions" --benchmark benchmark --data-dir data \
        --datasets "${DATASETS[@]}" --samples "${SAMPLES[@]}" > "$out/logs/eval.log" 2>&1
done
wait "$gt_pid" || { echo "ERROR: dump_gt failed (diagnosis/runs/dump_gt.log)" >&2; exit 1; }
echo DONE
