#!/usr/bin/env bash
# Diagnosis: [Description]-only prompt variants on the smoke-test samples + GT dump.
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
declare -A SAMPLE
for ds in "${DATASETS[@]}"; do
    SAMPLE[$ds]=$($PY -c "import json,sys; print(json.load(open('benchmark/splits.json'))['in_panel'][sys.argv[1]]['test'][0])" "$ds")
done
SAMPLES=(); for ds in "${DATASETS[@]}"; do SAMPLES+=("${SAMPLE[$ds]}"); done

echo "[1] build prompt variants"
$PY diagnosis/scripts/build_variants.py

echo "[2] API runs"
RUNS=("gpt_5_4 diag_oldnote" "gpt_5_4 diag_defdet" "gpt_6_luna diag_oldnote")
pids=()
for r in "${RUNS[@]}"; do
    read -r cfg slug <<< "$r"
    eval "$($PY -m src.llm_gate.scripts._config_to_shell --config src/llm_gate/configs/models/$cfg.yaml)"
    out="diagnosis/runs/${SERVED_NAME}__${slug}"; mkdir -p "$out/logs"
    for ds in "${DATASETS[@]}"; do
        $PY -m src.llm_gate.inference.run_openai --dataset_path "benchmark/$ds" --model "$SERVED_NAME" \
            --ablation_slug "$slug" --concurrency 16 --sample "${SAMPLE[$ds]}" --save-raw \
            --output_path "$out/${ds}.json" "${OPENAI_ARGS[@]}" > "$out/logs/${ds}_client.log" 2>&1 &
        pids+=($!)
    done
done
$PY diagnosis/scripts/dump_gt.py > diagnosis/runs/dump_gt.log 2>&1 &
gt_pid=$!
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
wait "$gt_pid" || { echo "ERROR: dump_gt failed (diagnosis/runs/dump_gt.log)" >&2; exit 1; }
(( fail == 0 )) || { echo "ERROR: $fail client runs failed" >&2; exit 1; }

echo "[3] propagate + eval"
for out in diagnosis/runs/*__diag_*; do
    name=$(basename "$out")
    preds=(); for ds in "${DATASETS[@]}"; do preds+=("$out/${ds}.json"); done
    $PY -m src.llm_gate.postprocess.propagate --eval_path "${preds[@]}" --dataset "${DATASETS[@]}" \
        --benchmark benchmark --data-dir data --method "$name" --output_dir "$out/predictions" \
        --tiebreak smallest > "$out/logs/propagate.log" 2>&1
    $PY -m src.eval --predictions "$out/predictions" --benchmark benchmark --data-dir data \
        --datasets "${DATASETS[@]}" --samples "${SAMPLES[@]}" > "$out/logs/eval.log" 2>&1
done
echo DONE
