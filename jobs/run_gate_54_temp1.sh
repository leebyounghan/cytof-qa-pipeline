#!/usr/bin/env bash
# gpt-5.4 at temperature 1.0, current notes.yaml definitions, first test sample
# per cohort → results/gate_gpt-5.4_full_v3_temp1/
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
OUT=results/gate_gpt-5.4_full_v3_temp1
[[ ! -e "$OUT" ]] || { echo "ERROR: $OUT exists; refusing to overwrite" >&2; exit 2; }
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
declare -A SAMPLE
for ds in "${DATASETS[@]}"; do
    SAMPLE[$ds]=$($PY -c "import json,sys; print(json.load(open('benchmark/splits.json'))['in_panel'][sys.argv[1]]['test'][0])" "$ds")
done
SAMPLES=(); for ds in "${DATASETS[@]}"; do SAMPLES+=("${SAMPLE[$ds]}"); done

echo "[1] cells.md from current notes.yaml (definition)"
for ds in "${DATASETS[@]}"; do
    $PY -m src.llm_gate.prompt --batch "benchmark/$ds/${SAMPLE[$ds]}" --save_md --ablation full --note-mode definition > /dev/null
    (cd "benchmark/$ds/${SAMPLE[$ds]}" && find . -path '*gate_full/cells.md' -print0 | cpio -0pdm --quiet "$OLDPWD/$OUT/prompts/$ds/${SAMPLE[$ds]}")
done

echo "[2] OpenAI calls (gpt-5.4, temperature 1.0)"
eval "$($PY -m src.llm_gate.scripts._config_to_shell --config src/llm_gate/configs/models/gpt_5_4.yaml)"
mkdir -p "$OUT/logs"
pids=()
for ds in "${DATASETS[@]}"; do
    $PY -m src.llm_gate.inference.run_openai --dataset_path "benchmark/$ds" --model "$SERVED_NAME" \
        --ablation_slug full --concurrency 16 --sample "${SAMPLE[$ds]}" --save-raw \
        --output_path "$OUT/${ds}_${SERVED_NAME}_full__test.json" "${OPENAI_ARGS[@]}" --temperature 1.0 \
        > "$OUT/logs/${ds}_client.log" 2>&1 &
    pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
(( fail == 0 )) || { echo "ERROR: $fail client runs failed" >&2; exit 1; }

echo "[3] propagate + eval"
preds=(); for ds in "${DATASETS[@]}"; do preds+=("$OUT/${ds}_${SERVED_NAME}_full__test.json"); done
$PY -m src.llm_gate.postprocess.propagate --eval_path "${preds[@]}" --dataset "${DATASETS[@]}" \
    --benchmark benchmark --data-dir data --method gate_gpt-5.4_full_v3_temp1 \
    --output_dir "$OUT/predictions" --tiebreak smallest > "$OUT/logs/propagate.log" 2>&1
$PY -m src.eval --predictions "$OUT/predictions" --benchmark benchmark --data-dir data \
    --datasets "${DATASETS[@]}" --samples "${SAMPLES[@]}" > "$OUT/logs/eval.log" 2>&1
echo DONE
