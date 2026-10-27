#!/usr/bin/env bash
# LLM-Gate smoke test: definition-only notes, first test sample per cohort,
# gpt-5.4 + gpt-6-luna → results/gate_{model}_full${RESULT_SUFFIX}/ (RESULT_SUFFIX default empty)
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
MODELS=(gpt_5_4 gpt_6_luna)
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
CONCURRENCY="${CONCURRENCY:-16}"
RESULT_SUFFIX="${RESULT_SUFFIX:-}"

declare -A SAMPLE
for ds in "${DATASETS[@]}"; do
    SAMPLE[$ds]=$($PY -c "import json,sys; print(json.load(open('benchmark/splits.json'))['in_panel'][sys.argv[1]]['test'][0])" "$ds")
    echo "[sample] $ds → ${SAMPLE[$ds]}"
done
SAMPLES=(); for ds in "${DATASETS[@]}"; do SAMPLES+=("${SAMPLE[$ds]}"); done

echo "[1/4] gate_task.json"
$PY -m src.llm_gate.task --benchmark benchmark/ --data-dir data/ \
    --datasets "${DATASETS[@]}" --samples "${SAMPLES[@]}" --n-workers 4 --skip-existing

echo "[2/4] cells.md (note-mode=definition, ablation=full)"
for ds in "${DATASETS[@]}"; do
    $PY -m src.llm_gate.prompt --batch "benchmark/$ds/${SAMPLE[$ds]}" --save_md \
        --ablation full --note-mode definition > /dev/null
done

for served in gpt-5.4 gpt-6-luna; do
    d="results/gate_${served}_full${RESULT_SUFFIX}"
    [[ ! -e "$d" ]] || { echo "ERROR: $d already exists; refusing to overwrite" >&2; exit 2; }
    for ds in "${DATASETS[@]}"; do
        src="benchmark/$ds/${SAMPLE[$ds]}"
        (cd "$src" && find . -path '*gate_full/cells.md' -print0 | cpio -0pdm --quiet "$OLDPWD/$d/prompts/$ds/${SAMPLE[$ds]}")
    done
done

echo "[3/4] OpenAI calls"
pids=()
for cfg in "${MODELS[@]}"; do
    eval "$($PY -m src.llm_gate.scripts._config_to_shell --config src/llm_gate/configs/models/$cfg.yaml)"
    out_dir="results/gate_${SERVED_NAME}_full${RESULT_SUFFIX}"
    mkdir -p "$out_dir/logs"
    for ds in "${DATASETS[@]}"; do
        $PY -m src.llm_gate.inference.run_openai \
            --dataset_path "benchmark/$ds" --model "$SERVED_NAME" --ablation_slug full \
            --concurrency "$CONCURRENCY" --sample "${SAMPLE[$ds]}" --save-raw \
            --output_path "$out_dir/${ds}_${SERVED_NAME}_full__test.json" \
            "${OPENAI_ARGS[@]}" > "$out_dir/logs/${ds}_client.log" 2>&1 &
        pids+=($!)
    done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "  client failures: $fail"
(( fail == 0 )) || { echo "ERROR: $fail client run(s) failed; see results/gate_*_full/logs/" >&2; exit 1; }

echo "[4/4] propagate + eval"
for served in gpt-5.4 gpt-6-luna; do
    out_dir="results/gate_${served}_full${RESULT_SUFFIX}"
    preds=(); for ds in "${DATASETS[@]}"; do preds+=("$out_dir/${ds}_${served}_full__test.json"); done
    $PY -m src.llm_gate.postprocess.propagate --eval_path "${preds[@]}" --dataset "${DATASETS[@]}" \
        --benchmark benchmark --data-dir data --method "gate_${served}_full" \
        --output_dir "$out_dir/predictions" --tiebreak smallest > "$out_dir/logs/propagate.log" 2>&1
    $PY -m src.eval --predictions "$out_dir/predictions" --benchmark benchmark --data-dir data \
        --datasets "${DATASETS[@]}" --samples "${SAMPLES[@]}" > "$out_dir/logs/eval.log" 2>&1
    tail -40 "$out_dir/logs/eval.log"
done
echo DONE
