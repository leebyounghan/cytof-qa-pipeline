#!/usr/bin/env bash
# LLM-Gate, full test split, an OpenAI reasoning model at API reasoning_effort settings
# → results/gate_{model}-{effort}_full/. CONFIG (default gpt_6_luna) and EFFORTS select the runs. Reuses benchmark gate_full prompts
# (built from the current notes.yaml by run_gate_full_test.sh STAGE=prereq).
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
EFFORTS=(${EFFORTS:-none low medium high})
CONFIG="${CONFIG:-gpt_6_luna}"
CONCURRENCY="${CONCURRENCY:-8}"
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
cmp -s benchmark/notes.yaml results/gate_gpt-6-luna_full/notes.yaml.used || {
    echo "ERROR: notes.yaml changed since the gate_full prompts were built; rebuild them first" >&2; exit 2; }
eval "$($PY -m src.llm_gate.scripts._config_to_shell --config src/llm_gate/configs/models/$CONFIG.yaml)"
for e in "${EFFORTS[@]}"; do
    out="results/gate_${SERVED_NAME}-${e}_full"
    [[ ! -e "$out" ]] || { echo "ERROR: $out exists; refusing to overwrite" >&2; exit 2; }
    mkdir -p "$out/logs"; cp benchmark/notes.yaml "$out/notes.yaml.used"
done
pids=()
for e in "${EFFORTS[@]}"; do
    out="results/gate_${SERVED_NAME}-${e}_full"
    for ds in "${DATASETS[@]}"; do
        mapfile -t samples < <($PY -m src.llm.scripts.list_split_samples --splits benchmark/splits.json --dataset "$ds" --split test)
        $PY -m src.llm_gate.inference.run_openai --dataset_path "benchmark/$ds" --model "$SERVED_NAME" \
            --ablation_slug full --concurrency "$CONCURRENCY" --sample "${samples[@]}" --save-raw \
            --api-reasoning-effort "$e" --output_path "$out/${ds}_${SERVED_NAME}-${e}_full__test.json" \
            "${OPENAI_ARGS[@]}" > "$out/logs/${ds}_client.log" 2>&1 &
        pids+=($!)
    done
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
if (( fail )); then
    echo "ERROR: $fail client runs failed:" >&2; grep -l "ERROR" results/gate_${SERVED_NAME}-*_full/logs/*_client.log >&2 || true; exit 1
fi
pids=()
for e in "${EFFORTS[@]}"; do
    (
        out="results/gate_${SERVED_NAME}-${e}_full"
        preds=(); for ds in "${DATASETS[@]}"; do preds+=("$out/${ds}_${SERVED_NAME}-${e}_full__test.json"); done
        $PY -m src.llm_gate.postprocess.propagate --eval_path "${preds[@]}" --dataset "${DATASETS[@]}" \
            --benchmark benchmark --data-dir data --method "gate_${SERVED_NAME}-${e}_full" \
            --output_dir "$out/predictions" --tiebreak smallest > "$out/logs/propagate.log" 2>&1
        $PY -m src.eval --predictions "$out/predictions" --benchmark benchmark --data-dir data \
            --datasets "${DATASETS[@]}" --split benchmark/splits.json --split-set test > "$out/logs/eval.log" 2>&1
    ) &
    pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
(( fail == 0 )) || { echo "ERROR: $fail propagate/eval runs failed" >&2; exit 1; }
echo DONE
