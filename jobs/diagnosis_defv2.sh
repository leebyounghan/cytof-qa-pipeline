#!/usr/bin/env bash
# Revised definitions (no details): rerun gpt-5.4 + gpt-6-luna on the changed steps only.
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
$PY diagnosis/scripts/build_defv2.py
mapfile -t DATASETS < <($PY -c "import csv; print('\n'.join(sorted({r['dataset'] for r in csv.DictReader(open('diagnosis/tables/defv2_steps.csv'))})))")
declare -A SAMPLE
for ds in "${DATASETS[@]}"; do
    SAMPLE[$ds]=$($PY -c "import json,sys; print(json.load(open('benchmark/splits.json'))['in_panel'][sys.argv[1]]['test'][0])" "$ds")
done
SAMPLES=(); for ds in "${DATASETS[@]}"; do SAMPLES+=("${SAMPLE[$ds]}"); done
pids=()
for cfg in gpt_5_4 gpt_6_luna; do
    eval "$($PY -m src.llm_gate.scripts._config_to_shell --config src/llm_gate/configs/models/$cfg.yaml)"
    out="diagnosis/runs/${SERVED_NAME}__diag_defv2"; mkdir -p "$out/logs"
    for ds in "${DATASETS[@]}"; do
        $PY -m src.llm_gate.inference.run_openai --dataset_path "benchmark/$ds" --model "$SERVED_NAME" \
            --ablation_slug diag_defv2 --concurrency 16 --sample "${SAMPLE[$ds]}" --save-raw \
            --output_path "$out/${ds}.json" "${OPENAI_ARGS[@]}" > "$out/logs/${ds}_client.log" 2>&1 &
        pids+=($!)
    done
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
(( fail == 0 )) || { echo "ERROR: $fail client runs failed" >&2; exit 1; }
for out in diagnosis/runs/*__diag_defv2; do
    name=$(basename "$out")
    preds=(); for ds in "${DATASETS[@]}"; do preds+=("$out/${ds}.json"); done
    $PY -m src.llm_gate.postprocess.propagate --eval_path "${preds[@]}" --dataset "${DATASETS[@]}" \
        --benchmark benchmark --data-dir data --method "$name" --output_dir "$out/predictions" \
        --tiebreak smallest > "$out/logs/propagate.log" 2>&1
    $PY -m src.eval --predictions "$out/predictions" --benchmark benchmark --data-dir data \
        --datasets "${DATASETS[@]}" --samples "${SAMPLES[@]}" > "$out/logs/eval.log" 2>&1
done
echo DONE
