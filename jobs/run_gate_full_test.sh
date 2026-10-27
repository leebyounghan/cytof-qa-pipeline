#!/usr/bin/env bash
# LLM-Gate, full test split, current notes.yaml definitions, temperature 0 (gpt-5.4)
# and 1.0 (gpt-6-luna, the only value it accepts) → results/gate_{model}_full/
# STAGE=prereq | api | all (default all). api expects prereq to have run.
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
STAGE="${STAGE:-all}"
WORKERS="${LSB_DJOB_NUMPROC:-4}"
CONCURRENCY="${CONCURRENCY:-8}"
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)

if [[ "$STAGE" == prereq || "$STAGE" == all ]]; then
    echo "[prereq 1/2] gate_task.json (test split, skip existing) workers=$WORKERS"
    $PY -m src.llm_gate.task --benchmark benchmark/ --data-dir data/ --datasets "${DATASETS[@]}" \
        --split benchmark/splits.json --split-set test --skip-existing --n-workers "$WORKERS"
    echo "[prereq 2/2] cells.md (definition) for every test sample"
    for ds in "${DATASETS[@]}"; do
        $PY -m src.llm.scripts.list_split_samples --splits benchmark/splits.json --dataset "$ds" --split test \
            | while IFS= read -r s; do printf '%s\0' "benchmark/$ds/$s"; done
    done | xargs -0 -P "$WORKERS" -I {} $PY -m src.llm_gate.prompt --batch {} --save_md --ablation full --note-mode definition > /dev/null
    n_task=0; n_md=0
    for ds in "${DATASETS[@]}"; do
        while IFS= read -r s; do
            n_task=$((n_task + $(find "benchmark/$ds/$s" -name gate_task.json | wc -l)))
            n_md=$((n_md + $(find "benchmark/$ds/$s" -path '*gate_full/cells.md' | wc -l)))
        done < <($PY -m src.llm.scripts.list_split_samples --splits benchmark/splits.json --dataset "$ds" --split test)
    done
    echo "  test gate_task.json: $n_task   gate_full/cells.md: $n_md"
    (( n_task == n_md )) || { echo "ERROR: prompt count != task count" >&2; exit 1; }
fi

if [[ "$STAGE" == api || "$STAGE" == all ]]; then
    for cfg in gpt_5_4 gpt_6_luna; do
        eval "$($PY -m src.llm_gate.scripts._config_to_shell --config src/llm_gate/configs/models/$cfg.yaml)"
        out="results/gate_${SERVED_NAME}_full"
        [[ ! -e "$out" ]] || { echo "ERROR: $out exists; refusing to overwrite" >&2; exit 2; }
        mkdir -p "$out/logs"
        cp benchmark/notes.yaml "$out/notes.yaml.used"
    done
    pids=()
    for cfg in gpt_5_4 gpt_6_luna; do
        eval "$($PY -m src.llm_gate.scripts._config_to_shell --config src/llm_gate/configs/models/$cfg.yaml)"
        out="results/gate_${SERVED_NAME}_full"
        for ds in "${DATASETS[@]}"; do
            mapfile -t samples < <($PY -m src.llm.scripts.list_split_samples --splits benchmark/splits.json --dataset "$ds" --split test)
            $PY -m src.llm_gate.inference.run_openai --dataset_path "benchmark/$ds" --model "$SERVED_NAME" \
                --ablation_slug full --concurrency "$CONCURRENCY" --sample "${samples[@]}" --save-raw \
                --output_path "$out/${ds}_${SERVED_NAME}_full__test.json" "${OPENAI_ARGS[@]}" \
                > "$out/logs/${ds}_client.log" 2>&1 &
            pids+=($!)
        done
    done
    fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
    if (( fail )); then
        echo "ERROR: $fail client runs reported failures:" >&2
        grep -l "ERROR:" results/gate_*_full/logs/*_client.log >&2 || true
        exit 1
    fi
    for served in gpt-5.4 gpt-6-luna; do
        out="results/gate_${served}_full"
        preds=(); for ds in "${DATASETS[@]}"; do preds+=("$out/${ds}_${served}_full__test.json"); done
        $PY -m src.llm_gate.postprocess.propagate --eval_path "${preds[@]}" --dataset "${DATASETS[@]}" \
            --benchmark benchmark --data-dir data --method "gate_${served}_full" --output_dir "$out/predictions" \
            --tiebreak smallest > "$out/logs/propagate.log" 2>&1
        $PY -m src.eval --predictions "$out/predictions" --benchmark benchmark --data-dir data \
            --datasets "${DATASETS[@]}" --split benchmark/splits.json --split-set test > "$out/logs/eval.log" 2>&1
    done
fi
echo DONE
