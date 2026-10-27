#!/usr/bin/env bash
# LLM-C2S, definition-only notes, hvp10 prompts, gpt-5.4 (temperature 0) + gpt-6-luna (1.0).
#   density-grid (flowDensity thresholds) → results/c2s_flowdensity_{model}/
#   clustering   (FlowSOM 10x10, k=20)    → results/c2s_flowsom_{model}/
# SCOPE=smoke (first test sample per cohort, default) | test (full test split)
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
SCOPE="${SCOPE:-smoke}"
WORKERS="${LSB_DJOB_NUMPROC:-4}"
CONCURRENCY="${CONCURRENCY:-8}"
ABL=hvp10
THR=src/llm/flowdensity_result
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
CYTOF=" Acute2020 Acute2021 Bjornson Vaccine "
declare -A SAMPLES_OF
for ds in "${DATASETS[@]}"; do
    if [[ "$SCOPE" == smoke ]]; then
        SAMPLES_OF[$ds]=$($PY -c "import json,sys; print(json.load(open('benchmark/splits.json'))['in_panel'][sys.argv[1]]['test'][0])" "$ds")
    else
        SAMPLES_OF[$ds]=$($PY -m src.llm.scripts.list_split_samples --splits benchmark/splits.json --dataset "$ds" --split test)
    fi
done
samples_of() { printf '%s\n' "${SAMPLES_OF[$1]}"; }
ALL_SAMPLES=(); for ds in "${DATASETS[@]}"; do mapfile -t s < <(samples_of "$ds"); ALL_SAMPLES+=("${s[@]}"); done
echo "scope=$SCOPE samples=${#ALL_SAMPLES[@]} workers=$WORKERS"

for paradigm in flowdensity flowsom; do
    for served in gpt-5.4 gpt-6-luna; do
        d="results/c2s_${paradigm}_${served}"
        [[ ! -e "$d" ]] || { echo "ERROR: $d exists; refusing to overwrite" >&2; exit 2; }
    done
done

echo "[1/4] density-grid prereqs: flowDensity thresholds → c2s_grid.json → c2s_den_$ABL/cells.md"
$PY -m src.baselines.flowdensity.pre_flowdensity --output-dir "$THR" --datasets "${DATASETS[@]}" --samples "${ALL_SAMPLES[@]}"
module load R/4.4
Rscript src/baselines/flowdensity/flowdensity_gate.R --results-dir "$THR" --cores "$WORKERS"
$PY -m src.llm.c2s_grid --benchmark benchmark --data-dir data --datasets "${DATASETS[@]}" --samples "${ALL_SAMPLES[@]}" \
    --thresholds-dir "$THR" --modality auto --n-workers "$WORKERS" --no-plots
for ds in "${DATASETS[@]}"; do
    samples_of "$ds" | while IFS= read -r s; do find "benchmark/$ds/$s" -name c2s_grid.json -print0; done
done | xargs -0 -P "$WORKERS" -I {} $PY -m src.llm.c2s_prompt {} --save_md --top-hvp-n 10 --prompt-subdir "c2s_den_$ABL" > /dev/null

echo "[2/4] clustering prereqs: FlowSOM c2s.json → c2s_clu_$ABL/cells.md"
for ds in "${DATASETS[@]}"; do
    mod=flow; [[ "$CYTOF" == *" $ds "* ]] && mod=cytof
    mapfile -t s < <(samples_of "$ds")
    $PY -m src.llm_flowsom.c2s --benchmark benchmark --data-dir data --datasets "$ds" --samples "${s[@]}" \
        --cluster-method flowsom --flowsom-grid 10 --flowsom-k 20 --modality "$mod" --n-workers "$WORKERS" --no-plots
done
for ds in "${DATASETS[@]}"; do
    samples_of "$ds" | while IFS= read -r s; do printf '%s\0' "benchmark/$ds/$s"; done
done | xargs -0 -P "$WORKERS" -I {} $PY -m src.llm_flowsom.c2s_prompt --batch {} --save_md --top-hvp-n 10 --prompt-subdir "c2s_clu_$ABL" > /dev/null

n_task=0; n_den=0; n_clu=0
for ds in "${DATASETS[@]}"; do
    while IFS= read -r s; do
        n_task=$((n_task + $(find "benchmark/$ds/$s" -name task.json | wc -l)))
        n_den=$((n_den + $(find "benchmark/$ds/$s" -path "*c2s_den_$ABL/cells.md" | wc -l)))
        n_clu=$((n_clu + $(find "benchmark/$ds/$s" -path "*c2s_clu_$ABL/cells.md" | wc -l)))
    done < <(samples_of "$ds")
done
echo "  task.json $n_task | c2s_den_$ABL prompts $n_den | c2s_clu_$ABL prompts $n_clu"

echo "[3/4] OpenAI calls"
pids=()
for cfg in gpt_5_4 gpt_6_luna; do
    eval "$($PY -m src.llm.scripts._config_to_shell --config src/llm/configs/models/$cfg.yaml)"
    for paradigm in flowdensity flowsom; do
        out="results/c2s_${paradigm}_${SERVED_NAME}"; mkdir -p "$out/logs"
        for ds in "${DATASETS[@]}"; do
            mapfile -t s < <(samples_of "$ds")
            if [[ $paradigm == flowdensity ]]; then
                mod=src.llm.inference.run_openai; extra=(--ablation_slug "den_$ABL" --c2s-filename c2s_grid.json)
            else
                mod=src.llm_flowsom.inference.run_openai; extra=(--ablation_slug "clu_$ABL")
            fi
            $PY -m $mod --dataset_path "benchmark/$ds" --model "$SERVED_NAME" "${extra[@]}" \
                --concurrency "$CONCURRENCY" --sample "${s[@]}" \
                --output_path "$out/${ds}_${SERVED_NAME}_${ABL}.json" "${OPENAI_ARGS[@]}" \
                > "$out/logs/${ds}_client.log" 2>&1 &
            pids+=($!)
        done
    done
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
if (( fail )); then
    echo "ERROR: $fail client runs failed:" >&2; grep -l "ERROR\|Traceback" results/c2s_*/logs/*_client.log >&2 || true; exit 1
fi

echo "[4/4] propagate + eval"
EVAL_SCOPE=(--samples "${ALL_SAMPLES[@]}")
[[ "$SCOPE" == test ]] && EVAL_SCOPE=(--split benchmark/splits.json --split-set test)
for served in gpt-5.4 gpt-6-luna; do
    for paradigm in flowdensity flowsom; do
        out="results/c2s_${paradigm}_${served}"
        preds=(); for ds in "${DATASETS[@]}"; do preds+=("$out/${ds}_${served}_${ABL}.json"); done
        if [[ $paradigm == flowdensity ]]; then
            $PY -m src.llm.postprocess.propagate --eval_path "${preds[@]}" --dataset "${DATASETS[@]}" \
                --benchmark benchmark --data-dir data --method "c2s_grid_${served}_${ABL}" \
                --output_dir "$out/predictions" --mapping-filename cell2grid.npz > "$out/logs/propagate.log" 2>&1
        else
            $PY -m src.llm_flowsom.postprocess.propagate --eval_path "${preds[@]}" --dataset "${DATASETS[@]}" \
                --benchmark benchmark --data-dir data --method "c2s_flowsom_${served}_${ABL}" \
                --output_dir "$out/predictions" > "$out/logs/propagate.log" 2>&1
        fi
        $PY -m src.eval --predictions "$out/predictions" --benchmark benchmark --data-dir data \
            --datasets "${DATASETS[@]}" "${EVAL_SCOPE[@]}" > "$out/logs/eval.log" 2>&1
        cp benchmark/notes.yaml "$out/notes.yaml.used"
    done
done
echo DONE
