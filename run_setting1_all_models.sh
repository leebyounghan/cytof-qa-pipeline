#!/usr/bin/env bash
# Setting 1 (11 cohorts × test split, NO perturbation) for every open-weight
# backbone in the paper roster (the closed-weight GPT-5.4 runs via
# run_setting1_openai.sh).
# - LLM gen + propagate per model. No eval (deferred to run_eval_all.sh).
# - Reuses existing vllm if currently up for the first model in the list.
# - Resumable: skips models with .propagate_done; skips datasets with pred.json.
#
# Throughput: PARALLEL_DS=12 (all cohorts at once) × CONCURRENCY=32 = 384 in-flight.
set -uo pipefail
cd "$(dirname "$0")"

PORT=${PORT:-8000}
ABLATION=full
PARALLEL_DS=${PARALLEL_DS:-12}
CONCURRENCY=${CONCURRENCY:-32}

ROOT=results/sweep
LOG_DIR="$ROOT/_logs/setting1_all"
mkdir -p "$LOG_DIR" /tmp/c2s_logs
MAIN_LOG="$LOG_DIR/main.log"
VLLM_PID_FILE="$LOG_DIR/vllm.pid"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"

# Lightest → heaviest. Qwen3.6-27B first because vllm may already be up.
# Override the model list by exporting MODELS_OVERRIDE as a newline-separated
# list of "config|served_name" entries (one per line).
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
    mapfile -t MODELS <<< "$MODELS_OVERRIDE"
else
    MODELS=(
        "src/llm_gate/configs/models/cell_o1.yaml|Cell-o1"
        "src/llm_gate/configs/models/qwen3_5_4b_tb8096.yaml|Qwen3.5-4B-tb8096"
        "src/llm_gate/configs/models/qwen3_5_27b_tb8096.yaml|Qwen3.5-27B-tb8096"
        "src/llm_gate/configs/models/qwen3_6_27b_tb8096.yaml|Qwen3.6-27B-tb8096"
        "src/llm_gate/configs/models/gemma_4_26b_a4b_it.yaml|Gemma4-26B-A4B-it"
        "src/llm_gate/configs/models/gemma_4_31b_it.yaml|Gemma4-31B-it"
    )
fi

DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)

cleanup() {
    log "[cleanup] tearing down vllm..."
    [[ -f "$VLLM_PID_FILE" ]] && kill "$(cat $VLLM_PID_FILE)" 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
}
trap cleanup EXIT INT TERM

bring_up_vllm() {
    local config=$1; local served_name=$2
    if curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then
        local current=$(curl -fs "http://localhost:$PORT/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "")
        if [[ "$current" == "$served_name" ]]; then
            log "[vllm] already up: $served_name (reuse)"
            return 0
        fi
        log "[vllm] tearing down existing ($current) before starting $served_name"
        pkill -f "vllm serve" 2>/dev/null || true
        sleep 8
    fi
    log "[vllm] starting $served_name"
    nohup bash src/llm_gate/scripts/run_vllm_gate.sh --config "$config" --serve-only \
        > "$LOG_DIR/vllm_${served_name}.log" 2>&1 &
    echo $! > "$VLLM_PID_FILE"
    local T0=$(date +%s)
    for i in $(seq 1 900); do
        if curl -fs "http://localhost:$PORT/health" >/dev/null 2>&1; then
            log "[vllm] ready (after $(($(date +%s) - T0))s) — $served_name"
            return 0
        fi
        sleep 2
    done
    log "[ERROR] vllm did not start within 1800s for $served_name"
    return 1
}

load_openai_args() {
    eval "$(python3 -m src.llm_gate.scripts._config_to_shell --config "$1")"
}

run_one_dataset() {
    local out_dir=$1; local served_name=$2; local ds=$3; shift 3
    local oa=("$@")
    local out="$out_dir/${ds}_${served_name}_${ABLATION}__test.json"
    local samples
    mapfile -t samples < <(python3 -m src.llm.scripts.list_split_samples \
        --splits benchmark/splits.json --dataset "$ds" --split test 2>/dev/null)
    [[ ${#samples[@]} -eq 0 ]] && { echo "[skip] $served_name/$ds — no test samples"; return 0; }
    echo "[start] $served_name/$ds (n=${#samples[@]})"
    if python3 -m src.llm_gate.inference.run_openai \
        --dataset_path "benchmark/$ds" --model "$served_name" --ablation_slug "$ABLATION" \
        --concurrency "$CONCURRENCY" --output_path "$out" \
        --sample "${samples[@]}" --save-raw \
        "${oa[@]}" \
        > "$LOG_DIR/${served_name}_${ds}.log" 2>&1; then
        echo "[done] $served_name/$ds"
    else
        echo "[ERROR] $served_name/$ds rc=$? (see $LOG_DIR/${served_name}_${ds}.log)"
    fi
}
export -f run_one_dataset
export ABLATION CONCURRENCY LOG_DIR OPENAI_BASE_URL OPENAI_API_KEY

run_setting1_for_model() {
    local config=$1; local served_name=$2
    local out_dir="$ROOT/setting1/${served_name}"
    mkdir -p "$out_dir"
    bring_up_vllm "$config" "$served_name" || return 1

    OPENAI_ARGS=()
    load_openai_args "$config"
    local oa=("${OPENAI_ARGS[@]}")
    log "[$served_name] OPENAI_ARGS: ${oa[*]}"
    log "[$served_name] running ${#DATASETS[@]} cohorts × test (PARALLEL_DS=$PARALLEL_DS, CONCURRENCY=$CONCURRENCY)"

    # Parallel cohorts via xargs
    printf '%s\n' "${DATASETS[@]}" | xargs -P "$PARALLEL_DS" -I {} bash -c '
        ds="{}"
        oa_str="'"${oa[*]}"'"
        # shellcheck disable=SC2206
        oa_arr=($oa_str)
        out=$(run_one_dataset "'"$out_dir"'" "'"$served_name"'" "$ds" "${oa_arr[@]}")
        echo "[$(date +%H:%M:%S)] $out" >> "'"$MAIN_LOG"'"
    '

    local n_done=0
    for ds in "${DATASETS[@]}"; do
        [[ -f "$out_dir/${ds}_${served_name}_${ABLATION}__test.json" ]] && ((n_done++))
    done
    log "[$served_name] LLM gen finished ($n_done/${#DATASETS[@]} cohorts)"
}

for entry in "${MODELS[@]}"; do
    config="${entry%|*}"
    served_name="${entry#*|}"
    log ""
    log "===================================================================="
    log "  Model: $served_name"
    log "===================================================================="
    run_setting1_for_model "$config" "$served_name" || log "[$served_name] FAILED, continuing"
done

log "all Setting 1 models processed"
