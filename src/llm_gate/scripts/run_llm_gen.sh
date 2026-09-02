#!/usr/bin/env bash
# LLM generation only — starts vLLM server, runs inference, stops server.
# No propagation or eval.
#
# Usage:
#   SPLIT=test bash src/llm_gate/scripts/run_llm_gen.sh \
#     --config src/llm_gate/configs/models/qwen3_6_27b_tb8096.yaml \
#     [ABLATION] [OUT_DIR] [DATASETS...]
#
# Env overrides:
#   SPLIT=test|val|train  CONCURRENCY=64  PARALLEL_CLIENTS=4
#   READY_TIMEOUT=900  PORT=8000  BENCHMARK_DIR=benchmark
set -euo pipefail

# ── CLI ──────────────────────────────────────────────────────────────────
CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="${2:?--config requires a path}"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
[[ -z "$CONFIG" ]] && { echo "ERROR: --config <path> required" >&2; exit 2; }
[[ ! -f "$CONFIG" ]] && { echo "ERROR: config not found: $CONFIG" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ABLATION="${1:-full}"
OUT_DIR="${2:-output/eval_gate}"
shift 2 2>/dev/null || true
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(Acute2020 Acute2021 Bjornson Vaccine
              FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19
              Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
fi

CONFIG_SHELL="$(python3 -m src.llm_gate.scripts._config_to_shell --config "$CONFIG")"
eval "$CONFIG_SHELL"

SPLIT="${SPLIT:-}"
CONCURRENCY="${CONCURRENCY:-64}"
PARALLEL_CLIENTS="${PARALLEL_CLIENTS:-4}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
PORT="${PORT:-8000}"

mkdir -p "$OUT_DIR" /tmp/c2s_logs

SUFFIX_SPLIT=""
[[ -n "$SPLIT" ]] && SUFFIX_SPLIT="__${SPLIT}"

echo "=========================================================="
echo " LLM Generation only"
echo "  CONFIG: $CONFIG"
echo "  MODEL: $MODEL"
echo "  SERVED: $SERVED_NAME"
echo "  ABLATION: $ABLATION"
echo "  SPLIT: ${SPLIT:-(all)}"
echo "  DATASETS (${#DATASETS[@]}): ${DATASETS[*]}"
echo "=========================================================="

# --- Start server ---
SERVER_LOG="/tmp/c2s_logs/vllm_server_${SERVED_NAME}.log"
echo "[1/2] Starting vLLM server → $SERVER_LOG"
vllm serve "$MODEL" --port "$PORT" "${VLLM_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup_vllm() {
    set +e  # disable errexit inside cleanup
    echo "[cleanup] stopping vllm PID=$SERVER_PID and all children..."
    kill -TERM "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 15); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[cleanup] SIGKILL on PID=$SERVER_PID"
        kill -KILL "$SERVER_PID" 2>/dev/null
    fi
    # Kill any remaining GPU processes
    for gpid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        kill -KILL "$gpid" 2>/dev/null
    done
    wait "$SERVER_PID" 2>/dev/null
    echo "[cleanup] done"
    set -e
}
trap cleanup_vllm EXIT INT TERM

echo "  PID=$SERVER_PID, waiting for /health (timeout ${READY_TIMEOUT}s)..."
for _ in $(seq 1 "$READY_TIMEOUT"); do
    curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[ERROR] server died"; tail -30 "$SERVER_LOG"; exit 1; }
    sleep 1
done
curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 || { echo "[ERROR] timeout"; exit 1; }
echo "  ready!"

export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"

# --- Inference ---
echo ""
echo "[2/2] Running inference (parallel groups of $PARALLEL_CLIENTS)"

run_dataset() {
    local ds=$1
    local out="$OUT_DIR/${ds}_${SERVED_NAME}_${ABLATION}${SUFFIX_SPLIT}.json"
    local extra_args=()
    if [[ -n "$SPLIT" ]]; then
        local list_log="/tmp/c2s_logs/gen_split_${ds}${SUFFIX_SPLIT}.log"
        local samples
        if ! mapfile -t samples < <(
            python3 -m src.llm.scripts.list_split_samples \
                --splits "$BENCHMARK_DIR/splits.json" --dataset "$ds" --split "$SPLIT" 2> "$list_log"
        ); then
            echo "  [ERROR] $ds — list_split_samples failed"
            return 1
        fi
        [[ ${#samples[@]} -eq 0 ]] && { echo "  [skip] $ds — no samples"; return 0; }
        extra_args+=(--sample "${samples[@]}")
    fi
    echo "  [start] $ds"
    python3 -m src.llm_gate.inference.run_openai \
        --dataset_path "$BENCHMARK_DIR/$ds" \
        --model "$SERVED_NAME" \
        --ablation_slug "$ABLATION" \
        --concurrency "$CONCURRENCY" \
        --output_path "$out" \
        "${OPENAI_ARGS[@]}" \
        "${extra_args[@]}" \
        > "/tmp/c2s_logs/gen_${SERVED_NAME}_${ds}${SUFFIX_SPLIT}.log" 2>&1
    local rc=$?
    echo "  [done]  $ds (rc=$rc)"
    return $rc
}

T0=$(date +%s)
PIDS=()
FAIL=0
for ds in "${DATASETS[@]}"; do
    run_dataset "$ds" &
    PIDS+=($!)
    if [[ ${#PIDS[@]} -ge $PARALLEL_CLIENTS ]]; then
        if ! wait "${PIDS[0]}"; then FAIL=$((FAIL+1)); fi
        PIDS=("${PIDS[@]:1}")
    fi
done
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAIL=$((FAIL+1)); fi
done
T1=$(date +%s)
echo ""
echo "[DONE] $((T1-T0))s, failures: $FAIL"
echo "  pred.json → $OUT_DIR/"
