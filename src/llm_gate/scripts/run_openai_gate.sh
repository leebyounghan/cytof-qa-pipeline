#!/usr/bin/env bash
# Unified vLLM LLM-Gate runner — model identity, server flags, and
# generation/sampling parameters all come from a per-model YAML in
# src/llm_gate/configs/models/. Run-orchestration knobs (split, dataset
# selection, concurrency, output dir, propagate/eval skips, etc.) stay
# as positional args / env overrides since they change per experiment.
#
# Usage:
#     bash src/llm_gate/scripts/run_vllm_gate.sh \
#          --config src/llm_gate/configs/models/qwen3_6_27b_tb8096.yaml \
#          [ABLATION] [OUT_DIR] [DATASETS...]
#
# Examples:
#     SPLIT=test \
#         bash src/llm_gate/scripts/run_vllm_gate.sh \
#              --config src/llm_gate/configs/models/qwen3_6_27b_tb8096.yaml \
#              full output/eval_gate_qwen3_6_27b
#
#     SPLIT=test \
#         bash src/llm_gate/scripts/run_vllm_gate.sh \
#              --config src/llm_gate/configs/models/gemma_4_31b_it.yaml \
#              full output/eval_gate_deepseek \
#              Acute2020 Bjornson
#
# Env overrides (orchestration only — model/server/generation come from YAML):
#     PORT=8000
#     CONCURRENCY=64
#     PARALLEL_CLIENTS=4
#     READY_TIMEOUT=900
#     SPLIT=test|val|train
#     BENCHMARK_DIR=benchmark
#     DATA_DIR=data
#     SKIP_PROPAGATE=0
#     SKIP_EVAL=0
#     TIEBREAK=smallest
#     METHOD_TAG=gate_<served>_<ablation>   (default; auto-derived)
#     SAVE_RAW=0
#     THINKING=                              (on|off; if set, overrides YAML)
#
# Serve-only mode (interactive debug, no clients/eval):
#     bash src/llm_gate/scripts/run_vllm_gate.sh \
#          --config src/llm_gate/configs/models/gemma_4_31b_it.yaml \
#          --serve-only

set -euo pipefail

# ── CLI: --config <path> [--serve-only] [POSITIONAL...] ────────────────────
CONFIG=""
SERVE_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="${2:?--config requires a path}"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
        --serve-only) SERVE_ONLY=1; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done
if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config <path/to/model.yaml> is required" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

ABLATION="${1:-full}"
OUT_DIR="${2:-output/eval_gate}"
shift 2 2>/dev/null || true
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(Acute2020 Acute2021 Bjornson
              FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
              Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg
              Vaccine FRDR_covid19)
fi

# ── Load YAML → MODEL, SERVED_NAME, VLLM_ARGS[], OPENAI_ARGS[], server env ──
CONFIG_SHELL="$(python3 -m src.llm_gate.scripts._config_to_shell --config "$CONFIG")" || {
    echo "ERROR: failed to parse $CONFIG (see stderr above)" >&2
    exit 2
}
eval "$CONFIG_SHELL"

# ── Orchestration env (NOT in YAML — these change per experiment) ──────────
PORT="${PORT:-8000}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# ── Sanity: an OpenAI-API config (model.openai_id) is for run_openai_gate.sh,
# not this vllm-serve runner. Bail early with a clear message instead of
# letting `vllm serve gpt-5.4` fail at health-check.
if [[ "${OPENAI_MODE:-0}" == "1" ]]; then
    echo "ERROR: $CONFIG is an OpenAI API config (model.openai_id set)." >&2
    echo "       Use run_openai_gate.sh for OpenAI models." >&2
    exit 2
fi

# ── Serve-only short-circuit: foreground exec, no clients / propagate / eval ──
if [[ "$SERVE_ONLY" == "1" ]]; then
    echo "=========================================================="
    echo " vLLM serve-only (foreground, Ctrl-C to stop)"
    echo "----------------------------------------------------------"
    echo "  CONFIG      : $CONFIG"
    echo "  MODEL       : $MODEL"
    echo "  SERVED_NAME : $SERVED_NAME"
    echo "  PORT        : $PORT"
    echo "  VLLM_ARGS   : ${VLLM_ARGS[*]}"
    echo "=========================================================="
    exec vllm serve "$MODEL" --port "$PORT" "${VLLM_ARGS[@]}"
fi

CONCURRENCY="${CONCURRENCY:-64}"
PARALLEL_CLIENTS="${PARALLEL_CLIENTS:-4}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
SPLIT="${SPLIT:-}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
DATA_DIR="${DATA_DIR:-data}"
SKIP_PROPAGATE="${SKIP_PROPAGATE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
TIEBREAK="${TIEBREAK:-smallest}"
METHOD_TAG="${METHOD_TAG:-gate_${SERVED_NAME}_${ABLATION}}"
SAVE_RAW="${SAVE_RAW:-0}"
THINKING_OVERRIDE="${THINKING:-}"

# Setting 2 (In-Panel Hard) passthrough. When HARD_DEPLETION or HARD_SHIFT
# is set, the same flag is appended to every stage (run_openai, propagate)
# and src.eval is replaced with src.eval_hard. The hard YAML files
# (benchmark/hard_depletion.yaml, benchmark/hard_shift.yaml) provide the
# spec; only the hard_id is needed here.
HARD_DEPLETION="${HARD_DEPLETION:-}"
HARD_SHIFT="${HARD_SHIFT:-}"
if [[ -n "$HARD_DEPLETION" && -n "$HARD_SHIFT" ]]; then
    echo "ERROR: HARD_DEPLETION and HARD_SHIFT are mutually exclusive" >&2
    exit 2
fi
HARD_FLAG_RUNNER=()
HARD_FLAG_PROP=()
HARD_FLAG_EVAL=()
HARD_MODE=0
if [[ -n "$HARD_DEPLETION" ]]; then
    HARD_FLAG_RUNNER=(--hard-depletion "$HARD_DEPLETION")
    HARD_FLAG_PROP=(--hard-depletion "$HARD_DEPLETION")
    HARD_FLAG_EVAL=(--hard-depletion "$HARD_DEPLETION")
    HARD_MODE=1
fi
if [[ -n "$HARD_SHIFT" ]]; then
    HARD_FLAG_RUNNER=(--hard-shift "$HARD_SHIFT")
    HARD_FLAG_PROP=(--hard-shift "$HARD_SHIFT")
    HARD_FLAG_EVAL=(--hard-shift "$HARD_SHIFT")
    HARD_MODE=1
fi

mkdir -p "$OUT_DIR" /tmp/c2s_logs
SUFFIX_SPLIT=""
[[ -n "$SPLIT" ]] && SUFFIX_SPLIT="__${SPLIT}"

echo "=========================================================="
echo " vLLM LLM-Gate runner"
echo "----------------------------------------------------------"
echo "  CONFIG         : $CONFIG"
echo "  MODEL          : $MODEL"
echo "  SERVED_NAME    : $SERVED_NAME"
echo "  PORT           : $PORT"
echo "  VLLM_ARGS      : ${VLLM_ARGS[*]}"
echo "  OPENAI_ARGS    : ${OPENAI_ARGS[*]:-(empty → run_openai defaults)}"
echo "  CONCURRENCY    : $CONCURRENCY (per client)"
echo "  PARALLEL_CLIENTS: $PARALLEL_CLIENTS"
echo "  ABLATION       : $ABLATION"
echo "  OUT_DIR        : $OUT_DIR"
echo "  METHOD_TAG     : $METHOD_TAG"
echo "  TIEBREAK       : $TIEBREAK"
echo "  SAVE_RAW       : $SAVE_RAW"
echo "  THINKING(override): ${THINKING_OVERRIDE:-(YAML default)}"
echo "  SPLIT          : ${SPLIT:-(all)}"
echo "  DATASETS (${#DATASETS[@]}): ${DATASETS[*]}"
echo "=========================================================="

# ── Start vLLM server ──────────────────────────────────────────────────────
SERVER_LOG="/tmp/c2s_logs/vllm_server_${SERVED_NAME}.log"
echo "[1/4] Starting vllm serve → $SERVER_LOG"
vllm serve "$MODEL" \
    --port "$PORT" \
    "${VLLM_ARGS[@]}" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[cleanup] stopping vllm serve PID=$SERVER_PID ..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "  server PID=$SERVER_PID, waiting for /health (timeout ${READY_TIMEOUT}s) ..."
T_READY_0=$(date +%s)
ready=0
for _ in $(seq 1 "$READY_TIMEOUT"); do
    if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
        ready=1; break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ERROR] server died during startup. Last 50 log lines:"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done
T_READY_1=$(date +%s)
if [[ $ready -ne 1 ]]; then
    echo "[ERROR] server did not become ready within ${READY_TIMEOUT}s."
    tail -50 "$SERVER_LOG"
    exit 1
fi
echo "  ready (after $((T_READY_1 - T_READY_0))s)"

# ── Run clients ────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Running clients (parallel groups of $PARALLEL_CLIENTS)"
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="EMPTY"

run_dataset() {
    local ds=$1
    local out="$OUT_DIR/${ds}_${SERVED_NAME}_${ABLATION}${SUFFIX_SPLIT}.json"
    local extra_args=()
    if [[ -n "${SAMPLES:-}" ]]; then
        # Explicit sample list override (space- or comma-separated). Applied
        # to every dataset in DATASETS — useful for smoke tests on one sample.
        local samples
        IFS=', ' read -r -a samples <<< "$SAMPLES"
        extra_args+=(--sample "${samples[@]}")
    elif [[ -n "$SPLIT" ]]; then
        local list_log="/tmp/c2s_logs/gate_split_${ds}${SUFFIX_SPLIT}.log"
        if ! mapfile -t samples < <(
            python3 -m src.llm.scripts.list_split_samples \
                --splits "$BENCHMARK_DIR/splits.json" \
                --dataset "$ds" --split "$SPLIT" 2> "$list_log"
        ); then
            echo "  [ERROR] $ds — list_split_samples failed (see $list_log)"
            sed -n '1,5p' "$list_log" | sed 's/^/      /' || true
            return 1
        fi
        if [[ ${#samples[@]} -eq 0 ]]; then
            echo "  [skip]  $ds — no samples in split=$SPLIT"
            return 0
        fi
        extra_args+=(--sample "${samples[@]}")
    fi
    [[ "$SAVE_RAW" == "1" ]] && extra_args+=(--save-raw)
    [[ -n "$THINKING_OVERRIDE" ]] && extra_args+=(--thinking "$THINKING_OVERRIDE")
    [[ ${#HARD_FLAG_RUNNER[@]} -gt 0 ]] && extra_args+=("${HARD_FLAG_RUNNER[@]}")
    echo "  [start] $ds → $out"
    python3 -m src.llm_gate.inference.run_openai \
        --dataset_path "$BENCHMARK_DIR/$ds" \
        --model "$SERVED_NAME" \
        --ablation_slug "$ABLATION" \
        --concurrency "$CONCURRENCY" \
        --output_path "$out" \
        "${OPENAI_ARGS[@]}" \
        "${extra_args[@]}" \
        > "/tmp/c2s_logs/gate_client_${ds}${SUFFIX_SPLIT}.log" 2>&1
    local rc=$?
    echo "  [done]  $ds (rc=$rc)"
    return $rc
}

T0=$(date +%s)
PIDS=()
FAIL=0
PRED_PATHS=()
DS_LABELS=()
for ds in "${DATASETS[@]}"; do
    PRED_PATHS+=("$OUT_DIR/${ds}_${SERVED_NAME}_${ABLATION}${SUFFIX_SPLIT}.json")
    DS_LABELS+=("$ds")
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
echo "  All clients done in $((T1 - T0))s (failures: $FAIL)"

if [[ "$SKIP_PROPAGATE" == "1" ]]; then
    echo "  SKIP_PROPAGATE=1 → stopping after pred.json"
    exit $FAIL
fi

echo ""
echo "  Releasing GPUs (server SIGTERM) before propagate ..."
cleanup
trap - EXIT INT TERM

# ── Propagate ─────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Propagating gates → prediction.json (method=$METHOD_TAG)"
PROP_DIR="$OUT_DIR/predictions"
mkdir -p "$PROP_DIR"
EVAL_PATH_ARGS=()
DATASET_ARGS=()
for i in "${!PRED_PATHS[@]}"; do
    pp="${PRED_PATHS[$i]}"
    if [[ -f "$pp" ]]; then
        EVAL_PATH_ARGS+=("$pp")
        DATASET_ARGS+=("${DS_LABELS[$i]}")
    else
        echo "  [skip] missing pred.json for ${DS_LABELS[$i]} ($pp)"
    fi
done
if [[ ${#EVAL_PATH_ARGS[@]} -eq 0 ]]; then
    echo "  no pred.json to propagate"
    exit $FAIL
fi
python3 -m src.llm_gate.postprocess.propagate \
    --eval_path "${EVAL_PATH_ARGS[@]}" \
    --dataset   "${DATASET_ARGS[@]}" \
    --benchmark "$BENCHMARK_DIR" \
    --data-dir  "$DATA_DIR" \
    --method    "$METHOD_TAG" \
    --output_dir "$PROP_DIR" \
    --tiebreak  "$TIEBREAK" \
    "${HARD_FLAG_PROP[@]}" \
    2>&1 | tee "/tmp/c2s_logs/gate_propagate_${SERVED_NAME}${SUFFIX_SPLIT}.log"

if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "  SKIP_EVAL=1 → stopping after prediction.json"
    exit $FAIL
fi

# ── Evaluate ──────────────────────────────────────────────────────────────
# Always use src.eval. For Setting 2 (--hard-* flag set) the dataset list
# is auto-inferred from the hard_id, so --datasets must be omitted (the
# two flags are mutually exclusive at the BenchmarkLoader CLI layer).
# src.eval_hard is a separate driver for cross-magnitude sweeps and is
# not invoked here.
echo ""
echo "[4/4] Eval (src.eval) → $PROP_DIR/eval_summary.json"
EVAL_DATASETS=("${DATASET_ARGS[@]}")
EVAL_EXTRA=()
[[ -n "$SPLIT" ]] && EVAL_EXTRA+=(--split "$BENCHMARK_DIR/splits.json" --split-set "$SPLIT")
EVAL_DS_ARGS=()
if [[ "$HARD_MODE" != "1" ]]; then
    EVAL_DS_ARGS+=(--datasets "${EVAL_DATASETS[@]}")
fi
python3 -m src.eval \
    --predictions "$PROP_DIR" \
    --benchmark   "$BENCHMARK_DIR" \
    --data-dir    "$DATA_DIR" \
    "${EVAL_DS_ARGS[@]}" \
    "${EVAL_EXTRA[@]}" \
    "${HARD_FLAG_EVAL[@]}" \
    2>&1 | tee "/tmp/c2s_logs/gate_eval_${SERVED_NAME}${SUFFIX_SPLIT}.log"

echo ""
echo "[DONE] failures (clients): $FAIL"
echo "  pred.json   :  $OUT_DIR/<DATASET>_${SERVED_NAME}_${ABLATION}${SUFFIX_SPLIT}.json"
echo "  predictions :  $PROP_DIR/<DATASET>/<sample>/step_NN/$METHOD_TAG/prediction.json"
echo "  eval_summary:  $PROP_DIR/eval_summary.json (per-dataset)"
exit $FAIL
