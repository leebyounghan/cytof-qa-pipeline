#!/usr/bin/env bash
# OpenAI-API agent-gate runner — model identity + sampling come from a
# per-model YAML in src/agent_gate/configs/models/ (model.openai_id set).
# Mirrors run_openai_gate.sh from llm_gate, with two differences:
#   - calls src.agent_gate.inference.run_agent (the tool-call loop)
#   - default CONCURRENCY lower (per-step makes multiple chat turns +
#     uploads images) — bump if your tier allows.
#
# Usage:
#     OPENAI_API_KEY=sk-... \
#         bash src/agent_gate/scripts/run_agent_gate.sh \
#              --config src/agent_gate/configs/models/gpt_5_4.yaml \
#              [ABLATION] [OUT_DIR] [DATASETS...]
#
# Examples:
#     # Default 12-cohort test split, full ablation
#     SPLIT=test OPENAI_API_KEY=sk-... \
#         bash src/agent_gate/scripts/run_agent_gate.sh \
#              --config src/agent_gate/configs/models/gpt_5_4.yaml \
#              full results/agent_gpt-5.4_full
#
#     # Subset, save the agent trace
#     SPLIT=test SAVE_RAW=1 OPENAI_API_KEY=sk-... \
#         bash src/agent_gate/scripts/run_agent_gate.sh \
#              --config src/agent_gate/configs/models/gpt_5_4.yaml \
#              full results/agent_gpt-5.4_full \
#              Acute2020 Bjornson
#
# Env overrides (orchestration only — model/generation come from YAML):
#     CONCURRENCY=10                 (per-client; lower than llm_gate)
#     PARALLEL_CLIENTS=4             Datasets in flight at once
#     SPLIT=test|val|train
#     BENCHMARK_DIR=benchmark
#     DATA_DIR=data
#     SKIP_PROPAGATE=0
#     SKIP_EVAL=0
#     TIEBREAK=smallest
#     METHOD_TAG=agent_<served>_<ablation>     (default; auto-derived)
#     SAVE_RAW=0                                1 → keep agent_trace
#     OPENAI_BASE_URL=               (set to override api.openai.com — e.g.
#                                     custom proxy; default unset)

set -euo pipefail

# ── CLI: --config <path> [POSITIONAL...] ───────────────────────────────────
CONFIG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="${2:?--config requires a path}"; shift 2 ;;
        --config=*) CONFIG="${1#*=}"; shift ;;
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
OUT_DIR="${2:-output/eval_agent_gate}"
shift 2 2>/dev/null || true
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(Acute2020 Acute2021 Bjornson Vaccine
              FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19
              Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
fi

# ── Load YAML → MODEL, SERVED_NAME, AGENT_ARGS[] ───────────────────────────
CONFIG_SHELL="$(uv run python -m src.agent_gate.scripts._config_to_shell --config "$CONFIG")" || {
    echo "ERROR: failed to parse $CONFIG (see stderr above)" >&2
    exit 2
}
eval "$CONFIG_SHELL"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set in the environment." >&2
    exit 2
fi

# ── Orchestration env (NOT in YAML — these change per experiment) ──────────
CONCURRENCY="${CONCURRENCY:-10}"
PARALLEL_CLIENTS="${PARALLEL_CLIENTS:-4}"
SPLIT="${SPLIT:-}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
DATA_DIR="${DATA_DIR:-data}"
SKIP_PROPAGATE="${SKIP_PROPAGATE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
TIEBREAK="${TIEBREAK:-smallest}"
METHOD_TAG="${METHOD_TAG:-agent_${SERVED_NAME}_${ABLATION}}"
SAVE_RAW="${SAVE_RAW:-0}"

mkdir -p "$OUT_DIR" /tmp/c2s_logs
SUFFIX_SPLIT=""
[[ -n "$SPLIT" ]] && SUFFIX_SPLIT="__${SPLIT}"

echo "=========================================================="
echo " agent-gate runner — OpenAI API tool-call loop"
echo "----------------------------------------------------------"
echo "  CONFIG         : $CONFIG"
echo "  MODEL          : $MODEL"
echo "  SERVED_NAME    : $SERVED_NAME"
echo "  BASE_URL       : ${OPENAI_BASE_URL:-(default api.openai.com)}"
echo "  AGENT_ARGS     : ${AGENT_ARGS[*]:-(empty → run_agent defaults)}"
echo "  CONCURRENCY    : $CONCURRENCY (per client)"
echo "  PARALLEL_CLIENTS: $PARALLEL_CLIENTS"
echo "  ABLATION       : $ABLATION"
echo "  OUT_DIR        : $OUT_DIR"
echo "  METHOD_TAG     : $METHOD_TAG"
echo "  TIEBREAK       : $TIEBREAK"
echo "  SAVE_RAW       : $SAVE_RAW"
echo "  SPLIT          : ${SPLIT:-(all)}"
echo "  DATASETS (${#DATASETS[@]}): ${DATASETS[*]}"
echo "=========================================================="

# ── Run clients ────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Running agentic clients (parallel groups of $PARALLEL_CLIENTS)"

run_dataset() {
    local ds=$1
    local out="$OUT_DIR/${ds}_${SERVED_NAME}_${ABLATION}${SUFFIX_SPLIT}.json"
    local extra_args=()
    if [[ -n "$SPLIT" ]]; then
        local list_log="/tmp/c2s_logs/agent_split_${ds}${SUFFIX_SPLIT}.log"
        if ! mapfile -t samples < <(
            uv run python -m src.llm.scripts.list_split_samples \
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
    echo "  [start] $ds → $out"
    uv run python -m src.agent_gate.inference.run_agent \
        --dataset_path "$BENCHMARK_DIR/$ds" \
        --model "$SERVED_NAME" \
        --ablation_slug "$ABLATION" \
        --concurrency "$CONCURRENCY" \
        --output_path "$out" \
        "${AGENT_ARGS[@]}" \
        "${extra_args[@]}" \
        > "/tmp/c2s_logs/agent_client_${ds}${SUFFIX_SPLIT}.log" 2>&1
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

# ── Propagate ─────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Propagating gates → prediction.json (method=$METHOD_TAG)"
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
uv run python -m src.agent_gate.postprocess.propagate \
    --eval_path "${EVAL_PATH_ARGS[@]}" \
    --dataset   "${DATASET_ARGS[@]}" \
    --benchmark "$BENCHMARK_DIR" \
    --data-dir  "$DATA_DIR" \
    --method    "$METHOD_TAG" \
    --output_dir "$PROP_DIR" \
    --tiebreak  "$TIEBREAK" \
    2>&1 | tee "/tmp/c2s_logs/agent_propagate_${SERVED_NAME}${SUFFIX_SPLIT}.log"

if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "  SKIP_EVAL=1 → stopping after prediction.json"
    exit $FAIL
fi

# ── Evaluate ──────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Eval (src.eval) → $PROP_DIR/eval_summary.json"
EVAL_DATASETS=("${DATASET_ARGS[@]}")
EVAL_EXTRA=()
[[ -n "$SPLIT" ]] && EVAL_EXTRA+=(--split "$BENCHMARK_DIR/splits.json" --split-set "$SPLIT")
uv run python -m src.eval \
    --predictions "$PROP_DIR" \
    --benchmark   "$BENCHMARK_DIR" \
    --data-dir    "$DATA_DIR" \
    --datasets    "${EVAL_DATASETS[@]}" \
    "${EVAL_EXTRA[@]}" \
    2>&1 | tee "/tmp/c2s_logs/agent_eval_${SERVED_NAME}${SUFFIX_SPLIT}.log"

echo ""
echo "[DONE] failures (clients): $FAIL"
echo "  pred.json   :  $OUT_DIR/<DATASET>_${SERVED_NAME}_${ABLATION}${SUFFIX_SPLIT}.json"
echo "  predictions :  $PROP_DIR/<DATASET>/<sample>/step_NN/$METHOD_TAG/prediction.json"
echo "  eval_summary:  $PROP_DIR/eval_summary.json (per-dataset)"
exit $FAIL
