#!/usr/bin/env bash
# OpenAI-API LLM density-grid C2S runner — model identity + sampling
# come from a per-model YAML in src/llm/configs/models/ (model.openai_id).
# Mirrors run_openai.sh (clustering paradigm) but drives the density-grid
# paradigm (per-step 2D density-binned grid cells from flowDensity
# thresholds), matching run_vllm_grid.sh on the open-source side.
#
# Difference from run_openai.sh (clustering):
#   --c2s-filename c2s_grid.json           (instead of default c2s.json)
#   --ablation_slug den_<ABLATION>         (prompt dir = c2s_den_<ABLATION>/)
#   METHOD_TAG default = c2s_grid_<served>_<ablation>
#                                          (matches run_vllm_grid.sh)
#
# Assumes prereqs already on disk (built by run_vllm_grid.sh BUILD_PREREQ=1
# on some earlier open-source run, OR manually via src.llm.c2s_grid +
# src.llm.c2s_prompt --prompt-subdir c2s_den_<ABLATION>):
#   - benchmark/<ds>/<sample>/step_NN/c2s_grid.json
#   - benchmark/<ds>/<sample>/step_NN/c2s_den_<ABLATION>/cells.md
# This runner does NOT rebuild them (no GPU on the OpenAI route).
#
# Usage:
#     OPENAI_API_KEY=sk-... \
#         bash src/llm/scripts/run_openai_grid.sh \
#              --config src/llm/configs/models/gpt_5_4.yaml \
#              [ABLATION] [OUT_DIR] [DATASETS...]
#
# Example:
#     SPLIT=test OPENAI_API_KEY=sk-... \
#         bash src/llm/scripts/run_openai_grid.sh \
#              --config src/llm/configs/models/gpt_5_4.yaml \
#              hvp10 output/eval_c2s_grid_gpt54_v2 \
#              Acute2020 Acute2021 Vaccine
#
# Env overrides (orchestration only — model/generation come from YAML):
#     CONCURRENCY=20                 (lower than vllm — OpenAI rate limits)
#     PARALLEL_CLIENTS=4
#     SPLIT=test|val|train
#     BENCHMARK_DIR=benchmark
#     DATA_DIR=data
#     SKIP_PROPAGATE=0
#     SKIP_EVAL=0
#     METHOD_TAG=c2s_grid_<served>_<ablation>   (default; auto-derived)
#     SAVE_RAW=0
#     OPENAI_BASE_URL=               (override api.openai.com — optional)

set -uo pipefail

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

ABLATION="${1:-hvp10}"
OUT_DIR="${2:-output/eval_c2s_grid_openai}"
shift 2 2>/dev/null || true
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(Acute2020 Acute2021 Vaccine)
fi

# Prompt dir convention (mirrors run_vllm_grid.sh): data_loader reads
# c2s_<ablation_slug>/cells.md, so DEN_SLUG=den_<ABLATION> → c2s_den_<ABL>/.
DEN_SLUG="den_${ABLATION}"
PROMPT_SUBDIR="c2s_${DEN_SLUG}"

# ── Load YAML → MODEL, SERVED_NAME, OPENAI_ARGS[] ──────────────────────────
CONFIG_SHELL="$(python3 -m src.llm.scripts._config_to_shell --config "$CONFIG")" || {
    echo "ERROR: failed to parse $CONFIG (see stderr above)" >&2
    exit 2
}
eval "$CONFIG_SHELL"

# ── Sanity: this runner is for OpenAI-API configs only. ────────────────────
if [[ "${OPENAI_MODE:-0}" != "1" ]]; then
    echo "ERROR: $CONFIG is not an OpenAI API config (model.openai_id missing)." >&2
    echo "       Use run_vllm_grid.sh for vllm-served (model.hf_id) configs." >&2
    exit 2
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set in the environment." >&2
    exit 2
fi

# ── Orchestration env (NOT in YAML — these change per experiment) ──────────
CONCURRENCY="${CONCURRENCY:-20}"
PARALLEL_CLIENTS="${PARALLEL_CLIENTS:-4}"
SPLIT="${SPLIT:-}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
DATA_DIR="${DATA_DIR:-data}"
SKIP_PROPAGATE="${SKIP_PROPAGATE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
METHOD_TAG="${METHOD_TAG:-c2s_grid_${SERVED_NAME}_${ABLATION}}"
SAVE_RAW="${SAVE_RAW:-0}"

mkdir -p "$OUT_DIR" /tmp/c2s_logs
SUFFIX_SPLIT=""
[[ -n "$SPLIT" ]] && SUFFIX_SPLIT="__${SPLIT}"

echo "=========================================================="
echo " LLM density-grid C2S runner — OpenAI API mode"
echo "----------------------------------------------------------"
echo "  CONFIG         : $CONFIG"
echo "  MODEL          : $MODEL"
echo "  SERVED_NAME    : $SERVED_NAME"
echo "  BASE_URL       : ${OPENAI_BASE_URL:-(default api.openai.com)}"
echo "  OPENAI_ARGS    : ${OPENAI_ARGS[*]:-(empty → run_openai defaults)}"
echo "  CONCURRENCY    : $CONCURRENCY (per client)"
echo "  PARALLEL_CLIENTS: $PARALLEL_CLIENTS"
echo "  ABLATION       : $ABLATION  (prompt dir: $PROMPT_SUBDIR)"
echo "  DEN_SLUG       : $DEN_SLUG  (--ablation_slug to run_openai)"
echo "  C2S filename   : c2s_grid.json"
echo "  OUT_DIR        : $OUT_DIR"
echo "  METHOD_TAG     : $METHOD_TAG"
echo "  SAVE_RAW       : $SAVE_RAW"
echo "  SPLIT          : ${SPLIT:-(all)}"
echo "  DATASETS (${#DATASETS[@]}): ${DATASETS[*]}"
echo "=========================================================="

# ── Prereq sanity (no-build): warn if any dataset is missing c2s_grid or cells.md
echo ""
echo "[0/3] Prereq sanity check (no build — re-render with run_vllm_grid.sh BUILD_PREREQ=1 if missing)"
for ds in "${DATASETS[@]}"; do
    n_grid=$(find "$BENCHMARK_DIR/$ds" -mindepth 3 -maxdepth 3 -name c2s_grid.json 2>/dev/null | wc -l)
    n_md=$(find "$BENCHMARK_DIR/$ds" -mindepth 4 -maxdepth 4 -path "*/${PROMPT_SUBDIR}/cells.md" 2>/dev/null | wc -l)
    if [[ "$n_grid" -eq 0 || "$n_md" -eq 0 ]]; then
        echo "  [WARN] $ds: c2s_grid.json=$n_grid, ${PROMPT_SUBDIR}/cells.md=$n_md (prereq missing)"
    else
        echo "  [ok]   $ds: c2s_grid.json=$n_grid, ${PROMPT_SUBDIR}/cells.md=$n_md"
    fi
done

# ── Run clients ────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Running clients (parallel groups of $PARALLEL_CLIENTS)"

run_dataset() {
    local ds=$1
    local out="$OUT_DIR/${ds}_${SERVED_NAME}_${ABLATION}${SUFFIX_SPLIT}.json"
    local extra_args=()
    if [[ -n "$SPLIT" ]]; then
        local list_log="/tmp/c2s_logs/c2s_grid_openai_split_${ds}${SUFFIX_SPLIT}.log"
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
    echo "  [start] $ds → $out"
    python3 -m src.llm.inference.run_openai \
        --dataset_path "$BENCHMARK_DIR/$ds" \
        --model "$SERVED_NAME" \
        --ablation_slug "$DEN_SLUG" \
        --c2s-filename c2s_grid.json \
        --concurrency "$CONCURRENCY" \
        --output_path "$out" \
        "${OPENAI_ARGS[@]}" \
        "${extra_args[@]}" \
        > "/tmp/c2s_logs/c2s_grid_openai_client_${ds}${SUFFIX_SPLIT}.log" 2>&1
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
echo "[2/3] Propagating grid predictions → prediction.json (method=$METHOD_TAG)"
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
python3 -m src.llm.postprocess.propagate \
    --eval_path "${EVAL_PATH_ARGS[@]}" \
    --dataset   "${DATASET_ARGS[@]}" \
    --benchmark "$BENCHMARK_DIR" \
    --data-dir  "$DATA_DIR" \
    --method    "$METHOD_TAG" \
    --output_dir "$PROP_DIR" \
    --mapping-filename cell2grid.npz \
    2>&1 | tee "/tmp/c2s_logs/c2s_grid_openai_propagate_${SERVED_NAME}${SUFFIX_SPLIT}.log"

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
python3 -m src.eval \
    --predictions "$PROP_DIR" \
    --benchmark   "$BENCHMARK_DIR" \
    --data-dir    "$DATA_DIR" \
    --datasets    "${EVAL_DATASETS[@]}" \
    "${EVAL_EXTRA[@]}" \
    2>&1 | tee "/tmp/c2s_logs/c2s_grid_openai_eval_${SERVED_NAME}${SUFFIX_SPLIT}.log"

echo ""
echo "[DONE] failures (clients): $FAIL"
echo "  pred.json   :  $OUT_DIR/<DATASET>_${SERVED_NAME}_${ABLATION}${SUFFIX_SPLIT}.json"
echo "  predictions :  $PROP_DIR/<DATASET>/<sample>/step_NN/$METHOD_TAG/prediction.json"
echo "  eval_summary:  $PROP_DIR/<DATASET>/eval_summary.json"
exit $FAIL
