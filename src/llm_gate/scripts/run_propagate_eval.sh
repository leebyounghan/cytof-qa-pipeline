#!/usr/bin/env bash
# Propagation + evaluation only — no vLLM server, CPU only.
# Reads pred.json from OUT_DIR, writes prediction.json + eval_summary.json.
#
# Usage:
#   SPLIT=test bash src/llm_gate/scripts/run_propagate_eval.sh \
#     <OUT_DIR> <METHOD_TAG> [DATASETS...]
#
# Example:
#   SPLIT=test bash src/llm_gate/scripts/run_propagate_eval.sh \
#     output/eval_gate_qwen3_6_27b_tb8096 \
#     gate_Qwen3.6-27B-tb8096_full
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

OUT_DIR="${1:?Usage: $0 <OUT_DIR> <METHOD_TAG> [DATASETS...]}"
METHOD="${2:?Usage: $0 <OUT_DIR> <METHOD_TAG> [DATASETS...]}"
shift 2
DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(Acute2020 Acute2021 Bjornson Vaccine
              FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19
              Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
fi

SPLIT="${SPLIT:-}"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
DATA_DIR="${DATA_DIR:-data}"
TIEBREAK="${TIEBREAK:-smallest}"
PROP_DIR="$OUT_DIR/predictions"

mkdir -p "$PROP_DIR" /tmp/c2s_logs

echo "=========================================================="
echo " Propagate + Eval"
echo "  OUT_DIR: $OUT_DIR"
echo "  METHOD: $METHOD"
echo "  DATASETS (${#DATASETS[@]}): ${DATASETS[*]}"
echo "=========================================================="

# --- Find pred.json files ---
EVAL_ARGS=()
DS_ARGS=()
for ds in "${DATASETS[@]}"; do
    # Match any pred.json for this dataset
    found=$(ls "$OUT_DIR"/${ds}_*.json 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
        EVAL_ARGS+=("$found")
        DS_ARGS+=("$ds")
    else
        echo "  [skip] no pred.json for $ds"
    fi
done

if [[ ${#EVAL_ARGS[@]} -eq 0 ]]; then
    echo "[ERROR] no pred.json found"
    exit 1
fi

# --- Propagate (parallel per dataset) ---
echo ""
echo "[1/2] Propagating (${#EVAL_ARGS[@]} datasets, parallel)"
PROP_PIDS=()
PROP_FAIL=0
for i in "${!EVAL_ARGS[@]}"; do
    echo "  [prop] ${DS_ARGS[$i]}"
    python3 -m src.llm_gate.postprocess.propagate \
        --eval_path "${EVAL_ARGS[$i]}" \
        --dataset "${DS_ARGS[$i]}" \
        --benchmark "$BENCHMARK_DIR" \
        --data-dir "$DATA_DIR" \
        --method "$METHOD" \
        --output_dir "$PROP_DIR" \
        --tiebreak "$TIEBREAK" \
        > "/tmp/c2s_logs/prop_${METHOD}_${DS_ARGS[$i]}.log" 2>&1 &
    PROP_PIDS+=($!)
done
for pid in "${PROP_PIDS[@]}"; do
    if ! wait "$pid"; then PROP_FAIL=$((PROP_FAIL+1)); fi
done
echo "  Propagation done (failures: $PROP_FAIL)"

# --- Eval ---
echo ""
echo "[2/2] Eval"
EVAL_EXTRA=()
[[ -n "$SPLIT" ]] && EVAL_EXTRA+=(--split "$BENCHMARK_DIR/splits.json" --split-set "$SPLIT")
python3 -m src.eval \
    --predictions "$PROP_DIR" \
    --benchmark "$BENCHMARK_DIR" \
    --data-dir "$DATA_DIR" \
    --datasets "${DS_ARGS[@]}" \
    "${EVAL_EXTRA[@]}" \
    2>&1 | tee "/tmp/c2s_logs/eval_${METHOD}.log"

echo ""
echo "[DONE] propagate failures: $PROP_FAIL"
