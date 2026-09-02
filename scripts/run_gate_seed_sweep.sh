#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LLM-Gate seed sweep — the 10-seed stability runs behind the paper's
# seed-stability table (open-weight LLM-Gate rows are reported as 10-seed means).
#
# 5 open backbones × 10 seeds, sequential. Each run drives the full
# LLM-Gate pipeline (vLLM serve → clients → propagate → eval) via
# src/llm_gate/scripts/run_vllm_gate.sh with GENERATION_SEED exported.
#
# Skips any (model, seed) whose predictions dir already holds 11
# eval_summary.json. Resumable: safe to re-run; only fills what is missing.
#
# Usage:   bash scripts/run_gate_seed_sweep.sh
# Env:     SEEDS, SPLIT (default test), BENCHMARK_DIR, DATA_DIR
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS=(${SEEDS:-1 2 3 4 5 6 7 42 123 456})

DATASETS=(Acute2020 Acute2021 Bjornson Vaccine
          FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19
          Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)

export SPLIT="${SPLIT:-test}"
export BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
export DATA_DIR="${DATA_DIR:-data}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p results/seed_sweep

# short_name|gate_config  (paper LLM-Gate roster — 5 open backbones)
MODELS=(
  "qwen3_5_4b|src/llm_gate/configs/models/qwen3_5_4b_tb8096.yaml"
  "qwen3_5_27b|src/llm_gate/configs/models/qwen3_5_27b_tb8096.yaml"
  "qwen3_6_27b|src/llm_gate/configs/models/qwen3_6_27b_tb8096.yaml"
  "gemma_4_26b_a4b|src/llm_gate/configs/models/gemma_4_26b_a4b_it.yaml"
  "gemma_4_31b|src/llm_gate/configs/models/gemma_4_31b_it.yaml"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# A run is "done" when its predictions dir holds >= 11 eval_summary.json.
is_done() {
  local n
  n=$(find "$1/predictions" -name eval_summary.json 2>/dev/null | wc -l)
  [ "$n" -ge 11 ]
}

RUN=0; FAIL=0; SKIP=0
for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r model_name gate_cfg <<< "$model_spec"
  for seed in "${SEEDS[@]}"; do
    OUT="results/seed_sweep/gate/${model_name}_seed${seed}"
    if is_done "$OUT"; then
      SKIP=$((SKIP+1)); continue
    fi
    RUN=$((RUN+1))
    log "gate / $model_name / seed=$seed → $OUT"
    GENERATION_SEED="$seed" \
      bash src/llm_gate/scripts/run_vllm_gate.sh \
        --config "$gate_cfg" \
        full "$OUT" "${DATASETS[@]}" \
      2>&1 | tee "results/seed_sweep/log_gate_${model_name}_seed${seed}.txt" \
      || { log "FAILED: gate/$model_name/seed$seed"; FAIL=$((FAIL+1)); }
  done
done

log "DONE: $RUN run, $SKIP skipped, $FAIL failed"
log "Aggregate with: python scripts/aggregate_seed_sweep.py --root results/seed_sweep"
