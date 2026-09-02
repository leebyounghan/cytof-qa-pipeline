#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LLM-C2S seed sweep — mirrors the LLM-Gate seed sweep for Table 2.
#
# Two paradigms:
#   • density_grid  (flowDensity grid cell-sentences)  → src/llm/scripts/run_vllm_grid.sh
#   • clustering    (FlowSOM cell-sentences)           → src/llm_flowsom/scripts/run_vllm_flowsom.sh
#
# 5 open backbones × 10 seeds × 2 paradigms = 100 runs, sequential.
# Skips any (paradigm, model, seed) that already has 11 eval_summary.json.
# Resumable: safe to re-run; only fills what is missing.
#
# Prereqs (already built under BENCHMARK_DIR):
#   c2s.json, c2s_grid.json, cell2cluster.npz, cell2grid.npz,
#   c2s_den_hvp10/, c2s_clu_hvp10/   → so BUILD_PREREQ=0 for density_grid.
#
# The seed is threaded to the LLM generation via GENERATION_SEED → --seed
# (OpenAI API `seed`), matching the gate sweep. FlowSOM/flowDensity geometry
# is fixed across seeds (prompts prebuilt), so this isolates generation
# stochasticity exactly as the gate sweep does.
#
# Usage:   bash scripts/run_c2s_seed_sweep.sh
# Env:     SEEDS, MODELS override; SPLIT (default test)
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS=(1 2 3 4 5 6 7 42 123 456)

DATASETS=(Acute2020 Acute2021 Bjornson Vaccine
          FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19
          Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)

export SPLIT="${SPLIT:-test}"
export BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark}"
export DATA_DIR="${DATA_DIR:-data}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ABLATION=hvp10   # matches the paper's LLM-C2S rows (c2s_{den,clu}_hvp10 prompts)

# Which paradigms to sweep. Default: clustering only (FlowSOM backend, the
# recommended C2S formulation). Set PARADIGMS="density_grid clustering" to
# also sweep density-grid. Space-separated.
PARADIGMS="${PARADIGMS:-clustering}"
run_para() { [[ " $PARADIGMS " == *" $1 "* ]]; }

mkdir -p results/seed_sweep

# short_name|c2s_config  (paper LLM-C2S roster — 5 open backbones)
MODELS=(
  "qwen3_5_4b|src/llm/configs/models/qwen3_5_4b_tb8096.yaml"
  "qwen3_5_27b|src/llm/configs/models/qwen3_5_27b_tb8096.yaml"
  "qwen3_6_27b|src/llm/configs/models/qwen3_6_27b_tb8096.yaml"
  "gemma_4_26b_a4b|src/llm/configs/models/gemma_4_26b_a4b_it.yaml"
  "gemma_4_31b|src/llm/configs/models/gemma_4_31b_it.yaml"
)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# A run is "done" when its predictions dir holds >= 11 eval_summary.json.
is_done() {
  local out="$1"
  local n
  n=$(find "$out/predictions" -name eval_summary.json 2>/dev/null | wc -l)
  [ "$n" -ge 11 ]
}

# Count work
TOTAL=0; SKIP=0
for paradigm in $PARADIGMS; do
  for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r model_name _cfg <<< "$model_spec"
    for seed in "${SEEDS[@]}"; do
      if is_done "results/seed_sweep/${paradigm}/${model_name}_seed${seed}"; then
        SKIP=$((SKIP+1)); else TOTAL=$((TOTAL+1)); fi
    done
  done
done

log "=========================================="
log "  LLM-C2S seed sweep: $TOTAL to run, $SKIP already done"
log "  paradigms: $PARADIGMS | ablation=$ABLATION"
log "  benchmark: $BENCHMARK_DIR"
log "=========================================="

RUN=0; FAIL=0
for model_spec in "${MODELS[@]}"; do
  IFS='|' read -r model_name c2s_cfg <<< "$model_spec"

  for seed in "${SEEDS[@]}"; do

    # ── 1. LLM-C2S density-grid ──
    OUT="results/seed_sweep/density_grid/${model_name}_seed${seed}"
    if ! run_para density_grid; then
      :
    elif is_done "$OUT"; then
      log "SKIP density_grid/$model_name/seed$seed (done)"
    else
      RUN=$((RUN+1))
      log "[$RUN/$TOTAL] density_grid / $model_name / seed=$seed"
      GENERATION_SEED="$seed" BUILD_PREREQ=0 \
        bash src/llm/scripts/run_vllm_grid.sh \
          --config "$c2s_cfg" \
          "$ABLATION" "$OUT" "${DATASETS[@]}" \
        > "results/seed_sweep/log_density_grid_${model_name}_seed${seed}.txt" 2>&1 \
        || { log "FAILED: density_grid/$model_name/seed$seed"; FAIL=$((FAIL+1)); }
    fi

    # ── 2. LLM-C2S clustering (FlowSOM) ──
    OUT="results/seed_sweep/clustering/${model_name}_seed${seed}"
    if ! run_para clustering; then
      :
    elif is_done "$OUT"; then
      log "SKIP clustering/$model_name/seed$seed (done)"
    else
      RUN=$((RUN+1))
      log "[$RUN/$TOTAL] clustering / $model_name / seed=$seed"
      GENERATION_SEED="$seed" \
        bash src/llm_flowsom/scripts/run_vllm_flowsom.sh \
          --config "$c2s_cfg" \
          "$ABLATION" "$OUT" "${DATASETS[@]}" \
        > "results/seed_sweep/log_clustering_${model_name}_seed${seed}.txt" 2>&1 \
        || { log "FAILED: clustering/$model_name/seed$seed"; FAIL=$((FAIL+1)); }
    fi

  done
done

log "=========================================="
log "  DONE: $RUN runs attempted, $FAIL failures, $SKIP skipped"
log "=========================================="
