#!/usr/bin/env bash
#BSUB -J "defrerun[1-11]%3"
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 5000
#BSUB -R "rusage[mem=5000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/defrerun.%J.%I.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/defrerun.%J.%I.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
ROOT=results/_definition_rerun_20261027
[[ -f "$ROOT/manifest.json" ]] || { echo "ERROR: run the preparation job first" >&2; exit 2; }
RUNS=(
  c2s_flowdensity_gpt-5.4 c2s_flowdensity_gpt-6-luna
  c2s_flowsom_gpt-5.4 c2s_flowsom_gpt-6-luna
  gate_gpt-5.4_full gate_gpt-6-luna_full
  gate_gpt-6-luna-none_full gate_gpt-6-luna-low_full
  gate_gpt-6-luna-medium_full gate_gpt-6-luna-high_full
  gate_gpt-6-sol-medium_full
)
RUN=${RUNS[$((LSB_JOBINDEX - 1))]}
case "$RUN" in
  c2s_flowdensity_*)
    MOD=src.llm.inference.run_openai
    EXTRA=(--ablation_slug den_hvp10 --c2s-filename c2s_grid.json)
    CONFIG_ROOT=src/llm
    ;;
  c2s_flowsom_*)
    MOD=src.llm_flowsom.inference.run_openai
    EXTRA=(--ablation_slug clu_hvp10)
    CONFIG_ROOT=src/llm
    ;;
  gate_*)
    MOD=src.llm_gate.inference.run_openai
    EXTRA=(--ablation_slug full --save-raw)
    CONFIG_ROOT=src/llm_gate
    ;;
esac
case "$RUN" in
  *gpt-5.4*) CFG=gpt_5_4; EFFORT= ;;
  *gpt-6-sol*) CFG=gpt_6_sol; EFFORT=medium ;;
  *gpt-6-luna-none*) CFG=gpt_6_luna; EFFORT=none ;;
  *gpt-6-luna-low*) CFG=gpt_6_luna; EFFORT=low ;;
  *gpt-6-luna-medium*) CFG=gpt_6_luna; EFFORT=medium ;;
  *gpt-6-luna-high*) CFG=gpt_6_luna; EFFORT=high ;;
  *gpt-6-luna*) CFG=gpt_6_luna; EFFORT= ;;
esac
CONFIG_MOD=${CONFIG_ROOT//\//.}
CONFIG_SHELL=$("$PY" -m "$CONFIG_MOD.scripts._config_to_shell" --config "$CONFIG_ROOT/configs/models/$CFG.yaml")
eval "$CONFIG_SHELL"
if [[ -n "$EFFORT" ]]; then EXTRA+=(--api-reasoning-effort "$EFFORT"); fi
OUT="$ROOT/$RUN"
mkdir -p "$OUT/logs"
for DS in Acute2020 Acute2021 Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue; do
  echo "[$RUN] $DS started"
  "$PY" -m "$MOD" --dataset_path "$ROOT/benchmark/$DS" --model "$SERVED_NAME" \
    "${EXTRA[@]}" --concurrency 8 --output_path "$OUT/$DS.json" "${OPENAI_ARGS[@]}" \
    > "$OUT/logs/$DS.log" 2>&1
  echo "[$RUN] $DS done"
done
echo "DONE: $RUN"
