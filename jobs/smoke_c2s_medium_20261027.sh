#!/usr/bin/env bash
#BSUB -J "c2smedsmoke[1-4]%2"
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 6000
#BSUB -R "rusage[mem=6000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/c2smedsmoke.%J.%I.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/c2smedsmoke.%J.%I.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
PARADIGMS=(flowdensity flowsom)
MODELS=(gpt-6-luna gpt-6-sol)
PARADIGM=${PARADIGMS[$(((LSB_JOBINDEX-1)/2))]}
MODEL=${MODELS[$(((LSB_JOBINDEX-1)%2))]}
CFG=${MODEL//-/_}
CONFIG_SHELL=$("$PY" -m src.llm.scripts._config_to_shell --config "src/llm/configs/models/$CFG.yaml")
eval "$CONFIG_SHELL"
DS=FR-FCM-Z74D_hc
mapfile -t SAMPLES < <("$PY" -m src.llm.scripts.list_split_samples --splits benchmark/splits.json --dataset "$DS" --split test)
[[ ${#SAMPLES[@]} -gt 0 ]] || { echo 'No test samples' >&2; exit 2; }
case "$PARADIGM" in
  flowdensity) MOD=src.llm.inference.run_openai; EXTRA=(--ablation_slug den_hvp10 --c2s-filename c2s_grid.json) ;;
  flowsom) MOD=src.llm_flowsom.inference.run_openai; EXTRA=(--ablation_slug clu_hvp10) ;;
esac
OUT="results/_c2s_medium_smoke_20261027/c2s_${PARADIGM}_${MODEL}-medium"
mkdir -p "$OUT"
"$PY" -m "$MOD" --dataset_path "benchmark/$DS" --model "$SERVED_NAME" \
  "${EXTRA[@]}" --sample "${SAMPLES[0]}" --max_samples 1 --concurrency 1 \
  --api-reasoning-effort medium --output_path "$OUT/$DS.json" "${OPENAI_ARGS[@]}"
echo "Smoke complete: $PARADIGM $MODEL medium"
