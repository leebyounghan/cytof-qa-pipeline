#!/usr/bin/env bash
#BSUB -J "c2smedium[1-44]%4"
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 8000
#BSUB -R "rusage[mem=8000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/c2smedium.%J.%I.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/c2smedium.%J.%I.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
PARADIGMS=(flowdensity flowsom)
MODELS=(gpt-6-luna gpt-6-sol)
METHOD_INDEX=$(((LSB_JOBINDEX-1)/11))
DS_INDEX=$(((LSB_JOBINDEX-1)%11))
PARADIGM=${PARADIGMS[$((METHOD_INDEX/2))]}
MODEL=${MODELS[$((METHOD_INDEX%2))]}
DS=${DATASETS[$DS_INDEX]}
RUN="c2s_${PARADIGM}_${MODEL}-medium"
OUT="results/$RUN"
CFG=${MODEL//-/_}
CONFIG_SHELL=$("$PY" -m src.llm.scripts._config_to_shell --config "src/llm/configs/models/$CFG.yaml")
eval "$CONFIG_SHELL"
mapfile -t SAMPLES < <("$PY" -m src.llm.scripts.list_split_samples --splits benchmark/splits.json --dataset "$DS" --split test)
[[ ${#SAMPLES[@]} -gt 0 ]] || { echo "ERROR: no test samples for $DS" >&2; exit 2; }
case "$PARADIGM" in
  flowdensity) MOD=src.llm.inference.run_openai; EXTRA=(--ablation_slug den_hvp10 --c2s-filename c2s_grid.json) ;;
  flowsom) MOD=src.llm_flowsom.inference.run_openai; EXTRA=(--ablation_slug clu_hvp10) ;;
esac
mkdir -p "$OUT/logs"
FILE="$OUT/${DS}_${MODEL}-medium_hvp10.json"
[[ ! -e "$FILE" ]] || { echo "ERROR: output already exists: $FILE" >&2; exit 2; }
echo "Running $RUN / $DS / samples=${#SAMPLES[@]}"
"$PY" -m "$MOD" --dataset_path "benchmark/$DS" --model "$SERVED_NAME" \
  "${EXTRA[@]}" --sample "${SAMPLES[@]}" --concurrency 8 \
  --api-reasoning-effort medium --output_path "$FILE" "${OPENAI_ARGS[@]}" \
  > "$OUT/logs/${DS}_client.log" 2>&1
"$PY" jobs/validate_c2s_medium_output_20261027.py "$DS" "$FILE"
echo "DONE: $RUN / $DS"
