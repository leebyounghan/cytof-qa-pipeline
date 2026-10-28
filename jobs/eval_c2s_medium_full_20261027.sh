#!/usr/bin/env bash
#BSUB -J "evalc2smedium[1-44]%2"
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -M 40000
#BSUB -R "rusage[mem=40000] span[hosts=1]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/evalc2smedium.%J.%I.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/evalc2smedium.%J.%I.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
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
FILE="$OUT/${DS}_${MODEL}-medium_hvp10.json"
[[ -s "$FILE" ]] || { echo "ERROR: missing $FILE" >&2; exit 2; }
"$PY" jobs/validate_c2s_medium_output_20261027.py "$DS" "$FILE"
case "$PARADIGM" in
  flowdensity)
    "$PY" -m src.llm.postprocess.propagate --eval_path "$FILE" --dataset "$DS" \
      --benchmark benchmark --data-dir data --method "c2s_grid_${MODEL}-medium_hvp10" \
      --output_dir "$OUT/predictions" --mapping-filename cell2grid.npz
    ;;
  flowsom)
    "$PY" -m src.llm_flowsom.postprocess.propagate --eval_path "$FILE" --dataset "$DS" \
      --benchmark benchmark --data-dir data --method "c2s_flowsom_${MODEL}-medium_hvp10" \
      --output_dir "$OUT/predictions"
    ;;
esac
"$PY" -m src.eval --predictions "$OUT/predictions" --benchmark benchmark --data-dir data \
  --datasets "$DS" --split benchmark/splits.json --split-set test
"$PY" -m src.eval --predictions "$OUT/predictions" --gt-labels results/rect1_oracle \
  --benchmark benchmark --data-dir data --datasets "$DS" \
  --split benchmark/splits.json --split-set test
echo "DONE: $RUN / $DS"
