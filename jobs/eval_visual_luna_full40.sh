#!/usr/bin/env bash
#BSUB -J "eval_visual_full40[1-6]%2"
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -R "rusage[mem=24576] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o jobs/logs/eval_visual_full40.%J.%I.out
#BSUB -e jobs/logs/eval_visual_full40.%J.%I.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
: "${LSB_JOBINDEX:?Submit as an array}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/results/visual_luna_medium_definition/mplconfig"
mkdir -p "$MPLCONFIGDIR"

datasets=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
styles=(scatter contour density)
zero=$((LSB_JOBINDEX - 1))
if (( zero < 3 )); then mode=vlm; module=src.vlm_gate.postprocess.propagate
else mode=agent; module=src.agent_gate.postprocess.propagate; zero=$((zero - 3))
fi
style="${styles[$zero]}"
condition="${mode}_${style}"
root="results/visual_luna_medium_definition/full_concurrency40/$condition"
eval_paths=()
for dataset in "${datasets[@]}"; do
  path="$root/$dataset/pred.json"
  if [[ ! -s "$path" ]]; then echo "Missing inference output: $path" >&2; exit 1; fi
  eval_paths+=("$path")
done
.venv/bin/python -m "$module" \
  --eval_path "${eval_paths[@]}" --dataset "${datasets[@]}" \
  --benchmark benchmark --data-dir data --method "$condition" \
  --output_dir "$root/predictions" --tiebreak smallest
.venv/bin/python -m src.eval \
  --predictions "$root/predictions" --benchmark benchmark --data-dir data \
  --datasets "${datasets[@]}" --split benchmark/splits.json --split-set test
echo "DONE $condition"
