#!/usr/bin/env bash
#BSUB -J "visual_full40[1-6]%4"
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -R "rusage[mem=4096] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o jobs/logs/visual_full40.%J.%I.out
#BSUB -e jobs/logs/visual_full40.%J.%I.err
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
if (( zero < 3 )); then
  mode=vlm; style="${styles[$zero]}"
  module=src.vlm_gate.inference.run_openai
  extra=(--save-raw)
else
  mode=agent; style="${styles[$((zero - 3))]}"
  module=src.agent_gate.inference.run_agent_responses
  extra=()
fi

input_root=results/visual_luna_medium_definition/inputs
root=results/visual_luna_medium_definition/full_concurrency40
for dataset in "${datasets[@]}"; do
  out="$root/${mode}_${style}/${dataset}/pred.json"
  if [[ -s "$out" ]]; then
    echo "[resume] Existing completed output: $out"
    continue
  fi
  if [[ -e "$out" ]]; then
    echo "Incomplete output requires review: $out" >&2
    exit 1
  fi
  echo "[start] $mode $style $dataset; per-client concurrency=10"
  .venv/bin/python -m "$module" \
    --dataset_path "benchmark/$dataset" --input-root "$input_root" \
    --plot-style "$style" --model gpt-6-luna \
    --api-reasoning-effort medium --temperature 1.0 \
    --max_tokens 32768 --concurrency 10 "${extra[@]}" \
    --output_path "$out"
  echo "[done] $mode $style $dataset"
done
