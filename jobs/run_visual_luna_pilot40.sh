#!/usr/bin/env bash
#BSUB -J "visual_pilot40[1-66]%4"
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -R "rusage[mem=4096] span[hosts=1]"
#BSUB -W 08:00
#BSUB -o jobs/logs/visual_pilot40.%J.%I.out
#BSUB -e jobs/logs/visual_pilot40.%J.%I.err
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
dataset="${datasets[$((zero / 6))]}"
condition=$((zero % 6))
if (( condition < 3 )); then
  mode=vlm
  style="${styles[$condition]}"
  module=src.vlm_gate.inference.run_openai
  extra=(--save-raw)
else
  mode=agent
  style="${styles[$((condition - 3))]}"
  module=src.agent_gate.inference.run_agent_responses
  extra=()
fi

input_root=results/visual_luna_medium_definition/inputs
mapfile -t samples < <(find "$input_root/$dataset" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort)
if (( ${#samples[@]} == 0 )); then echo "No test sample for $dataset" >&2; exit 1; fi
sample="${samples[0]}"
root=results/visual_luna_medium_definition/pilot_concurrency40
out="$root/${mode}_${style}/${dataset}/pred.json"
if [[ -e "$out" ]]; then echo "Existing output: $out" >&2; exit 1; fi
mkdir -p "$(dirname "$out")"
printf '%s\n' "$sample" > "$(dirname "$out")/sample.txt"
echo "Running $dataset $mode $style with per-job concurrency 10"
.venv/bin/python -m "$module" \
  --dataset_path "benchmark/$dataset" --input-root "$input_root" \
  --plot-style "$style" --sample "$sample" \
  --model gpt-6-luna --api-reasoning-effort medium \
  --temperature 1.0 --max_tokens 32768 --concurrency 10 \
  "${extra[@]}" --output_path "$out"
