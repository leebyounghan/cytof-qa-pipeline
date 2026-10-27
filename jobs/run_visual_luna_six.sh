#!/usr/bin/env bash
#BSUB -J "visual_luna[1-66]%3"
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -R "rusage[mem=4096] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o jobs/logs/visual_luna.%J.%I.out
#BSUB -e jobs/logs/visual_luna.%J.%I.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
: "${LSB_JOBINDEX:?Submit as an array}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/results/visual_luna_medium_definition/mplconfig"
mkdir -p "$MPLCONFIGDIR"
.venv/bin/python -m py_compile src/visual_variants.py src/vlm_gate/inference/run_openai.py src/agent_gate/inference/run_agent_responses.py

datasets=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
styles=(scatter contour density)
zero=$((LSB_JOBINDEX - 1))
if (( zero < 33 )); then mode=vlm; module=src.vlm_gate.inference.run_openai
else mode=agent; module=src.agent_gate.inference.run_agent_responses; zero=$((zero - 33))
fi
style="${styles[$((zero / 11))]}"
dataset="${datasets[$((zero % 11))]}"
root="results/visual_luna_medium_definition"
out="$root/${mode}_${style}/${dataset}/pred.json"
if [[ -e "$out" ]]; then
  echo "Existing output requires explicit review before rerun: $out" >&2
  exit 1
fi
extra=()
if [[ "$mode" == vlm ]]; then extra+=(--save-raw); fi
echo "Running $mode / $style / $dataset with gpt-6-luna effort=medium"
.venv/bin/python -m "$module" \
  --dataset_path "benchmark/$dataset" \
  --input-root "$root/inputs" --plot-style "$style" \
  --model gpt-6-luna --api-reasoning-effort medium \
  --temperature 1.0 --max_tokens 32768 --concurrency 4 \
  "${extra[@]}" --output_path "$out"
