#!/usr/bin/env bash
#BSUB -J smoke_visual_luna
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -R "rusage[mem=4096] span[hosts=1]"
#BSUB -W 02:00
#BSUB -o jobs/logs/smoke_visual_luna.%J.out
#BSUB -e jobs/logs/smoke_visual_luna.%J.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/results/visual_luna_medium_definition/mplconfig"
mkdir -p "$MPLCONFIGDIR" results/visual_luna_medium_definition/smoke
PY=.venv/bin/python
$PY -m py_compile src/visual_variants.py src/vlm_gate/inference/run_openai.py src/agent_gate/inference/run_agent_responses.py jobs/build_visual_inputs.py
for mode in vlm agent; do
  for style in scatter contour density; do
    if [[ "$mode" == vlm ]]; then module=src.vlm_gate.inference.run_openai; else module=src.agent_gate.inference.run_agent_responses; fi
    out="results/visual_luna_medium_definition/smoke/${mode}_${style}.json"
    if [[ -f "$out" ]]; then echo "[skip] $out"; continue; fi
    echo "[start] $mode $style"
    extra=()
    if [[ "$mode" == vlm ]]; then extra+=(--save-raw); fi
    $PY -m "$module" --dataset_path benchmark/Bjornson \
      --input-root results/visual_luna_medium_definition/inputs \
      --plot-style "$style" --model gpt-6-luna \
      --api-reasoning-effort medium --temperature 1.0 --max_tokens 16384 \
      --sample R15W11_W070515101208_Hu_M_Basal --max_samples 1 --seed 42 \
      --concurrency 1 "${extra[@]}" --output_path "$out"
    echo "[done] $mode $style $out"
  done
done
