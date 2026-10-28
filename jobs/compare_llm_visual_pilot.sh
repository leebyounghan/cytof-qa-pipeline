#!/usr/bin/env bash
#BSUB -J compare_llm_visual_pilot
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -R "rusage[mem=8192] span[hosts=1]"
#BSUB -W 04:00
#BSUB -o jobs/logs/compare_llm_visual_pilot.%J.out
#BSUB -e jobs/logs/compare_llm_visual_pilot.%J.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD:$PWD/jobs" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
root=results/visual_luna_medium_definition/pilot_concurrency40/llm_gate_matched
datasets=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
.venv/bin/python jobs/compare_llm_visual_pilot.py slice
paths=()
for dataset in "${datasets[@]}"; do
  paths+=("$root/$dataset/pred.json")
done
.venv/bin/python -m src.llm_gate.postprocess.propagate \
  --eval_path "${paths[@]}" --dataset "${datasets[@]}" \
  --benchmark benchmark --data-dir data --method llm_gate \
  --output_dir "$root/predictions" --tiebreak smallest
.venv/bin/python -m src.eval \
  --predictions "$root/predictions" --benchmark benchmark --data-dir data \
  --datasets "${datasets[@]}" --split benchmark/splits.json --split-set test
.venv/bin/python jobs/compare_llm_visual_pilot.py summarize
