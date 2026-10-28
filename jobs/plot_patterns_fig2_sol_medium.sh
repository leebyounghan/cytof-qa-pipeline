#!/usr/bin/env bash
#BSUB -J patterns_fig2_sol_medium
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -R "rusage[mem=8192] span[hosts=1]"
#BSUB -W 01:00
#BSUB -o jobs/logs/patterns_fig2_sol_medium.%J.out
#BSUB -e jobs/logs/patterns_fig2_sol_medium.%J.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/results/fig2_gpt-6-sol-medium/mplconfig"
mkdir -p "$MPLCONFIGDIR"
.venv/bin/python -m case_study.compare_five \
    --model gpt-6-sol-medium \
    --cymae-root /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline_old_20260514/results/cyMAE \
    --strict \
    --out results/fig2_gpt-6-sol-medium/fig_method_compare_5way.png
