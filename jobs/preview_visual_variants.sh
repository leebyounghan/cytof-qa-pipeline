#!/usr/bin/env bash
#BSUB -J preview_visual_luna
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -R "rusage[mem=8192] span[hosts=1]"
#BSUB -W 00:30
#BSUB -o jobs/logs/preview_visual_luna.%J.out
#BSUB -e jobs/logs/preview_visual_luna.%J.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/results/visual_luna_medium_definition/mplconfig"
mkdir -p "$MPLCONFIGDIR"
.venv/bin/python jobs/preview_visual_variants.py
