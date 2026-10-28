#!/usr/bin/env bash
#BSUB -J summarize_visual_pilot
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -R "rusage[mem=4096] span[hosts=1]"
#BSUB -W 01:00
#BSUB -o jobs/logs/summarize_visual_pilot.%J.out
#BSUB -e jobs/logs/summarize_visual_pilot.%J.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python -m py_compile jobs/summarize_visual_pilot.py
.venv/bin/python jobs/summarize_visual_pilot.py
