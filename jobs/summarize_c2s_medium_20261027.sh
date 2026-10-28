#!/usr/bin/env bash
#BSUB -J summarize_c2smedium
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 6000
#BSUB -R "rusage[mem=6000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/summarize_c2smedium.%J.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/summarize_c2smedium.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python jobs/summarize_c2s_medium_20261027.py
