#!/usr/bin/env bash
#BSUB -J prep_def_rerun
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 5000
#BSUB -R "rusage[mem=5000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/prep_def_rerun.%J.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/prep_def_rerun.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python jobs/prepare_definition_rerun_20261027.py
