#!/usr/bin/env bash
#BSUB -J update_def_tables
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 3000
#BSUB -R "rusage[mem=3000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/update_def_tables.%J.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/update_def_tables.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python jobs/update_definition_rerun_tables_20261027.py
