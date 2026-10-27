#!/usr/bin/env bash
#BSUB -J validate_def_merge
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 5000
#BSUB -R "rusage[mem=5000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/validate_def_merge.%J.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/validate_def_merge.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python jobs/merge_definition_rerun_20261027.py c2s_flowdensity_gpt-5.4 --validate-only
