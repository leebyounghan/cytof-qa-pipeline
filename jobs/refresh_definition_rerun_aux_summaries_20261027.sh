#!/usr/bin/env bash
#BSUB -J refresh_def_summaries
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 3000
#BSUB -R "rusage[mem=3000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/refresh_def_summaries.%J.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/refresh_def_summaries.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python results/compare_c2s_fulltest.py > results/c2s_full_test_equal_cohort.txt
.venv/bin/python results/compare_fulltest.py > results/gate_full_test_equal_cohort.txt
.venv/bin/python results/compare_luna_effort.py > results/luna_effort_equal_cohort.txt
echo 'Updated C2S, Gate, and Luna effort comparison tables.'
