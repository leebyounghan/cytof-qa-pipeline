#!/usr/bin/env bash
#BSUB -J equal_cohort_comparisons
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 4G
#BSUB -R "rusage[mem=4G]"
#BSUB -W 00:30
#BSUB -o jobs/logs/equal_cohort_comparisons.%J.out
#BSUB -e jobs/logs/equal_cohort_comparisons.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python results/compare_fulltest.py > results/gate_full_test_equal_cohort.txt
.venv/bin/python results/compare_luna_effort.py > results/luna_effort_equal_cohort.txt
.venv/bin/python results/compare_c2s_fulltest.py > results/c2s_full_test_equal_cohort.txt
.venv/bin/python jobs/summarize_gate_vs_rect1.py
.venv/bin/python jobs/summarize_rect1.py > results/rect1_oracle/summary_equal_cohort.txt
echo DONE
