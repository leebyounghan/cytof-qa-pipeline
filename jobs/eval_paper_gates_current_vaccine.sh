#!/usr/bin/env bash
#BSUB -J paper_gates_vaccine
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 32G
#BSUB -R "rusage[mem=32G]"
#BSUB -W 04:00
#BSUB -o jobs/logs/paper_gates_vaccine.%J.out
#BSUB -e jobs/logs/paper_gates_vaccine.%J.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
export LSB_JOBINDEX=4
bash /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/eval_paper_gates_current_rect1.sh
