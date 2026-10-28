#!/usr/bin/env bash
#BSUB -J check_c2s_validator
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "rusage[mem=4000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/check_c2s_validator.%J.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/check_c2s_validator.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
.venv/bin/python jobs/validate_c2s_medium_output_20261027.py Acute2020 \
  results/c2s_flowdensity_gpt-6-luna/Acute2020_gpt-6-luna_hvp10.json
