#!/usr/bin/env bash
#BSUB -J update_c2smedium_docs
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "rusage[mem=4000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/update_c2smedium_docs.%J.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/update_c2smedium_docs.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
.venv/bin/python jobs/update_c2s_medium_docs_20261027.py
