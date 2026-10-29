#!/usr/bin/env bash
#BSUB -J check_agent_resume
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -R "rusage[mem=1024] span[hosts=1]"
#BSUB -W 00:10
#BSUB -o jobs/logs/check_agent_resume.%J.out
#BSUB -e jobs/logs/check_agent_resume.%J.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
bash -n jobs/run_visual_luna_full40.sh
.venv/bin/python -m py_compile src/agent_gate/inference/run_agent_responses.py
echo 'Agent resume syntax checks passed'
