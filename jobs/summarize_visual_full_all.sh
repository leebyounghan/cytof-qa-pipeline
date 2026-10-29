#!/usr/bin/env bash
#BSUB -J summarize_visual_full_all
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -R "rusage[mem=1024] span[hosts=1]"
#BSUB -W 00:30
#BSUB -o jobs/logs/summarize_visual_full_all.%J.out
#BSUB -e jobs/logs/summarize_visual_full_all.%J.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
.venv/bin/python jobs/summarize_visual_full_vlm.py \
  --conditions llm_gate vlm_scatter vlm_contour vlm_density \
               agent_scatter agent_contour agent_density \
  --output-prefix visual_full_comparison
