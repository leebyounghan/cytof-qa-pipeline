#!/usr/bin/env bash
#BSUB -J "build_visual[1-11]%3"
#BSUB -q i2c2_normal
#BSUB -n 2
#BSUB -R "rusage[mem=8192] span[hosts=1]"
#BSUB -W 08:00
#BSUB -o jobs/logs/build_visual.%J.%I.out
#BSUB -e jobs/logs/build_visual.%J.%I.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
: "${LSB_JOBINDEX:?Submit as an array}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/results/visual_luna_medium_definition/mplconfig"
mkdir -p "$MPLCONFIGDIR"
datasets=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
dataset="${datasets[$((LSB_JOBINDEX-1))]}"
.venv/bin/python jobs/build_visual_inputs.py --dataset "$dataset"
