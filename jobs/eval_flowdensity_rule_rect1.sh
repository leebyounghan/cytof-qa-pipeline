#!/usr/bin/env bash
#BSUB -J "flowdensity_rect1[1-11]%3"
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 32G
#BSUB -R "rusage[mem=32G]"
#BSUB -W 03:00
#BSUB -o jobs/logs/flowdensity_rect1.%J.%I.out
#BSUB -e jobs/logs/flowdensity_rect1.%J.%I.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD"
cohorts=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
         FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
cohort="${cohorts[$((LSB_JOBINDEX-1))]}"
.venv/bin/python -m src.eval --predictions results/flowdensity_rule \
    --gt-labels results/rect1_oracle --benchmark benchmark --data-dir data \
    --datasets "$cohort" --split benchmark/splits.json --split-set test
.venv/bin/python - "$cohort" <<'PY'
import json
import sys
from pathlib import Path

cohort = sys.argv[1]
base = Path('results/flowdensity_rule') / cohort
gt = json.loads((base / 'eval_summary.json').read_text())
rect = json.loads((base / 'eval_summary__gt_rect1_oracle.json').read_text())
gt_ids = {row['task_id'] for row in gt['per_task']}
rect_ids = {row['task_id'] for row in rect['per_task']}
if gt_ids != rect_ids or len(gt_ids) != gt['n_tasks'] or len(rect_ids) != rect['n_tasks']:
    raise ValueError(f'{cohort}: task coverage differs: GT={len(gt_ids)}, rect={len(rect_ids)}')
print(f'{cohort}: matched {len(gt_ids)} GT and rect GT tasks; '
      f"rect F1={rect['overall']['f1_macro_mean']}, "
      f"BA={rect['overall']['balanced_accuracy_mean']}, "
      f"IoU={rect['overall']['iou_hull_macro_mean']}")
PY
