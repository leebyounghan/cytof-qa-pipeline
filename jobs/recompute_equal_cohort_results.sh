#!/usr/bin/env bash
#BSUB -J equal_cohort_results
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 2G
#BSUB -R "rusage[mem=2G]"
#BSUB -W 00:10
#BSUB -o jobs/logs/equal_cohort_results.%J.out
#BSUB -e jobs/logs/equal_cohort_results.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python results/summarize_vs_gt_rect1.py > results/vs_gt_rect1_summary.md
.venv/bin/python - <<'PY'
import json
from pathlib import Path

cohorts = ['Acute2020', 'Acute2021', 'Bjornson', 'Vaccine',
           'FR-FCM-Z74D_hc', 'FR-FCM-Z74D_tissue', 'FRDR_covid19',
           'Lyoplate_DC', 'Lyoplate_bcell', 'Lyoplate_tcell', 'Lyoplate_treg']
rows = []
for cohort in cohorts:
    path = Path('results/flowdensity_rule') / cohort / 'eval_summary.json'
    rows.append(json.loads(path.read_text())['overall'])
keys = ['f1_macro_mean', 'f1_weighted_mean', 'balanced_accuracy_mean',
        'population_proportion_error_mean', 'iou_hull_macro_mean']
print('flowDensity equal cohort means; cohorts', len(rows),
      'steps', sum(row['n_tasks'] for row in rows))
for key in keys:
    print(key, sum(row[key] for row in rows) / len(rows))
PY
cat results/vs_gt_rect1_summary.md
