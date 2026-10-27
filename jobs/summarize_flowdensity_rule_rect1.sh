#!/usr/bin/env bash
#BSUB -J summarize_flowdensity_rect1
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 2G
#BSUB -R "rusage[mem=2G]"
#BSUB -W 00:10
#BSUB -o jobs/logs/summarize_flowdensity_rect1.%J.out
#BSUB -e jobs/logs/summarize_flowdensity_rect1.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PY'
import json
from pathlib import Path

cohorts = ['Acute2020', 'Acute2021', 'Bjornson', 'Vaccine',
           'FR-FCM-Z74D_hc', 'FR-FCM-Z74D_tissue', 'FRDR_covid19',
           'Lyoplate_DC', 'Lyoplate_bcell', 'Lyoplate_tcell', 'Lyoplate_treg']
root = Path('results/flowdensity_rule')
rows = []
for cohort in cohorts:
    base = root / cohort
    gt = json.loads((base / 'eval_summary.json').read_text())
    rect = json.loads((base / 'eval_summary__gt_rect1_oracle.json').read_text())
    gt_ids = {row['task_id'] for row in gt['per_task']}
    rect_ids = {row['task_id'] for row in rect['per_task']}
    if gt_ids != rect_ids or len(gt_ids) != gt['n_tasks'] or len(rect_ids) != rect['n_tasks']:
        raise ValueError(f'{cohort}: GT and rect GT task coverage differs')
    rows.append({'cohort': cohort, 'n_tasks': len(gt_ids),
                 'gt': gt['overall'], 'rect_gt': rect['overall']})

keys = ['f1_macro_mean', 'balanced_accuracy_mean', 'iou_hull_macro_mean']
overall = {label: {key: sum(row[label][key] for row in rows) / len(rows)
                   for key in keys} for label in ('gt', 'rect_gt')}
summary = {'method': 'flowdensity_rule', 'aggregation': 'equal_cohort_mean',
           'n_cohorts': len(rows), 'n_tasks': sum(row['n_tasks'] for row in rows),
           'overall': overall, 'per_cohort': rows}
(root / 'rect_gt_summary.json').write_text(json.dumps(summary, indent=2))
print('OVERALL', json.dumps(overall))
print('COVERAGE', summary['n_tasks'])
for row in rows:
    print(row['cohort'], row['n_tasks'], row['rect_gt']['f1_macro_mean'],
          row['rect_gt']['balanced_accuracy_mean'], row['rect_gt']['iou_hull_macro_mean'])
PY
