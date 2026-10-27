#!/usr/bin/env bash
#BSUB -J summarize_paper_gates_rect
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 2G
#BSUB -R "rusage[mem=2G]"
#BSUB -W 00:10
#BSUB -o jobs/logs/summarize_paper_gates_rect.%J.out
#BSUB -e jobs/logs/summarize_paper_gates_rect.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PY'
import json
from pathlib import Path

cohorts = ['Acute2020', 'Acute2021', 'Bjornson', 'Vaccine',
           'FR-FCM-Z74D_hc', 'FR-FCM-Z74D_tissue', 'FRDR_covid19',
           'Lyoplate_DC', 'Lyoplate_bcell', 'Lyoplate_tcell', 'Lyoplate_treg']
root = Path('results/gate_gpt-5.4_paper_gates_current_benchmark')
rows = []
for cohort in cohorts:
    raw = json.loads((Path('results/gate_gpt-5.4_full_old') / cohort / 'pred.json').read_text())
    raw_steps = [(sample, int(step)) for sample, record in raw['samples'].items()
                 for step in record['steps']]
    missing = [(sample, step) for sample, step in raw_steps
               if not (Path('benchmark') / cohort / sample / f'step_{step:02d}' / 'task.json').exists()]
    base = root / 'predictions' / cohort
    gt = json.loads((base / 'eval_summary.json').read_text())
    rect = json.loads((base / 'eval_summary__gt_rect1_oracle.json').read_text())
    expected = len(raw_steps) - len(missing)
    if gt['n_tasks'] != expected or rect['n_tasks'] != expected:
        raise ValueError(f'{cohort}: expected {expected} available tasks; got GT {gt["n_tasks"]}, rect {rect["n_tasks"]}')
    rows.append({'cohort': cohort, 'n_tasks': expected,
                 'excluded_missing_task': len(missing),
                 'gt': gt['overall'], 'rect_gt': rect['overall']})
    print(cohort, expected, 'excluded_missing_task', len(missing),
          'rect F1', rect['overall']['f1_macro_mean'],
          'BA', rect['overall']['balanced_accuracy_mean'],
          'IoU', rect['overall']['iou_hull_macro_mean'])
keys = ['f1_macro_mean', 'balanced_accuracy_mean', 'iou_hull_macro_mean']
overall = {label: {key: sum(row[label][key] for row in rows) / len(rows) for key in keys}
           for label in ('gt', 'rect_gt')}
summary = {'method': 'old GPT-5.4 gates repropagated on current benchmark',
           'aggregation': 'equal_cohort_mean', 'n_cohorts': len(rows),
           'n_tasks': sum(row['n_tasks'] for row in rows),
           'excluded_missing_tasks': sum(row['excluded_missing_task'] for row in rows),
           'overall': overall, 'per_cohort': rows}
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
print('OVERALL', json.dumps(summary['overall']))
print('COVERAGE', summary['n_tasks'], 'excluded_missing_tasks', summary['excluded_missing_tasks'])
PY
