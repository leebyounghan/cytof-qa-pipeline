#!/usr/bin/env bash
#BSUB -J "old_gpt54_rect[1-11]%3"
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 12G
#BSUB -R "rusage[mem=12G]"
#BSUB -W 04:00
#BSUB -o jobs/logs/old_gpt54_rect.%J.%I.out
#BSUB -e jobs/logs/old_gpt54_rect.%J.%I.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
: "${LSB_JOBINDEX:?Submit as an LSF job array}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
cohort="${DATASETS[$((LSB_JOBINDEX - 1))]}"
.venv/bin/python - "$cohort" <<'PY'
import json
import sys
from pathlib import Path

from src.eval import evaluate_predictions

cohort = sys.argv[1]
root = Path('results/gate_gpt-5.4_full_old')
samples = set(json.loads(Path('benchmark/splits.json').read_text())['in_panel'][cohort]['test'])
summary = evaluate_predictions(
    predictions_dir=root,
    benchmark_dir=Path('benchmark'),
    data_dir=Path('data'),
    dataset=cohort,
    sample_filter=samples,
    gt_labels_dir=Path('results/rect1_oracle'),
)
expected = json.loads((root / cohort / 'eval_summary.json').read_text())['n_tasks']
if not summary or summary['n_tasks'] != expected:
    raise ValueError(f'{cohort}: expected {expected} evaluated tasks, got {summary.get("n_tasks")}')
summary['gt_labels'] = 'results/rect1_oracle'
destination = Path('results/gate_gpt-5.4_full_old_rect_eval') / cohort / 'eval_summary__gt_rect1_oracle.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
overall = summary['overall']
print(cohort, summary['n_tasks'], overall['f1_macro_mean'],
      overall['balanced_accuracy_mean'], overall['iou_hull_macro_mean'],
      'iou_tasks', overall['iou_hull_macro_n'])
print('SAVED', destination)
PY
