#!/usr/bin/env bash
#BSUB -J compare_paper_gpt54
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 2G
#BSUB -R "rusage[mem=2G]"
#BSUB -W 00:10
#BSUB -o jobs/logs/compare_paper_gpt54.%J.out
#BSUB -e jobs/logs/compare_paper_gpt54.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('results')
for name, pattern in (
    ('old54', 'gate_gpt-5.4_full_old/*/eval_summary.json'),
    ('new54', 'gate_gpt-5.4_full/predictions/*/eval_summary.json'),
):
    paths = sorted(root.glob(pattern))
    paths = [p for p in paths if p.parent.name != 'YoungOld']
    if len(paths) != 11:
        raise ValueError(f'{name}: expected 11 cohorts, got {len(paths)}')
    overall = []
    print(name, 'cohorts', len(paths))
    for path in paths:
        o = json.loads(path.read_text())['overall']
        overall.append(o)
        print(path.parent.name, o['n_tasks'], o['f1_macro_mean'],
              o['balanced_accuracy_mean'], o['iou_hull_macro_mean'])
    print('TOTAL', sum(x['n_tasks'] for x in overall),
          'F1', sum(x['f1_macro_mean'] for x in overall) / len(overall),
          'BA', sum(x['balanced_accuracy_mean'] for x in overall) / len(overall),
          'IoU', sum(x['iou_hull_macro_mean'] for x in overall) / len(overall))
PY
