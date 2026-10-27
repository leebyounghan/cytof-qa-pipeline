#!/usr/bin/env bash
#BSUB -J plot_maitnkt_gt
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 4000
#BSUB -R "rusage[mem=4000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/results/_tmp_maitnkt_gt_20261027/plot.%J.log

set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/results/_tmp_maitnkt_gt_20261027/mplconfig

.venv/bin/python - <<'PY'
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path('/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline')
sample = '994570_Normalized'
task_path = root / 'benchmark' / 'Acute2020' / sample / 'step_61' / 'task.json'
parquet_path = root / 'data' / 'Acute2020' / 'parquet' / f'{sample}.parquet'
out_dir = root / 'results' / '_tmp_maitnkt_gt_20261027'
task = json.loads(task_path.read_text())
assert task['step'] == 61 and task['dataset'] == 'Acute2020'
parent_col, parent_label = [part.strip() for part in task['parent'].split('==')]
x_col, y_col, label_col = task['x_marker'], task['y_marker'], task['annotation_column']
df = pd.read_parquet(parquet_path, columns=[parent_col, x_col, y_col, label_col])
parent = df.loc[df[parent_col] == parent_label, [x_col, y_col, label_col]].copy()
labels = parent[label_col].astype(object).fillna('Unassigned').replace('', 'Unassigned')
if set(labels.unique()) != {'MAITNKT', 'Unassigned'}:
    raise ValueError(f'Unexpected GT labels: {set(labels.unique())}')
counts = labels.value_counts()
expected = task['gt_summary']['n_cells_per_category']
if len(parent) != task['gt_summary']['n_parent_cells'] or any(int(counts[k]) != expected[k] for k in expected):
    raise ValueError('Parquet GT counts do not match task.json')
positive = labels.eq('MAITNKT').to_numpy()
x = parent[x_col].to_numpy(dtype=float)
y = parent[y_col].to_numpy(dtype=float)
if not np.isfinite(x).all() or not np.isfinite(y).all():
    raise ValueError('Non-finite plot coordinates')

fig, ax = plt.subplots(figsize=(8, 6.8), dpi=170)
ax.scatter(x[~positive], y[~positive], s=5, alpha=0.23, color='#64748b',
           label=f'Unassigned (n={int((~positive).sum()):,})', rasterized=True)
ax.scatter(x[positive], y[positive], s=12, alpha=0.92, color='#db2777',
           label=f'MAITNKT (n={int(positive.sum()):,})', rasterized=True)
ax.set_xlabel('CD4 (145Nd)', fontsize=12)
ax.set_ylabel('CD161 (151Eu)', fontsize=12)
ax.set_title(f'Acute2020 · Step 61 · {sample}\nGT labels within parent αβ T cells', fontsize=12)
ax.grid(alpha=0.12)
ax.legend(loc='best', frameon=True, fontsize=10, markerscale=2)
fig.tight_layout()
out = out_dir / 'acute2020_step61_maitnkt_gt_994570.png'
fig.savefig(out, dpi=170, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
print(f'Parent={len(parent)} MAITNKT={int(positive.sum())} Unassigned={int((~positive).sum())}')
PY
