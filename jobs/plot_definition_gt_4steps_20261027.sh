#!/usr/bin/env bash
#BSUB -J plot_definition_gt_4steps
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 5000
#BSUB -R "rusage[mem=5000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/results/_tmp_definition_gt_4steps_20261027/plot.%J.log

set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/results/_tmp_definition_gt_4steps_20261027/mplconfig

.venv/bin/python - <<'PY'
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.preprocess import get_parent_mask, resolve_channel

root = Path('/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline')
out_dir = root / 'results' / '_tmp_definition_gt_4steps_20261027'
specs = [
    ('FR-FCM-Z74D_hc', 'exp_exp_Donor__A_panel-A_Clean cells_T cells', 5,
     'Z74D hc · step 5 · Tfh / Tfh_hi_PD1', 'z74d_hc_step05_tfh.png'),
    ('FR-FCM-Z74D_hc', 'exp_exp_Donor__A_panel-A_Clean cells_T cells', 7,
     'Z74D hc · step 7 · Th17 / Th22', 'z74d_hc_step07_th17_th22.png'),
    ('Bjornson', 'R15W11_W070515101208_Hu_M_Basal', 3,
     'Bjornson · step 3 · Granulocyte', 'bjornson_step03_granulocyte.png'),
    ('Lyoplate_DC', '1349_1_DcMonNk_D04', 2,
     'Lyoplate DC · step 2 · CD14pos_CD16pos', 'lyoplate_dc_step02_cd16pos.png'),
]
palette = {
    'Unassigned': '#94a3b8', 'non_Tfh': '#94a3b8',
    'Tfh': '#2563eb', 'Tfh_hi_PD1': '#e11d48',
    'Th17': '#2563eb', 'Th22': '#e11d48',
    'Granulocyte': '#e11d48', 'CD45hiCD66lo': '#2563eb',
    'CD14pos_CD16pos': '#e11d48',
}

def load_step(cohort, sample, step):
    task_path = root / 'benchmark' / cohort / sample / f'step_{step:02d}' / 'task.json'
    data_path = root / 'data' / cohort / 'parquet' / f'{sample}.parquet'
    task = json.loads(task_path.read_text())
    if task['dataset'] != cohort or task['sample'] != sample or task['step'] != step:
        raise ValueError(f'Task identity mismatch: {task_path}')
    meta = json.loads((root / 'data' / cohort / 'parquet' / 'meta.json').read_text())
    schema_cols = pq.ParquetFile(data_path).schema.names
    marker_map = {channel: info[0] for channel, info in meta['channels'].items()}
    x_col = resolve_channel(task['x_marker'], schema_cols, marker_map)
    y_col = resolve_channel(task['y_marker'], schema_cols, marker_map)
    if x_col is None or y_col is None:
        raise KeyError(f'Cannot resolve marker columns: {cohort} step {step}')
    parent_cols = []
    for clause in task['parent'].split('&'):
        match = re.fullmatch(r'\s*(.*?)\s+(?:==|!=|in)\s+.+\s*', clause)
        if match is None:
            raise ValueError(f'Unsupported parent condition: {task["parent"]}')
        parent_cols.append(match.group(1))
    label_col = task['annotation_column']
    columns = list(dict.fromkeys([*parent_cols, x_col, y_col, label_col]))
    df = pd.read_parquet(data_path, columns=columns)
    mask = get_parent_mask(df, task['parent'])
    parent = df.loc[mask, [x_col, y_col, label_col]]
    labels = parent[label_col].astype(object).fillna('Unassigned').replace('', 'Unassigned').to_numpy(dtype=object)
    x = parent[x_col].to_numpy(dtype=float)
    y = parent[y_col].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f'Non-finite axis value: {cohort} step {step}')
    counts = {category: int(np.sum(labels == category)) for category in task['categories']}
    expected = task['gt_summary']['n_cells_per_category']
    if len(parent) != task['gt_summary']['n_parent_cells'] or counts != expected:
        raise ValueError(f'GT count mismatch: {cohort} step {step}: {counts} != {expected}')
    return task, x, y, labels, counts

def plot(ax, task, x, y, labels, counts, title):
    category_order = sorted(task['categories'], key=lambda c: (c not in {'Unassigned', 'non_Tfh'}, c))
    for category in category_order:
        select = labels == category
        if not select.any():
            continue
        background = category in {'Unassigned', 'non_Tfh'}
        ax.scatter(x[select], y[select], s=3 if background else 5,
                   alpha=0.14 if background else 0.57,
                   color=palette[category], rasterized=True,
                   label=f'{category} ({counts[category]:,})')
    ax.set_xlabel(task['x_marker'])
    ax.set_ylabel(task['y_marker'])
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.12)
    ax.legend(loc='best', fontsize=7, markerscale=2.5, frameon=True)

overview, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=160)
for ax, (cohort, sample, step, title, filename) in zip(axes.flat, specs):
    task, x, y, labels, counts = load_step(cohort, sample, step)
    plot(ax, task, x, y, labels, counts, title)
    fig_one, ax_one = plt.subplots(figsize=(8, 6.5), dpi=170)
    plot(ax_one, task, x, y, labels, counts, f'{title}\n{sample}')
    fig_one.tight_layout()
    fig_one.savefig(out_dir / filename, bbox_inches='tight')
    plt.close(fig_one)
    print(f'{cohort} step {step}: {sample}; parent={len(labels)}; counts={counts}; saved={filename}')
overview.tight_layout()
overview.savefig(out_dir / 'four_step_gt_overview.png', bbox_inches='tight')
plt.close(overview)
print(f'Saved {out_dir / "four_step_gt_overview.png"}')
PY
