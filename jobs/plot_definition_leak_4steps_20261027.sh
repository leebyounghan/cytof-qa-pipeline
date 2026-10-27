#!/usr/bin/env bash
#BSUB -J plot_definition_leak_4steps
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 5000
#BSUB -R "rusage[mem=5000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/results/_tmp_definition_leak_4steps_20261027/plot.%J.log

set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/results/_tmp_definition_leak_4steps_20261027/mplconfig

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
out_dir = root / 'results' / '_tmp_definition_leak_4steps_20261027'
hc_sample = 'exp_exp_Donor__A_panel-A_Clean cells_T cells'
specs = [
    ('FR-FCM-Z74D_hc', hc_sample, 6, 'Z74D hc · step 6 · Th1', 'z74d_hc_step06_th1.png',
     ['Unassigned', 'non_17_22_1', 'Th17_22', 'Th1']),
    ('FR-FCM-Z74D_hc', hc_sample, 11, 'Z74D hc · step 11 · Memory_Treg', 'z74d_hc_step11_memory_treg.png',
     ['Unassigned', 'Memory_Treg', 'Naive_Treg']),
    ('FR-FCM-Z74D_hc', hc_sample, 5, 'Z74D hc · step 5 · non_Tfh', 'z74d_hc_step05_non_tfh.png',
     ['Unassigned', 'non_Tfh', 'Tfh', 'Tfh_hi_PD1']),
    ('Acute2020', '994590_Normalized', 53, 'Acute2020 · step 53 · Treg/activated',
     'acute2020_step53_treg_activated.png',
     ['Unassigned', 'Treg/CD38pos', 'Treg/activated', 'Treg/activated+CD38pos']),
]
colors = {
    'Unassigned': '#a3afbf', 'non_17_22_1': '#a3afbf', 'non_Tfh': '#a3afbf',
    'Th17_22': '#2563eb', 'Th1': '#e11d48', 'Memory_Treg': '#2563eb',
    'Naive_Treg': '#e11d48', 'Tfh': '#2563eb', 'Tfh_hi_PD1': '#e11d48',
    'Treg/CD38pos': '#2563eb', 'Treg/activated': '#e11d48',
    'Treg/activated+CD38pos': '#9333ea',
}

def load_step(cohort, sample, step):
    task_path = root / 'benchmark' / cohort / sample / f'step_{step:02d}' / 'task.json'
    data_path = root / 'data' / cohort / 'parquet' / f'{sample}.parquet'
    task = json.loads(task_path.read_text())
    if (task['dataset'], task['sample'], task['step']) != (cohort, sample, step):
        raise ValueError(f'Task identity mismatch: {task_path}')
    meta = json.loads((root / 'data' / cohort / 'parquet' / 'meta.json').read_text())
    schema_cols = pq.ParquetFile(data_path).schema.names
    channel_map = {channel: info[0] for channel, info in meta['channels'].items()}
    x_col = resolve_channel(task['x_marker'], schema_cols, channel_map)
    y_col = resolve_channel(task['y_marker'], schema_cols, channel_map)
    if x_col is None or y_col is None:
        raise KeyError(f'Cannot resolve markers for {cohort} step {step}')
    parent_cols = []
    for clause in task['parent'].split('&'):
        match = re.fullmatch(r'\s*(.*?)\s+(?:==|!=|in)\s+.+\s*', clause)
        if match is None:
            raise ValueError(f'Unsupported parent condition: {task["parent"]}')
        parent_cols.append(match.group(1))
    label_col = task['annotation_column']
    columns = list(dict.fromkeys([*parent_cols, x_col, y_col, label_col]))
    df = pd.read_parquet(data_path, columns=columns)
    parent = df.loc[get_parent_mask(df, task['parent']), [x_col, y_col, label_col]]
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

def plot(ax, task, x, y, labels, counts, title, order):
    if set(order) != set(task['categories']):
        raise ValueError(f'Plot order differs from task categories: {title}')
    for category in order:
        selected = labels == category
        if not selected.any():
            continue
        background = category in {'Unassigned', 'non_17_22_1', 'non_Tfh'}
        ax.scatter(x[selected], y[selected], s=3 if background else 6,
                   alpha=0.13 if background else 0.60,
                   color=colors[category], rasterized=True,
                   label=f'{category} ({counts[category]:,})')
    ax.set_xlabel(task['x_marker'])
    ax.set_ylabel(task['y_marker'])
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.12)
    ax.legend(loc='best', fontsize=7, markerscale=2.5, frameon=True)

overview, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=160)
for ax, (cohort, sample, step, title, filename, order) in zip(axes.flat, specs):
    task, x, y, labels, counts = load_step(cohort, sample, step)
    plot(ax, task, x, y, labels, counts, title, order)
    fig_one, ax_one = plt.subplots(figsize=(8, 6.5), dpi=170)
    plot(ax_one, task, x, y, labels, counts, f'{title}\n{sample}', order)
    fig_one.tight_layout()
    fig_one.savefig(out_dir / filename, bbox_inches='tight')
    plt.close(fig_one)
    print(f'{cohort} step {step}: {sample}; parent={len(labels)}; counts={counts}; saved={filename}')
overview.tight_layout()
overview.savefig(out_dir / 'four_step_gt_overview.png', bbox_inches='tight')
plt.close(overview)
print(f'Saved {out_dir / "four_step_gt_overview.png"}')
PY
