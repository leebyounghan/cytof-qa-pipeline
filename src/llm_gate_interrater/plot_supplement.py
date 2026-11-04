"""Publication-style shared-parent GT and LLM-Gate comparison plates."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import pandas as pd

from src.eval import compute_metrics
from .run_sample import DATA, GATERS, META, OUT, PLAN


STEPS = (2, 4, 5)
FIGURES = OUT / 'figures'
COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728')
GREY = '#9ca3af'
MAX_DISPLAY = 22000


def _step(number: int) -> dict:
    return next(s for s in PLAN['steps'] if s['step'] == number)


def _parent_column(step: dict, gater: str) -> tuple[str, str] | None:
    if step['parent'] == 'ALL':
        return None
    column, category = (p.strip() for p in step['parent'].split(' == ', 1))
    return META['gater_columns'][gater][column], category


def _task_f1(metrics: dict, number: int) -> float:
    step = _step(number)
    categories = [*step['annotation categories'], 'Unassigned']
    entries = metrics['steps'][str(number)]['category_metrics']
    return float(np.mean([entries[c]['f1'] or 0.0 for c in categories]))


def _robust_distance(values: np.ndarray, index: int) -> float:
    median = np.median(values)
    spread = np.percentile(values, 75) - np.percentile(values, 25)
    return abs(float(values[index] - median)) / max(float(spread), 0.01)


def choose_sample() -> dict:
    """Choose a typical sample using saved F1 and four-parent overlap."""
    samples = [item['sample'] for item in json.loads(
        (DATA / 'dataset_manifest.json').read_text())['files']]
    rows = []
    for sample in samples:
        metrics = {g: json.loads((OUT / sample / g / 'metrics.json').read_text())
                   for g in GATERS}
        f1 = {str(n): float(np.mean([_task_f1(metrics[g], n) for g in GATERS]))
              for n in STEPS}
        gt_counts = {}
        for n in STEPS:
            step = _step(n)
            for g in GATERS:
                path = OUT / sample / g / 'benchmark' / sample / f'step_{n:02d}' / 'gate_task.json'
                task = json.loads(path.read_text())
                for cat in step['annotation categories']:
                    gt_counts[(n, g, cat)] = int(task['gt_category_counts'].get(cat, 0))
        parent_cols = sorted({value[0] for n in STEPS for g in GATERS
                              if (value := _parent_column(_step(n), g)) is not None})
        frame = pd.read_parquet(DATA / 'parquet' / f'{sample}.parquet', columns=parent_cols)
        overlaps, shared_counts = {}, {}
        for n in STEPS:
            masks = []
            for g in GATERS:
                col, cat = _parent_column(_step(n), g)
                masks.append(frame[col].to_numpy() == cat)
            common = np.logical_and.reduce(masks)
            union = np.logical_or.reduce(masks)
            shared_counts[str(n)] = int(common.sum())
            overlaps[str(n)] = float(common.sum() / union.sum()) if union.any() else 0.0
        rows.append({'sample': sample, 'f1': f1, 'overlap': overlaps,
                     'shared_counts': shared_counts,
                     'min_named_gt': min(gt_counts.values())})
    eligible = [row for row in rows if row['min_named_gt'] >= 30
                and all(row['shared_counts'][str(n)] >= 1000 for n in STEPS)]
    if not eligible:
        raise ValueError('No sample has all named gates and sufficient common-parent cells')
    for row in eligible:
        score = 0.0
        idx = eligible.index(row)
        for n in STEPS:
            key = str(n)
            for field in ('f1', 'overlap'):
                vals = np.array([r[field][key] for r in eligible])
                score += _robust_distance(vals, idx)
            counts = np.log1p([r['shared_counts'][key] for r in eligible])
            score += 0.35 * _robust_distance(np.asarray(counts), idx)
        row['selection_score'] = score
    eligible.sort(key=lambda r: (r['selection_score'], r['sample']))
    return {'selection_rule': 'minimum sum of median/IQR deviations for mean rater F1, four-parent overlap, and log common-parent size across steps 2, 4, 5; each named GT >=30 cells, each common parent >=1000',
            'n_eligible': len(eligible), 'chosen': eligible[0],
            'top_10': eligible[:10]}


def _clean(series: pd.Series) -> np.ndarray:
    return series.astype(object).fillna('Unassigned').replace('', 'Unassigned').to_numpy(dtype=str)


def load_step(sample: str, number: int) -> dict:
    step = _step(number)
    canonical = step['annotation column name']
    parent_cols = [value[0] for g in GATERS
                   if (value := _parent_column(step, g)) is not None]
    gt_cols = [META['gater_columns'][g][canonical] for g in GATERS]
    columns = list(dict.fromkeys([step['x_marker'], step['y_marker'], *parent_cols, *gt_cols]))
    frame = pd.read_parquet(DATA / 'parquet' / f'{sample}.parquet', columns=columns)
    masks = {}
    for g in GATERS:
        parent = _parent_column(step, g)
        masks[g] = (np.ones(len(frame), dtype=bool) if parent is None
                    else frame[parent[0]].to_numpy() == parent[1])
    common = np.logical_and.reduce(list(masks.values()))
    union = np.logical_or.reduce(list(masks.values()))
    if not common.any():
        raise ValueError(f'No shared parent at step {number}')
    x_all = frame[step['x_marker']].to_numpy(dtype=np.float32)
    y_all = frame[step['y_marker']].to_numpy(dtype=np.float32)
    gt = {g: _clean(frame[META['gater_columns'][g][canonical]][common]) for g in GATERS}
    pred, gates, f1 = {}, {}, {}
    options = [*step['annotation categories'], 'Unassigned']
    for g in GATERS:
        base = OUT / sample / g
        saved = pd.read_parquet(base / 'predicted_labels.parquet', columns=[canonical])
        if len(saved) != len(frame):
            raise ValueError(f'Step {number}/{g}: prediction length mismatch')
        pred[g] = _clean(saved[canonical][common])
        if not set(np.unique(pred[g])).issubset(options):
            raise ValueError(f'Step {number}/{g}: unexpected prediction label')
        pred_file = json.loads((base / 'pred.json').read_text())
        gates[g] = pred_file['samples'][sample]['steps'][str(number)]['gates']
        f1[g] = compute_metrics(gt[g], pred[g], options)['f1_macro']
    return {'number': number, 'step': step, 'x': x_all[common], 'y': y_all[common],
            'gt': gt, 'pred': pred, 'gates': gates, 'f1': f1,
            'n_common': int(common.sum()), 'n_union': int(union.sum()),
            'n_parent': {g: int(masks[g].sum()) for g in GATERS}}


def _display_indices(n: int, number: int) -> np.ndarray:
    if n <= MAX_DISPLAY:
        return np.arange(n)
    return np.sort(np.random.default_rng(1100 + number).choice(n, MAX_DISPLAY, replace=False))


def _limits(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.001, 0.999])
    width = max(float(hi - lo), 0.1)
    return float(lo - 0.04 * width), float(hi + 0.04 * width)


def _scatter(ax, x, y, labels, categories, cmap):
    for cat in ['Unassigned', *categories]:
        selected = labels == cat
        if not selected.any():
            continue
        background = cat == 'Unassigned'
        ax.scatter(x[selected], y[selected],
                   s=1.5 if background else 2.6,
                   alpha=0.22 if background else 0.48,
                   color=cmap[cat], linewidths=0, rasterized=True,
                   zorder=1 if background else 2)


def _gate_items(value):
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return [value] if isinstance(value, dict) else []


def _outlines(ax, gates: dict, categories: list[str], cmap, annotate=True):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for cat in categories:
        for gate in _gate_items(gates.get(cat)):
            if gate.get('absent'):
                continue
            color = cmap[cat]
            if 'vertices' in gate:
                points = np.asarray(gate['vertices'], dtype=float)
                patch = Polygon(points, closed=True, fill=False,
                                edgecolor=color, linewidth=1.3, zorder=4)
                anchor = points[0]
            else:
                required = ('x_min', 'x_max', 'y_min', 'y_max')
                if not all(k in gate for k in required):
                    raise ValueError(f'Invalid gate for {cat}: {gate}')
                x0, x1 = float(gate['x_min']), float(gate['x_max'])
                y0, y1 = float(gate['y_min']), float(gate['y_max'])
                if x1 <= x0 or y1 <= y0:
                    raise ValueError(f'Degenerate gate for {cat}: {gate}')
                patch = Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                  edgecolor=color, linewidth=1.3, zorder=4)
                anchor = (x0, y1)
            ax.add_patch(patch)
            if (annotate and xlim[0] <= anchor[0] <= xlim[1]
                    and ylim[0] <= anchor[1] <= ylim[1]):
                ax.annotate(cat, anchor, fontsize=5.7, color=color,
                            ha='left', va='bottom', clip_on=True,
                            bbox={'facecolor': 'white', 'edgecolor': color,
                                  'linewidth': 0.45, 'pad': 0.7, 'alpha': 0.9},
                            zorder=5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def render(step_data: list[dict], output_stem: Path) -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    rows = 2 * len(step_data)
    fig = plt.figure(figsize=(12.6, 2.4 * rows + 0.65 * len(step_data) + 0.9),
                     facecolor='white')
    grid = fig.add_gridspec(3 * len(step_data), 4,
                            height_ratios=[0.34, 2.4, 2.4] * len(step_data),
                            left=0.075, right=0.985, top=0.935,
                            bottom=0.11 if len(step_data) == 1 else 0.055,
                            wspace=0.12, hspace=0.19)
    headers = [fig.add_subplot(grid[3 * block, :])
               for block in range(len(step_data))]
    axes = np.asarray([[fig.add_subplot(grid[3 * (row // 2) + 1 + row % 2, j])
                        for j in range(4)] for row in range(rows)])
    for j, _ in enumerate(GATERS):
        box = axes[0, j].get_position()
        fig.text((box.x0 + box.x1) / 2, 0.972, f'Gater {chr(65 + j)}',
                 ha='center', va='center', fontsize=10, fontweight='bold')
    for block, data in enumerate(step_data):
        number = data['number']
        step = data['step']
        categories = step['annotation categories']
        cmap = {'Unassigned': GREY,
                **{cat: COLORS[k] for k, cat in enumerate(categories)}}
        header = headers[block]
        header.set_axis_off()
        fraction = data['n_common'] / data['n_union']
        header.text(0, 0.5,
                    f'Step {number:02d}  ·  {META["channels"][step["x_marker"]][0]} × '
                    f'{META["channels"][step["y_marker"]][0]}',
                    transform=header.transAxes, va='center', ha='left',
                    fontsize=9, fontweight='bold')
        header.text(0.34, 0.5,
                    f'Shared parent: {data["n_common"]:,} cells ({fraction:.1%} of union)',
                    transform=header.transAxes, va='center', ha='left', fontsize=8)
        for k, cat in enumerate(categories):
            legend_x = 0.68 + 0.105 * k
            header.scatter([legend_x], [0.5], s=22, color=cmap[cat],
                           transform=header.transAxes, clip_on=False)
            header.text(legend_x + 0.012, 0.5, cat,
                        transform=header.transAxes, va='center', ha='left',
                        fontsize=7.5)
        idx = _display_indices(data['n_common'], number)
        x, y = data['x'][idx], data['y'][idx]
        xlim, ylim = _limits(data['x']), _limits(data['y'])
        for j, g in enumerate(GATERS):
            for kind in ('GT', 'LLM'):
                row = 2 * block + (kind == 'LLM')
                ax = axes[row, j]
                labels = (data['gt'][g] if kind == 'GT' else data['pred'][g])[idx]
                _scatter(ax, x, y, labels, categories, cmap)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if kind == 'LLM':
                    _outlines(ax, data['gates'][g], categories, cmap)
                    ax.text(0.98, 0.97, f'F1 = {data["f1"][g]:.3f}',
                            transform=ax.transAxes, ha='right', va='top',
                            fontsize=7, bbox={'facecolor': 'white',
                                              'edgecolor': 'none', 'alpha': 0.86},
                            zorder=6)
                if kind == 'LLM':
                    ax.set_xlabel(META['channels'][step['x_marker']][0])
                if j == 0:
                    ax.set_ylabel(META['channels'][step['y_marker']][0])
                else:
                    ax.tick_params(labelleft=False)
                ax.grid(color='#d1d5db', linewidth=0.35, alpha=0.35)
                ax.tick_params(length=2, pad=1.5)
        gt_box = axes[2*block, 0].get_position()
        llm_box = axes[2*block+1, 0].get_position()
        fig.text(0.018, (gt_box.y0+gt_box.y1)/2, 'Expert GT', rotation=90,
                 ha='center', va='center', fontsize=8, fontweight='bold')
        fig.text(0.018, (llm_box.y0+llm_box.y1)/2, 'LLM-Gate', rotation=90,
                 ha='center', va='center', fontsize=8, fontweight='bold')
    fig.text(0.53, 0.019,
             f'Same shared-parent cells in all four columns (up to {MAX_DISPLAY:,} plotted per step); '
             'grey = Unassigned. Rectangles = saved LLM gates.',
             ha='center', va='bottom', fontsize=7.2)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.05)
    fig.savefig(output_stem.with_suffix('.png'), dpi=230, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def render_publication(step_data: list[dict], output_stem: Path) -> None:
    """Compact 2 × 2-step plate, with explanatory details in the caption."""
    if len(step_data) != 2 or step_data[0]['number'] == step_data[1]['number']:
        raise ValueError('Publication plate requires two distinct steps')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'axes.labelsize': 7, 'xtick.labelsize': 6,
                         'ytick.labelsize': 6, 'pdf.fonttype': 42,
                         'ps.fonttype': 42})
    fig = plt.figure(figsize=(7.45, 7.25), facecolor='white')
    grid = fig.add_gridspec(6, 4, height_ratios=[0.25, 1.75, 1.75] * 2,
                            left=0.09, right=0.99, top=0.935, bottom=0.055,
                            wspace=0.12, hspace=0.24)
    for j in range(4):
        center = (0.09 + j * (0.90 / 4) + 0.90 / 8)
        fig.text(center, 0.971, f'Gater {chr(65+j)}', ha='center',
                 va='center', fontsize=8.5, fontweight='bold')
    for block, data in enumerate(step_data):
        step = data['step']
        categories = step['annotation categories']
        cmap = {'Unassigned': '#adb5bd',
                **{cat: COLORS[k] for k, cat in enumerate(categories)}}
        header = fig.add_subplot(grid[3*block, :])
        header.set_axis_off()
        header.text(0, 0.5,
                    f'{chr(65+block)}  ·  {META["channels"][step["x_marker"]][0]} × '
                    f'{META["channels"][step["y_marker"]][0]}',
                    transform=header.transAxes, ha='left', va='center',
                    fontsize=8.2, fontweight='bold')
        legend_x = {
            2: [0.67, 0.85],
            3: [0.50, 0.69, 0.86],
            4: [0.39, 0.55, 0.70, 0.86],
        }[len(categories) + 1]
        for x0, category in zip(legend_x, [*categories, 'Unassigned']):
            header.scatter([x0], [0.5], s=15, color=cmap[category],
                           transform=header.transAxes, clip_on=False)
            header.text(x0+0.012, 0.5, category, transform=header.transAxes,
                        va='center', fontsize=6.7)
        n = data['n_common']
        rng = np.random.default_rng(1100 + data['number'])
        idx = np.sort(rng.choice(n, min(n, 16000), replace=False))
        x, y = data['x'][idx], data['y'][idx]
        xlim, ylim = _limits(data['x']), _limits(data['y'])
        for j, gater in enumerate(GATERS):
            for kind in ('GT', 'LLM'):
                row = 3*block + 1 + (kind == 'LLM')
                ax = fig.add_subplot(grid[row, j])
                labels = (data['gt'][gater] if kind == 'GT'
                          else data['pred'][gater])[idx]
                for category in ['Unassigned', *categories]:
                    selected = labels == category
                    if not selected.any():
                        continue
                    ax.scatter(x[selected], y[selected],
                               s=0.7 if category == 'Unassigned' else 1.25,
                               alpha=0.18 if category == 'Unassigned' else 0.48,
                               color=cmap[category], linewidths=0,
                               rasterized=True)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                if kind == 'LLM':
                    _outlines(ax, data['gates'][gater], categories, cmap,
                              annotate=False)
                    ax.set_xlabel(META['channels'][step['x_marker']][0],
                                  labelpad=2)
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0:
                    ax.set_ylabel(META['channels'][step['y_marker']][0],
                                  labelpad=2)
                else:
                    ax.tick_params(labelleft=False)
                ax.tick_params(length=2, width=0.6, pad=1)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.65)
                    spine.set_color('#3f3f46')
                if j == 0:
                    box = ax.get_position()
                    fig.text(0.027, (box.y0+box.y1)/2,
                             'GT' if kind == 'GT' else 'LLM-Gate',
                             rotation=90, ha='center', va='center',
                             fontsize=7.5, fontweight='bold')
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.025)
    fig.savefig(output_stem.with_suffix('.png'), dpi=300,
                bbox_inches='tight', pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Submit plot computation through bsub')
    FIGURES.mkdir(parents=True, exist_ok=True)
    choice = choose_sample()
    (FIGURES / 'selection.json').write_text(json.dumps(choice, indent=2))
    sample = choice['chosen']['sample']
    data = [load_step(sample, n) for n in STEPS]
    for d in data:
        render([d], FIGURES / f'interrater_step{d["number"]:02d}')
    render(data, FIGURES / 'interrater_steps02_04_05')
    render_publication([d for d in data if d['number'] in (4, 5)],
                       FIGURES / 'interrater_steps04_05')
    (FIGURES / 'interrater_steps04_05_caption.txt').write_text(
        'Supplementary Figure. Human ground-truth labels and rater-conditioned '
        'GPT-6 Luna (medium) predictions for MDIPA interrater steps 4 and 5. '
        'Each column corresponds to one expert. The upper panel shows that '
        'expert’s ground-truth cell labels, and the lower panel shows the '
        'corresponding LLM predictions and saved LLM gate outlines. All four '
        'columns within a step contain the same cells: the intersection of '
        'the four experts’ ground-truth parent populations. Axes and displayed '
        'events are matched within each step. Human gate polygons were not '
        'available, so no human boundaries are inferred or shown. Grey denotes '
        'Unassigned. Colored outlines are the saved LLM gates. '
        'Only the scatter display is subsampled; all metrics use complete data. '
        'The representative sample was selected by the pre-specified median '
        'proximity rule recorded in selection.json. The common-parent count '
        'was 76,878 cells for step 4 and 111,369 for step 5, corresponding '
        'to 91.7% and 92.6% of the respective four-parent unions.\n')
    caption = (
        'Supplementary Figure. Expert ground truth and rater-conditioned LLM-Gate '
        'on the shared parent population at Steps 2, 4, and 5 of the MDIPA '
        'interrater tree. The same cells and axes are used in all four columns '
        'within each step. The upper panel of each step shows each expert’s '
        'labels; the lower panel shows GPT-6 Luna (medium) predictions from '
        'the run conditioned on that expert’s GT parent. Colored outlines are '
        'the saved LLM gates; human gate polygons were not available in this '
        'dataset and are not reconstructed. Grey denotes Unassigned. Reported '
        'F1 is per-task macro F1 including Unassigned on the shared parent. '
        'Points are uniformly subsampled for display only; scores and parent '
        'overlap use all events. A single representative sample was selected '
        'before plotting by median proximity in LLM F1, four-rater parent '
        'overlap, and shared-parent size across the three steps; see selection.json. '
        'The shared-parent view excludes upstream parent-discordant cells, '
        'whose extent is indicated by the shared/union parent fraction.\n'
    )
    (FIGURES / 'caption.txt').write_text(caption)
    print('Completed supplementary interrater figure', flush=True)


if __name__ == '__main__':
    main()
