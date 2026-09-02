#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-sample cluster prediction vs GT plot.

For each sample in the hierarchical eval_data, draws an N×3 grid where
each row is one step on its own (x_marker, y_marker) biaxial plane:
    * Col A: every parent cell tinted by its per-cell GT label
             (no rep overlay) — the raw ground truth.
    * Col B: parent cells tinted by the GT label of their assigned
             meta cluster; rep cells overlaid as large markers.
    * Col C: parent cells tinted by the LLM-predicted label of their
             cluster; rep cells whose prediction disagrees with their
             GT majority label get a red outer ring.

Invoked as a standalone CLI step after ``run_openai`` / ``run_vllm``
write pred.json — the runners do not auto-trigger plotting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.bench import BenchmarkLoader, parse_perturbations
from src.llm.utils.task_input_adapter import _TaskInputDataset, _step_from_task


def _palette(options: list[str]):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import cm
    cmap = cm.get_cmap('tab20', max(len(options), 1))
    return {lab: cmap(i % 20) for i, lab in enumerate(options)}, '#cccccc'


def _norm(s):
    return s.strip().lower() if isinstance(s, str) else s


def _slot_label_lookup(
    step_cells: list[dict],
    field: str,
) -> dict[int, str | None]:
    """Extract ``cell_N`` → ``cell[field]`` as a slot-int-keyed lookup."""
    out: dict[int, str | None] = {}
    for c in step_cells:
        cid = str(c.get('cell_id', ''))
        if not cid.startswith('cell_'):
            continue
        try:
            slot = int(cid.split('_', 1)[1]) - 1
        except ValueError:
            continue
        out[slot] = c.get(field)
    return out


def _draw_panel(
    ax, x_bg, y_bg, parent_cluster_labels,
    slot_to_label,
    x_rep, y_rep, rep_labels, wrong_mask,
    options, color_lookup, none_color, x_marker, y_marker,
    max_bg: int = 20000,
):
    """Parent cells coloured by the slot-label map + rep-cell overlay.

    ``slot_to_label`` maps cluster slot (0..K-1) → displayable label
    (GT or Prediction depending on the column). Parent cells inherit
    the label of their assigned slot and are rendered with the same
    categorical palette as the rep cells.
    """
    n_bg = len(x_bg)
    if n_bg > max_bg:
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(n_bg, max_bg, replace=False)
    else:
        bg_idx = np.arange(n_bg)

    bg_slots = parent_cluster_labels[bg_idx]
    bg_labels = np.asarray(
        [slot_to_label.get(int(s), None) for s in bg_slots], dtype=object,
    )
    bg_colors = np.array([
        color_lookup.get(l, none_color) if l is not None else none_color
        for l in bg_labels
    ])
    ax.scatter(x_bg[bg_idx], y_bg[bg_idx], s=4, c=bg_colors,
               alpha=0.35, rasterized=True, linewidths=0, zorder=1)

    rep_labels_arr = np.asarray(
        [l if l is not None else '__none__' for l in rep_labels], dtype=object,
    )
    for lab in list(options) + ['__none__']:
        mask = rep_labels_arr == lab
        if not mask.any():
            continue
        color = (color_lookup.get(lab, none_color)
                 if lab != '__none__' else none_color)
        display = lab if lab != '__none__' else 'None (parse-fail)'
        ax.scatter(x_rep[mask], y_rep[mask], s=90, c=[color],
                   edgecolors='black', linewidths=0.6, label=display,
                   zorder=4)

    if wrong_mask is not None and wrong_mask.any():
        ax.scatter(x_rep[wrong_mask], y_rep[wrong_mask], s=210,
                   facecolors='none', edgecolors='red', linewidths=1.4,
                   zorder=5)

    ax.set_xlabel(x_marker)
    ax.set_ylabel(y_marker)
    ax.legend(loc='best', fontsize=8, framealpha=0.85)


def _draw_per_cell_panel(
    ax, x_bg, y_bg, per_cell_labels,
    legend_order, color_lookup, none_color, x_marker, y_marker,
    max_bg: int = 20000,
):
    """Background scatter coloured by per-cell GT label (no overlay)."""
    n_bg = len(x_bg)
    if n_bg > max_bg:
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(n_bg, max_bg, replace=False)
    else:
        bg_idx = np.arange(n_bg)

    labels_arr = np.asarray(
        [l if l is not None else '__none__' for l in per_cell_labels],
        dtype=object,
    )[bg_idx]
    xb, yb = x_bg[bg_idx], y_bg[bg_idx]

    for lab in list(legend_order) + ['__none__']:
        mask = labels_arr == lab
        if not mask.any():
            continue
        color = (color_lookup.get(lab, none_color)
                 if lab != '__none__' else none_color)
        display = lab if lab != '__none__' else 'None'
        ax.scatter(xb[mask], yb[mask], s=4, c=[color],
                   alpha=0.5, rasterized=True, linewidths=0,
                   label=display, zorder=1)

    ax.set_xlabel(x_marker)
    ax.set_ylabel(y_marker)
    ax.legend(loc='best', fontsize=8, framealpha=0.85, markerscale=3)


def _prepare_step_data(
    ti, c2s_data: dict, mapping, step_cells: list[dict],
) -> dict | None:
    """Collect arrays + stats for one (sample, step). No plotting."""
    dataset = _TaskInputDataset(ti)
    step = _step_from_task(ti)

    parent_mask = dataset.get_parent_mask(step)
    parent_idx = np.where(parent_mask)[0]
    if len(parent_idx) == 0:
        return None

    x_all, y_all = dataset.get_xy_vals(step)
    x_bg = x_all[parent_idx]
    y_bg = y_all[parent_idx]
    gt_per_cell = np.asarray(dataset.get_gt_labels(step))[parent_idx]

    cluster_labels = mapping['cluster_labels']
    rep_parent_local = mapping['rep_parent_local']

    slot_to_gt = _slot_label_lookup(step_cells, 'gt')
    slot_to_pd = _slot_label_lookup(step_cells, 'prediction')

    x_r: list[float] = []
    y_r: list[float] = []
    gt_labels: list = []
    pd_labels: list = []
    for slot, local in enumerate(rep_parent_local):
        local = int(local)
        if local < 0:
            continue
        if slot not in slot_to_gt and slot not in slot_to_pd:
            continue
        x_r.append(float(x_bg[local]))
        y_r.append(float(y_bg[local]))
        gt_labels.append(slot_to_gt.get(slot))
        pd_labels.append(slot_to_pd.get(slot))

    if not x_r:
        return None

    options = c2s_data.get('options', [])
    extra_gt = [
        l for l in dict.fromkeys(gt_per_cell.tolist())
        if l is not None and l not in options
    ]
    palette_keys = list(options) + extra_gt
    color_lookup, none_color = _palette(palette_keys)
    wrong = np.asarray([
        _norm(p) != _norm(g) for p, g in zip(pd_labels, gt_labels)
    ])
    n_clusters = len(gt_labels)
    n_wrong = int(wrong.sum())
    acc = 1.0 - n_wrong / n_clusters if n_clusters else 0.0

    return {
        'step':                   int(ti.step),
        'x_bg':                   x_bg,
        'y_bg':                   y_bg,
        'parent_cluster_labels':  np.asarray(cluster_labels),
        'slot_to_gt':             slot_to_gt,
        'slot_to_pd':             slot_to_pd,
        'x_r':                    np.asarray(x_r),
        'y_r':                    np.asarray(y_r),
        'gt_labels':              gt_labels,
        'pd_labels':              pd_labels,
        'wrong':                  wrong,
        'options':                options,
        'gt_per_cell':            gt_per_cell,
        'palette_keys':           palette_keys,
        'color_lookup':           color_lookup,
        'none_color':             none_color,
        'x_marker':               c2s_data.get('x_marker', 'x'),
        'y_marker':               c2s_data.get('y_marker', 'y'),
        'n_clusters':             n_clusters,
        'n_wrong':                n_wrong,
        'accuracy':               acc,
    }


def plot_sample(
    sample_name: str,
    prepared_steps: list[dict],
    output_path: str,
) -> dict | None:
    """Render an N-row × 2-col GT-vs-Pred grid for one sample."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not prepared_steps:
        return None

    n_steps = len(prepared_steps)
    fig, axes = plt.subplots(
        n_steps, 3, figsize=(25.5, 7 * n_steps), squeeze=False,
    )

    total_clusters = total_wrong = 0
    for row, data in enumerate(prepared_steps):
        ax_all, ax_gt, ax_pd = axes[row, 0], axes[row, 1], axes[row, 2]
        _draw_per_cell_panel(
            ax_all, data['x_bg'], data['y_bg'], data['gt_per_cell'],
            data['palette_keys'], data['color_lookup'], data['none_color'],
            data['x_marker'], data['y_marker'],
        )
        ax_all.set_title(
            f'Step {data["step"]:02d} · GT (all events)  —  '
            f'{len(data["gt_per_cell"])} cells',
            fontsize=10,
        )
        _draw_panel(
            ax_gt, data['x_bg'], data['y_bg'],
            data['parent_cluster_labels'], data['slot_to_gt'],
            data['x_r'], data['y_r'], data['gt_labels'], None,
            data['options'], data['color_lookup'], data['none_color'],
            data['x_marker'], data['y_marker'],
        )
        ax_gt.set_title(
            f'Step {data["step"]:02d} · GT (cluster majority)  —  '
            f'{data["n_clusters"]} clusters',
            fontsize=10,
        )
        _draw_panel(
            ax_pd, data['x_bg'], data['y_bg'],
            data['parent_cluster_labels'], data['slot_to_pd'],
            data['x_r'], data['y_r'], data['pd_labels'], data['wrong'],
            data['options'], data['color_lookup'], data['none_color'],
            data['x_marker'], data['y_marker'],
        )
        correct = data['n_clusters'] - data['n_wrong']
        ax_pd.set_title(
            f'Step {data["step"]:02d} · Prediction  —  '
            f'{correct}/{data["n_clusters"]} correct '
            f'({data["accuracy"] * 100:.1f}%)',
            fontsize=10,
        )
        total_clusters += data['n_clusters']
        total_wrong += data['n_wrong']

    overall_acc = (1.0 - total_wrong / total_clusters) if total_clusters else 0.0
    fig.suptitle(
        f'Sample {sample_name}  —  {n_steps} steps · '
        f'{total_clusters - total_wrong}/{total_clusters} correct '
        f'({overall_acc * 100:.1f}%)',
        fontsize=12,
    )
    top = 1.0 - min(0.05, 0.4 / n_steps)
    fig.tight_layout(rect=[0, 0, 1, top])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    return {
        'n_steps':    n_steps,
        'n_clusters': total_clusters,
        'n_wrong':    total_wrong,
        'accuracy':   overall_acc,
    }


def plot_eval(
    eval_data: dict,
    dataset_name: str,
    benchmark_dir: str,
    data_dir: str,
    output_dir: str,
    loader: BenchmarkLoader | None = None,
    filename: str = 'c2s_pred_vs_gt.png',
) -> tuple[int, int]:
    """Walk eval_data hierarchy → one PNG per sample (all steps stacked).

    Output layout::

        {output_dir}/{sample}/{filename}
    """
    pred_meta = eval_data.get('meta', {}) if isinstance(eval_data, dict) else {}
    pred_slug = pred_meta.get('perturbation')
    loader = loader if loader is not None else BenchmarkLoader(benchmark_dir, data_dir)
    loader_slug = loader.perturbation_name
    if pred_slug != loader_slug:
        raise SystemExit(
            f'[ERROR] perturbation mismatch between pred.json and loader: '
            f'pred={pred_slug!r}, loader={loader_slug!r}. Pass the same '
            f'--perturbation spec as was used when generating c2s.'
        )
    c2s_filename = (f'c2s__{loader_slug}.json'
                    if loader_slug else 'c2s.json')
    mapping_filename = (f'cell2cluster__{loader_slug}.npz'
                        if loader_slug else 'cell2cluster.npz')

    out_root = Path(output_dir)
    n_done = n_skipped = 0

    samples = eval_data.get('samples', {})
    for sample_name in sorted(samples.keys()):
        sample_entry = samples[sample_name]
        prepared: list[dict] = []
        for step_str in sorted(sample_entry.get('steps', {}).keys(), key=int):
            step_entry = sample_entry['steps'][step_str]
            step_num = int(step_str)
            step_dir_name = f'step_{step_num:02d}'
            step_dir = (Path(benchmark_dir) / dataset_name / sample_name
                        / step_dir_name)
            task_json = step_dir / 'task.json'
            c2s_path = step_dir / c2s_filename
            mapping_path = step_dir / mapping_filename
            missing = [p.name for p in (task_json, c2s_path, mapping_path)
                       if not p.exists()]
            if missing:
                print(f'    [SKIP] {sample_name}/{step_dir_name}: '
                      f'missing {missing}')
                continue

            ti = loader.load_task(str(task_json))
            c2s_data = json.loads(c2s_path.read_text())
            mapping = np.load(mapping_path)

            step_cells = step_entry.get('cells', [])
            data = _prepare_step_data(ti, c2s_data, mapping, step_cells)
            if data is None:
                print(f'    [SKIP] {sample_name}/{step_dir_name}: '
                      f'no usable clusters')
                continue
            prepared.append(data)

        if not prepared:
            print(f'    [SKIP] {sample_name}: no plottable steps')
            n_skipped += 1
            continue

        out_path = out_root / sample_name / filename
        stats = plot_sample(sample_name, prepared, str(out_path))
        print(f'    [ok] {sample_name}: {stats["n_steps"]} steps · '
              f'{stats["n_clusters"] - stats["n_wrong"]}/'
              f'{stats["n_clusters"]} correct '
              f'({stats["accuracy"] * 100:.1f}%)  → {out_path}')
        n_done += 1

    print(f'\n  {n_done} sample plot(s) written under {out_root}/  '
          f'({n_skipped} skipped)')
    return n_done, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Per-sample cluster GT-vs-Pred plot from a pred.json',
    )
    parser.add_argument('--eval_path', required=True,
                        help='pred.json from run_openai / run_vllm')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (folder under benchmark/)')
    parser.add_argument('--benchmark', default='benchmark/')
    parser.add_argument('--data-dir', default='data/', dest='data_dir')
    parser.add_argument('--output_dir', required=True,
                        help='PNGs land under {output_dir}/{sample}/')
    parser.add_argument(
        '--perturbation', action='append', default=[],
        help='Inline perturbation spec (repeatable; must match the spec '
             'used at c2s time).',
    )
    parser.add_argument('--perturbation-name', default=None,
                        help='Override auto-derived perturbation slug.')
    from src.hard_depletions import add_hard_depletion_args, resolve_hard_depletion_args
    add_hard_depletion_args(parser)
    args = parser.parse_args()

    resolve_hard_depletion_args(args)

    perturbation_name, perturbation = parse_perturbations(
        args.perturbation or [], name_override=args.perturbation_name,
    )
    loader = BenchmarkLoader(
        args.benchmark, args.data_dir,
        perturbation=perturbation, perturbation_name=perturbation_name,
    )

    eval_data = json.loads(Path(args.eval_path).read_text())
    plot_eval(
        eval_data,
        dataset_name=args.dataset,
        benchmark_dir=args.benchmark,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        loader=loader,
    )


if __name__ == '__main__':
    main()
