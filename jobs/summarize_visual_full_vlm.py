"""Equal-cohort full-test comparison of VLM variants with LLM-Gate."""

import json
from pathlib import Path
from statistics import mean


DATASETS = (
    'Acute2020', 'Acute2021', 'Bjornson', 'Vaccine',
    'FR-FCM-Z74D_hc', 'FR-FCM-Z74D_tissue', 'FRDR_covid19',
    'Lyoplate_DC', 'Lyoplate_bcell', 'Lyoplate_tcell', 'Lyoplate_treg',
)
ROOT = Path('results/visual_luna_medium_definition/full_concurrency40')
BASELINE = Path('results/gate_gpt-6-luna-medium_full')
VLM_CONDITIONS = ('llm_gate', 'vlm_scatter', 'vlm_contour', 'vlm_density')
CONDITIONS = VLM_CONDITIONS + ('agent_scatter', 'agent_contour', 'agent_density')
METRICS = ('f1_macro_mean', 'balanced_accuracy_mean', 'iou_hull_macro_mean')


def main(conditions: tuple[str, ...] = VLM_CONDITIONS,
         output_prefix: str = 'vlm_full_comparison',
         root: Path = ROOT, baseline_root: Path = BASELINE,
         model_label: str = 'GPT-6 Luna medium') -> None:
    if not conditions or conditions[0] != 'llm_gate':
        raise ValueError('The first condition must be llm_gate')
    if (baseline_root / 'notes.yaml.used').read_bytes() != (
        root.parent / 'notes.yaml.used').read_bytes()
    ):
        raise ValueError('Definition snapshots differ between LLM and visual runs')
    rows = []
    for condition in conditions:
        base = baseline_root if condition == 'llm_gate' else root / condition
        cohorts = []
        for dataset in DATASETS:
            path = base / 'predictions' / dataset / 'eval_summary.json'
            summary = json.loads(path.read_text())
            if summary['n_tasks'] < 1:
                raise ValueError(f'Empty evaluation: {path}')
            cohorts.append({'dataset': dataset, 'n_tasks': summary['n_tasks'],
                            **{metric: summary['overall'][metric] for metric in METRICS}})
        row = {'condition': condition, 'n_tasks': sum(c['n_tasks'] for c in cohorts),
               'cohorts': cohorts}
        for metric in METRICS:
            row[metric] = mean(c[metric] for c in cohorts)
        rows.append(row)
    baseline = rows[0]
    for row in rows[1:]:
        for cohort, llm_cohort in zip(row['cohorts'], baseline['cohorts']):
            if (cohort['dataset'], cohort['n_tasks']) != (
                llm_cohort['dataset'], llm_cohort['n_tasks']
            ):
                raise ValueError(f'Task mismatch: {row["condition"]}, {cohort["dataset"]}')
        for metric in METRICS:
            row[f'delta_{metric}'] = row[metric] - baseline[metric]
    out = root / f'{output_prefix}.json'
    out.write_text(json.dumps({'aggregation': 'equal-cohort mean; full test split',
                               'rows': rows}, ensure_ascii=False, indent=2) + '\n')

    lines = ['# Full test: Gate variants versus LLM-Gate', '',
             f'All rows use {model_label}, the same definition snapshot,',
             'the test split, and an equal mean of 11 cohort scores.', '',
             '| Method | Steps | Macro F1 | Δ F1 vs LLM | Balanced accuracy | Hull IoU |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["condition"]} | {row["n_tasks"]} | '
                     f'{row["f1_macro_mean"]:.3f} | '
                     f'{row.get("delta_f1_macro_mean", 0.0):+.3f} | '
                     f'{row["balanced_accuracy_mean"]:.3f} | '
                     f'{row["iou_hull_macro_mean"]:.3f} |')
    lines += ['', '## F1 by cohort', '',
              '| Cohort | Steps | ' + ' | '.join(conditions) + ' |',
              '|---|---:|' + '---:|' * len(conditions)]
    for idx, dataset in enumerate(DATASETS):
        lines.append(f'| {dataset} | {baseline["cohorts"][idx]["n_tasks"]} | '
                     f'{baseline["cohorts"][idx]["f1_macro_mean"]:.3f} | '
                     + ' | '.join(f'{row["cohorts"][idx]["f1_macro_mean"]:.3f}'
                                  for row in rows[1:]) + ' |')
    lines += ['', 'The visual methods use different prompts from LLM-Gate;',
              'method differences cannot be attributed solely to image style.']
    markdown = root / f'{output_prefix}.md'
    markdown.write_text('\n'.join(lines) + '\n')
    print(markdown.read_text(), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS,
                        default=list(VLM_CONDITIONS))
    parser.add_argument('--output-prefix', default='vlm_full_comparison')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--baseline-root', type=Path, default=BASELINE)
    parser.add_argument('--model-label', default='GPT-6 Luna medium')
    args = parser.parse_args()
    main(tuple(args.conditions), args.output_prefix,
         args.root, args.baseline_root, args.model_label)
