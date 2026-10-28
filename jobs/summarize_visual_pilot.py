"""Equal-cohort summary of the one-sample visual gate pilot."""

import json
from pathlib import Path
from statistics import mean


ROOT = Path('results/visual_luna_medium_definition/pilot_concurrency40')
CONDITIONS = tuple(f'{mode}_{style}' for mode in ('vlm', 'agent')
                   for style in ('scatter', 'contour', 'density'))
DATASETS = (
    'Acute2020', 'Acute2021', 'Bjornson', 'Vaccine',
    'FR-FCM-Z74D_hc', 'FR-FCM-Z74D_tissue', 'FRDR_covid19',
    'Lyoplate_DC', 'Lyoplate_bcell', 'Lyoplate_tcell', 'Lyoplate_treg',
)
METRICS = ('f1_macro_mean', 'balanced_accuracy_mean', 'iou_hull_macro_mean')


def main() -> None:
    rows = []
    for condition in CONDITIONS:
        summaries = []
        for dataset in DATASETS:
            path = ROOT / condition / 'predictions' / dataset / 'eval_summary.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            summary = json.loads(path.read_text())
            if summary['n_tasks'] < 1:
                raise ValueError(f'No evaluated tasks: {path}')
            summaries.append(summary)
        row = {'condition': condition,
               'n_tasks': sum(s['n_tasks'] for s in summaries)}
        for metric in METRICS:
            values = [s['overall'][metric] for s in summaries]
            row[metric] = mean(values)
        rows.append(row)
    out = ROOT / 'pilot_summary.json'
    out.write_text(json.dumps({'aggregation': 'equal-cohort mean', 'rows': rows},
                              indent=2) + '\n')
    lines = [
        '# One-sample-per-cohort visual gate pilot', '',
        'All six conditions use the same single test sample within each of 11 cohorts.',
        'The table reports an equal mean of the 11 cohort scores. This pilot',
        'is a functionality and throughput check; it is not the full test result.',
        '',
        '| Condition | Steps | Macro F1 | Balanced accuracy | Hull IoU |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f'| {row["condition"]} | {row["n_tasks"]} | '
            f'{row["f1_macro_mean"]:.3f} | '
            f'{row["balanced_accuracy_mean"]:.3f} | '
            f'{row["iou_hull_macro_mean"]:.3f} |'
        )
    markdown = ROOT / 'pilot_summary.md'
    markdown.write_text('\n'.join(lines) + '\n')
    print(markdown.read_text())


if __name__ == '__main__':
    main()
