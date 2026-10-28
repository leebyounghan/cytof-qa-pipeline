"""Match the existing Luna-medium LLM-Gate predictions to visual pilot tasks."""

import json
from pathlib import Path
from statistics import mean

from summarize_visual_pilot import CONDITIONS, DATASETS, METRICS


PILOT = Path('results/visual_luna_medium_definition/pilot_concurrency40')
LLM = Path('results/gate_gpt-6-luna-medium_full')
OUT = PILOT / 'llm_gate_matched'


def slice_predictions() -> None:
    if (LLM / 'notes.yaml.used').read_bytes() != (
        Path('results/visual_luna_medium_definition/notes.yaml.used').read_bytes()
    ):
        raise ValueError('LLM and visual definition snapshots differ')
    matched = []
    for dataset in DATASETS:
        visual = json.loads((PILOT / 'vlm_scatter' / dataset / 'pred.json').read_text())
        (sample, sample_record), = visual['samples'].items()
        expected_steps = set(sample_record['steps'])
        source = LLM / f'{dataset}_gpt-6-luna-medium_full__test.json'
        llm = json.loads(source.read_text())
        if llm['meta']['model'] != 'gpt-6-luna':
            raise ValueError(f'Model mismatch: {source}')
        if sample not in llm['samples']:
            raise ValueError(f'Missing LLM sample: {dataset}/{sample}')
        llm_steps = llm['samples'][sample]['steps']
        missing = expected_steps - set(llm_steps)
        if missing:
            raise ValueError(f'Missing LLM steps: {dataset}/{sample}: {sorted(missing)}')
        for step in expected_steps:
            visual_options = sample_record['steps'][step]['options']
            llm_options = llm_steps[step]['options']
            if visual_options != llm_options:
                raise ValueError(f'Option mismatch: {dataset}/{sample}/{step}')
        selected = {step: llm_steps[step] for step in sorted(expected_steps, key=int)}
        out = OUT / dataset / 'pred.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'meta': {**llm['meta'], 'n_samples': 1,
                                             'matched_visual_pilot': True},
                                   'samples': {sample: {'steps': selected}}},
                                  ensure_ascii=False, indent=2) + '\n')
        matched.append({'dataset': dataset, 'sample': sample,
                        'n_steps': len(selected)})
    (OUT / 'matched_samples.json').write_text(
        json.dumps(matched, ensure_ascii=False, indent=2) + '\n')
    print(f'Matched {len(matched)} cohorts, {sum(x["n_steps"] for x in matched)} steps',
          flush=True)


def summarize() -> None:
    rows = []
    for condition in ('llm_gate', *CONDITIONS):
        root = OUT if condition == 'llm_gate' else PILOT / condition
        summaries = []
        for dataset in DATASETS:
            path = root / 'predictions' / dataset / 'eval_summary.json'
            summary = json.loads(path.read_text())
            if summary['n_tasks'] < 1:
                raise ValueError(f'No evaluated tasks: {path}')
            summaries.append(summary)
        row = {'condition': condition,
               'n_tasks': sum(s['n_tasks'] for s in summaries)}
        for metric in METRICS:
            row[metric] = mean(s['overall'][metric] for s in summaries)
        rows.append(row)
    if len({row['n_tasks'] for row in rows}) != 1:
        raise ValueError(f'Evaluated step counts differ: {rows}')
    baseline = rows[0]
    for row in rows[1:]:
        for metric in METRICS:
            row[f'delta_{metric}'] = row[metric] - baseline[metric]
    (OUT / 'comparison.json').write_text(
        json.dumps({'aggregation': 'equal-cohort mean; matched pilot samples',
                    'rows': rows}, indent=2) + '\n')
    lines = ['# Matched one-sample comparison with LLM-Gate', '',
             'Same 11 cohort samples and 260 steps; all methods use the same',
             'definition snapshot and GPT-6 Luna medium. Each cohort has equal weight.',
             'The LLM-Gate gates come from the saved full test run and were',
             'propagated and evaluated again on these pilot samples.', '',
             '| Condition | Steps | Macro F1 | Δ F1 vs LLM | Balanced accuracy | Hull IoU |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        delta = row.get('delta_f1_macro_mean', 0.0)
        lines.append(f'| {row["condition"]} | {row["n_tasks"]} | '
                     f'{row["f1_macro_mean"]:.3f} | {delta:+.3f} | '
                     f'{row["balanced_accuracy_mean"]:.3f} | '
                     f'{row["iou_hull_macro_mean"]:.3f} |')
    lines += ['', 'This is a pilot subset, not the full test comparison.']
    path = OUT / 'comparison.md'
    path.write_text('\n'.join(lines) + '\n')
    print(path.read_text(), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('slice', 'summarize'))
    args = parser.parse_args()
    if args.action == 'slice':
        slice_predictions()
    else:
        summarize()
