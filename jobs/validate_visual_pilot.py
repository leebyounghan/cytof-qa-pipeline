"""Validate one test sample per cohort across all six visual conditions."""

import json
from pathlib import Path


DATASETS = (
    'Acute2020', 'Acute2021', 'Bjornson', 'Vaccine',
    'FR-FCM-Z74D_hc', 'FR-FCM-Z74D_tissue', 'FRDR_covid19',
    'Lyoplate_DC', 'Lyoplate_bcell', 'Lyoplate_tcell', 'Lyoplate_treg',
)
CONDITIONS = tuple(f'{mode}_{style}' for mode in ('vlm', 'agent')
                   for style in ('scatter', 'contour', 'density'))
ROOT = Path('results/visual_luna_medium_definition/pilot_concurrency40')
INPUT_ROOT = Path('results/visual_luna_medium_definition/inputs')


def main() -> None:
    rows = []
    for dataset in DATASETS:
        expected_sample = None
        expected_steps = None
        for condition in CONDITIONS:
            directory = ROOT / condition / dataset
            path = directory / 'pred.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text())
            samples = data['samples']
            if len(samples) != 1:
                raise ValueError(f'{path}: expected one sample, got {len(samples)}')
            sample = next(iter(samples))
            sample_file = (directory / 'sample.txt').read_text().strip()
            if sample_file != sample:
                raise ValueError(f'{path}: sample.txt disagrees')
            steps = set(samples[sample]['steps'])
            input_steps = {str(json.loads(p.read_text())['step']) for p in
                           (INPUT_ROOT / dataset / sample).glob('step_*/meta.json')}
            if steps != input_steps:
                raise ValueError(f'{path}: step set differs from prepared input')
            if expected_sample is None:
                expected_sample, expected_steps = sample, steps
            if sample != expected_sample or steps != expected_steps:
                raise ValueError(f'{path}: conditions use different samples/steps')
            n_missing = sum(
                gate is None
                for step in samples[sample]['steps'].values()
                for gate in step['gates'].values()
            )
            tool_calls = 0
            if condition.startswith('agent_'):
                traces = [step['agent_trace']
                          for step in samples[sample]['steps'].values()]
                if any(t['n_tool_calls'] < 1 for t in traces):
                    raise ValueError(f'{path}: agent step without overlay call')
                tool_calls = sum(t['n_tool_calls'] for t in traces)
            rows.append({'dataset': dataset, 'condition': condition,
                         'sample': sample, 'n_steps': len(steps),
                         'n_missing_gates': n_missing,
                         'tool_calls': tool_calls})
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'validation.json'
    out.write_text(json.dumps({'n_conditions': len(CONDITIONS),
                               'n_datasets': len(DATASETS), 'rows': rows},
                              ensure_ascii=False, indent=2) + '\n')
    print(f'Validated {len(rows)} outputs across {len(DATASETS)} cohorts; '
          f'wrote {out}', flush=True)
    for condition in CONDITIONS:
        group = [r for r in rows if r['condition'] == condition]
        print(f'{condition}: {sum(r["n_steps"] for r in group)} steps, '
              f'{sum(r["n_missing_gates"] for r in group)} missing gates, '
              f'{sum(r["tool_calls"] for r in group)} tool calls', flush=True)


if __name__ == '__main__':
    main()
