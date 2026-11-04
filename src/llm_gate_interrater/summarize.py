"""Aggregate completed per-sample interrater gate metrics."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .run_sample import DATA, GATERS, OUT, PLAN


def main() -> None:
    samples = [item['sample'] for item in json.loads(
        (DATA / 'dataset_manifest.json').read_text())['files']]
    counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    complete = []
    missing = []
    for sample in samples:
        for gater in GATERS:
            path = OUT / sample / gater / 'metrics.json'
            labels = OUT / sample / gater / 'predicted_labels.parquet'
            if not path.exists() or not labels.exists():
                missing.append(f'{sample}/{gater}')
                continue
            report = json.loads(path.read_text())
            if {int(s) for s in report['steps']} != {s['step'] for s in PLAN['steps']}:
                raise ValueError(f'{path}: incomplete step set')
            complete.append(f'{sample}/{gater}')
            for step, data in report['steps'].items():
                for category, metric in data['category_metrics'].items():
                    dst = counts[(gater, step, category)]
                    for key in ('tp', 'fp', 'fn'):
                        dst[key] += metric[key]
    result = {'expected_sample_gaters': len(samples) * len(GATERS),
              'completed': len(complete), 'missing': missing, 'gaters': {}}
    for gater in GATERS:
        steps = {}
        for step in PLAN['steps']:
            number = str(step['step'])
            categories = {}
            for category in [*step['annotation categories'], 'Unassigned']:
                c = counts[(gater, number, category)]
                denom = 2 * c['tp'] + c['fp'] + c['fn']
                categories[category] = {**c, 'f1': 2 * c['tp'] / denom if denom else None}
            steps[number] = categories
        result['gaters'][gater] = steps
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2))
    print(f"completed {len(complete)}/{result['expected_sample_gaters']} sample-gaters")
    if missing:
        raise SystemExit(f'{len(missing)} sample-gaters missing; see summary.json')


if __name__ == '__main__':
    main()
