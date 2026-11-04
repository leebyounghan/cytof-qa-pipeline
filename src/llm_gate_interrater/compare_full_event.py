"""Interrater named-category F1 on all input events, including parent divergence."""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .run_sample import DATA, GATERS, META, OUT, PLAN


def main() -> None:
    samples = [item['sample'] for item in json.loads(
        (DATA / 'dataset_manifest.json').read_text())['files']]
    pairs = list(combinations(GATERS, 2))
    counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for i, sample in enumerate(samples, 1):
        columns = [c for m in META['gater_columns'].values() for c in m.values()]
        frame = pd.read_parquet(DATA / 'parquet' / f'{sample}.parquet', columns=columns)
        for step in PLAN['steps']:
            num = str(step['step'])
            canonical = step['annotation column name']
            for cat in step['annotation categories']:
                bits = {g: frame[META['gater_columns'][g][canonical]].to_numpy() == cat
                        for g in GATERS}
                for a, b in pairs:
                    x, y = bits[a], bits[b]
                    c = counts[(a, b, num, cat)]
                    c['tp'] += int(np.count_nonzero(x & y))
                    c['fp'] += int(np.count_nonzero(~x & y))
                    c['fn'] += int(np.count_nonzero(x & ~y))
        if i % 25 == 0 or i == len(samples):
            print(f'compared {i}/{len(samples)} samples', flush=True)

    def f1(c):
        d = 2*c['tp'] + c['fp'] + c['fn']
        return 2*c['tp']/d if d else None

    result = {'n_samples': len(samples), 'protocol': 'all events, named gate memberships; outside-parent and Unassigned do not count as named membership',
              'pairs': {}, 'categories': {}}
    for a, b in pairs:
        cat_scores = [f1(counts[(a, b, str(s['step']), cat)])
                      for s in PLAN['steps'] for cat in s['annotation categories']]
        result['pairs'][f'{a}__{b}'] = float(np.mean([v for v in cat_scores if v is not None]))
    for step in PLAN['steps']:
        for cat in step['annotation categories']:
            key = f"{step['step']}:{cat}"
            result['categories'][key] = {
                f'{a}__{b}': {'f1': f1(counts[(a,b,str(step['step']),cat)]),
                              **counts[(a,b,str(step['step']),cat)]}
                for a, b in pairs}
    (OUT / 'rater_full_event.json').write_text(json.dumps(result, indent=2))
    lines = ['# Interrater agreement across all events', '',
             'Named-category macro F1 across 194 samples. A cell outside a rater’s '
             'parent population has no membership in that step’s named categories. '
             'These scores include disagreement inherited from upstream gates.', '',
             '| Rater pair | Macro F1 |', '|---|---:|']
    for pair, score in result['pairs'].items():
        lines.append(f"| {pair.replace('__', '–')} | {score:.3f} |")
    lines.append('')
    (OUT / 'rater_full_event.md').write_text('\n'.join(lines))
    print(f'Wrote {OUT / "rater_full_event.json"}')


if __name__ == '__main__':
    main()
