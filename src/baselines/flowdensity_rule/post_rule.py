"""
post_rule.py — Stage 3 of the flowDensity + notes.yaml rule pipeline.

Maps each cell to ``rule.regions[y_bin][x_bin]`` using the flowDensity
valleys, where bin index = number of valleys strictly below the value
(equality stays in the lower bin). Writes prediction.json per step and
deletes input.csv.

A step fails — and is written with all-None labels so eval scores it as
wrong instead of dropping it — when the rule is unsupported, deGate errored,
or the valley count differs from what the rule requires. Failures are listed
in <results-dir>/post_summary.json.

Usage:
    python -m src.baselines.flowdensity_rule.post_rule --results-dir results/flowdensity_rule/Bjornson
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

METHOD = 'flowdensity_rule'


def bins(values: np.ndarray, valleys: list[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(valleys, dtype=float), values, side='left')


def failure_reason(meta: dict, valleys: dict | None) -> str | None:
    rule = meta['rule']
    if not rule['supported']:
        return 'unsupported_rule'
    for axis in ('x', 'y'):
        if f'{axis}_error' in valleys:
            return f'degate_error_{axis}'
        want = rule[f'n_{axis}_valleys']
        if len(valleys[axis]) != want:
            return f'valley_count_{axis}'
    return None


def postprocess_step(step_dir: Path) -> str:
    meta = json.loads((step_dir / 'meta.json').read_text())
    n = meta['n_parent']
    rule = meta['rule']
    csv_path = step_dir / 'input.csv'
    pred = {'task_id': meta['task_id'], 'method': METHOD}

    if n == 0:
        pred.update(predicted_labels=[], info={'n_parent': 0})
        (step_dir / 'prediction.json').write_text(json.dumps(pred, ensure_ascii=False))
        return 'empty_parent'

    valleys = None
    if rule['supported']:
        valleys = json.loads((step_dir / 'valleys.json').read_text())
    reason = failure_reason(meta, valleys)
    info = {'n_parent': n, 'valleys': valleys}

    if reason is not None:
        pred.update(predicted_labels=[None] * n, info={**info, 'failed': reason})
    else:
        df = pd.read_csv(csv_path)
        if len(df) != n:
            raise ValueError(f'{csv_path}: {len(df)} rows != meta n_parent {n}')
        xb = bins(df['x'].to_numpy(float), valleys['x'])
        yb = bins(df['y'].to_numpy(float), valleys['y'])
        table = np.asarray(rule['regions'], dtype=object)
        pred.update(predicted_labels=table[yb, xb].tolist(), info=info)

    (step_dir / 'prediction.json').write_text(json.dumps(pred, ensure_ascii=False))
    if csv_path.exists():
        csv_path.unlink()
    return reason or 'ok'


def main():
    parser = argparse.ArgumentParser(description='flowDensity rule pipeline — Stage 3: assign regions')
    parser.add_argument('--results-dir', type=Path, required=True)
    args = parser.parse_args()

    step_dirs = sorted(p.parent for p in args.results_dir.rglob('step_*/meta.json'))
    statuses = Counter()
    failed = []
    for d in step_dirs:
        s = postprocess_step(d)
        statuses[s] += 1
        if s not in ('ok', 'empty_parent'):
            failed.append({'step_dir': str(d.relative_to(args.results_dir)), 'reason': s})
    summary = {'n_steps': len(step_dirs), 'status_counts': dict(statuses), 'failed': failed}
    (args.results_dir / 'post_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'post_rule: {len(step_dirs)} steps → {dict(statuses)}')


if __name__ == '__main__':
    main()
