"""
post_rule.py — Stage 3 of the flowDensity + notes.yaml rule pipeline.

Maps each cell to ``rule.regions[y_bin][x_bin]`` using the flowDensity
valleys, where bin index = number of valleys strictly below the value
(equality stays in the lower bin). Writes prediction.json per step and
deletes input.csv.

An axis whose R stage reports ``middle_dropped`` (rule needs 2 valleys,
only 1 found even after the split rescue) keeps its single valley and maps
the two sides to the outer region labels; the step's status is
``ok_middle_dropped_{axis}``.

Unsupported rules (``regions: unsupported``) predict every parent cell as
the step's single target category (status ``unsupported_all_target``); a
step with more than one target category is an error.

A step fails — and is written with all-None labels so eval scores it as
wrong instead of dropping it — when deGate errored or the valley count
differs from what the rule requires. Statuses and failures are listed in
<results-dir>/post_summary.json.

``--reuse-existing`` re-writes only steps that still have input.csv (or no
prediction.json) and reads every other status back from prediction.json.

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


def single_target(meta: dict) -> str:
    targets = [c for c in meta['categories'] if c != 'Unassigned']
    if len(targets) != 1:
        raise ValueError(f"{meta['task_id']}: unsupported rule with {len(targets)} "
                         f"target categories {targets}; all-target prediction needs exactly one")
    return targets[0]


def middle_dropped(valleys: dict, axis: str) -> bool:
    return valleys.get(f'{axis}_method') == 'middle_dropped'


def failure_reason(meta: dict, valleys: dict) -> str | None:
    rule = meta['rule']
    for axis in ('x', 'y'):
        if f'{axis}_error' in valleys:
            return f'degate_error_{axis}'
        want = rule[f'n_{axis}_valleys']
        if middle_dropped(valleys, axis):
            want -= 1
        if len(valleys[axis]) != want:
            return f'valley_count_{axis}'
    return None


def axis_bins(values: np.ndarray, valleys: dict, axis: str, n_bins: int) -> np.ndarray:
    b = bins(values, valleys[axis])
    if middle_dropped(valleys, axis):
        if n_bins != 3:
            raise ValueError(f'middle_dropped on {axis} needs a 3-bin rule, got {n_bins}')
        b = np.where(b == 0, 0, n_bins - 1)
    return b


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

    if not rule['supported']:
        target = single_target(meta)
        pred.update(predicted_labels=[target] * n,
                    info={'n_parent': n, 'unsupported_all_target': target,
                          'status': 'unsupported_all_target'})
        (step_dir / 'prediction.json').write_text(json.dumps(pred, ensure_ascii=False))
        return 'unsupported_all_target'

    valleys = json.loads((step_dir / 'valleys.json').read_text())
    reason = failure_reason(meta, valleys)
    info = {'n_parent': n, 'valleys': valleys}

    if reason is not None:
        status = reason
        pred.update(predicted_labels=[None] * n, info={**info, 'failed': reason, 'status': status})
    else:
        df = pd.read_csv(csv_path)
        if len(df) != n:
            raise ValueError(f'{csv_path}: {len(df)} rows != meta n_parent {n}')
        table = np.asarray(rule['regions'], dtype=object)
        xb = axis_bins(df['x'].to_numpy(float), valleys, 'x', table.shape[1])
        yb = axis_bins(df['y'].to_numpy(float), valleys, 'y', table.shape[0])
        dropped = [a for a in ('x', 'y') if middle_dropped(valleys, a)]
        status = 'ok' + ''.join(f'_middle_dropped_{a}' for a in dropped)
        pred.update(predicted_labels=table[yb, xb].tolist(), info={**info, 'status': status})

    (step_dir / 'prediction.json').write_text(json.dumps(pred, ensure_ascii=False))
    if csv_path.exists():
        csv_path.unlink()
    return status


def existing_status(step_dir: Path) -> str:
    info = json.loads((step_dir / 'prediction.json').read_text())['info']
    if info['n_parent'] == 0:
        return 'empty_parent'
    if 'status' in info:
        return info['status']
    if 'unsupported_all_target' in info:
        return 'unsupported_all_target'
    return info.get('failed', 'ok')


def main():
    parser = argparse.ArgumentParser(description='flowDensity rule pipeline — Stage 3: assign regions')
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--reuse-existing', action='store_true',
                        help='Re-write only steps with input.csv (or no prediction.json); '
                             'read other statuses from their existing prediction.json.')
    args = parser.parse_args()

    step_dirs = sorted(p.parent for p in args.results_dir.rglob('step_*/meta.json'))
    statuses = Counter()
    failed = []
    for d in step_dirs:
        reuse = (args.reuse_existing and not (d / 'input.csv').exists()
                 and (d / 'prediction.json').exists())
        s = existing_status(d) if reuse else postprocess_step(d)
        statuses[s] += 1
        if not (s.startswith('ok') or s in ('empty_parent', 'unsupported_all_target')):
            failed.append({'step_dir': str(d.relative_to(args.results_dir)), 'reason': s})
    summary = {'n_steps': len(step_dirs), 'status_counts': dict(statuses), 'failed': failed}
    (args.results_dir / 'post_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'post_rule: {len(step_dirs)} steps → {dict(statuses)}')


if __name__ == '__main__':
    main()
