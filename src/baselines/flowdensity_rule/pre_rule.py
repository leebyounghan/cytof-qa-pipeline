"""
pre_rule.py — Stage 1 of the flowDensity + notes.yaml rule pipeline (no LLM).

For every task, looks up ``rule`` in benchmark/notes.yaml, checks it against
task.json, and writes the step inputs for the R valley stage:

  input.csv  — x, y (parent-population cells; no GT labels)
  meta.json  — task_id, categories, rule, required valley counts

Unsupported rules (``regions: unsupported``) get meta.json only; the post
stage scores them as failed steps.

Usage:
    python -m src.baselines.flowdensity_rule.pre_rule \\
        --output-dir results/flowdensity_rule/ \\
        --split benchmark/splits.json --split-set test --datasets Bjornson
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.bench import BenchmarkLoader, TaskInput
from src.notes import NOTES_PATH, load_notes


def rule_for(ti: TaskInput) -> dict:
    """Return the validated notes.yaml rule for this task."""
    where = f'{NOTES_PATH}: ({ti.dataset}, step {ti.task["step"]})'
    entry = (load_notes().get(ti.dataset) or {}).get(int(ti.task['step']))
    if not isinstance(entry, dict) or not isinstance(entry.get('rule'), dict):
        raise KeyError(f'{where}: missing rule mapping')
    rule = entry['rule']
    for axis in ('x_marker', 'y_marker'):
        if rule.get(axis) != ti.task[axis]:
            raise ValueError(f'{where}: rule.{axis}={rule.get(axis)!r} != '
                             f'task.json {axis}={ti.task[axis]!r}')
    regions = rule.get('regions')
    if regions == 'unsupported':
        return {**rule, 'supported': False}
    if (not isinstance(regions, list) or not regions
            or not all(isinstance(r, list) and r for r in regions)
            or len({len(r) for r in regions}) != 1):
        raise ValueError(f'{where}: regions must be a non-empty rectangular '
                         f'table or "unsupported", got {regions!r}')
    unknown = {c for row in regions for c in row} - set(ti.categories)
    if unknown:
        raise ValueError(f'{where}: region labels {sorted(unknown)} not in '
                         f'task categories {ti.categories}')
    return {**rule, 'supported': True,
            'n_x_valleys': len(regions[0]) - 1,
            'n_y_valleys': len(regions) - 1}


def write_step(ti: TaskInput, out_dir: Path) -> bool:
    rule = rule_for(ti)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        'task_id': ti.task_id,
        'dataset': ti.dataset,
        'step': int(ti.task['step']),
        'categories': ti.categories,
        'n_parent': int(len(ti.y)),
        'rule': rule,
    }
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if rule['supported'] and len(ti.y) > 0:
        X = ti.X_2d
        pd.DataFrame({'x': X[:, 0], 'y': X[:, 1]}).to_csv(out_dir / 'input.csv', index=False)
    return rule['supported']


def main():
    parser = argparse.ArgumentParser(description='flowDensity rule pipeline — Stage 1: prepare inputs')
    parser.add_argument('--output-dir', type=Path, default=Path('results/flowdensity_rule/'))
    BenchmarkLoader.add_cli_args(parser)
    args = parser.parse_args()

    loader = BenchmarkLoader.from_cli(args)
    n_sup = n_unsup = 0
    for ti in loader.iter_tasks_from_cli(args):
        out_dir = args.output_dir / ti.dataset / ti.sample / ti.step_dir_name
        if write_step(ti, out_dir):
            n_sup += 1
        else:
            n_unsup += 1
    print(f'pre_rule: {n_sup} supported steps written, {n_unsup} unsupported (meta only)')


if __name__ == '__main__':
    main()
