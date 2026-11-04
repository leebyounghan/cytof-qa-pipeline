"""Prepare, infer, and evaluate one interrater sample in an LSF job."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.llm_gate.task import generate_gate_task
from src.llm_gate.prompt import format_prompt
from src.llm_gate.postprocess.propagate import _propagate_step


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data_interrater'
OUT = ROOT / 'results' / 'interrater_gate_gpt-6-luna-medium'
PLAN = json.loads((DATA / 'gating_plan.json').read_text())
META = json.loads((DATA / 'parquet' / 'meta.json').read_text())
GATERS = tuple(META['gaters'])


def parent_mask(frame: pd.DataFrame, step: dict) -> np.ndarray:
    parent = step['parent']
    if parent == 'ALL':
        return np.ones(len(frame), dtype=bool)
    column, category = (part.strip() for part in parent.split(' == ', 1))
    return frame[column].to_numpy() == category


class Dataset:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_parent_mask(self, step):
        return parent_mask(self.frame, step)

    def get_xy_vals(self, step):
        return (self.frame[step['x_marker']].to_numpy(),
                self.frame[step['y_marker']].to_numpy())

    def get_gt_labels(self, step):
        return self.frame[step['annotation column name']].to_numpy()

    def get_display_name(self, channel):
        return META['channels'][channel][0]


def metrics(truth: np.ndarray, predicted: np.ndarray, categories: list[str]) -> dict:
    result = {}
    for cat in categories:
        gt = truth == cat
        pred = predicted == cat
        tp = int(np.count_nonzero(gt & pred))
        fp = int(np.count_nonzero(~gt & pred))
        fn = int(np.count_nonzero(gt & ~pred))
        result[cat] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
    return result


def run(sample: str, prepare_only: bool = False) -> None:
    if Path(sample).name != sample or sample not in {
        item['sample'] for item in json.loads((DATA / 'dataset_manifest.json').read_text())['files']
    }:
        raise ValueError(f'Unknown sample: {sample!r}')
    out = OUT / sample
    out.mkdir(parents=True, exist_ok=True)
    channels = list(META['channels'])
    labels = [c for mapping in META['gater_columns'].values() for c in mapping.values()]
    frame = pd.read_parquet(DATA / 'parquet' / f'{sample}.parquet',
                            columns=['event_index', *channels, *labels])
    if not np.array_equal(frame['event_index'].to_numpy(), np.arange(len(frame))):
        raise ValueError(f'{sample}: event_index is not original row order')
    python = ROOT / '.venv' / 'bin' / 'python'
    for gater in GATERS:
        gdir = out / gater
        gdir.mkdir(exist_ok=True)
        if not prepare_only and (gdir / 'metrics.json').exists():
            print(f'skipping completed {sample}/{gater}', flush=True)
            continue
        canonical = frame[channels].copy(deep=False)
        canonical = canonical.assign(**{
            col: frame[actual] for col, actual in META['gater_columns'][gater].items()
        })
        dataset = Dataset(canonical)
        sample_dir = gdir / 'benchmark' / sample
        expected_steps = []
        for step in PLAN['steps']:
            result = generate_gate_task(dataset, step, modality='cytof',
                                        cofactor=5.0, cohort=PLAN['dataset'])
            if result is None:
                raise ValueError(f'{sample}/{gater}/step{step["step"]}: empty GT parent')
            step_dir = sample_dir / f'step_{step["step"]:02d}'
            prompt_dir = step_dir / 'gate_full'
            prompt_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / 'gate_task.json').write_text(json.dumps(result, indent=2))
            prompt = format_prompt(result, note_path=DATA / 'interrater_notes.yaml')
            (prompt_dir / 'cells.md').write_text(prompt)
            expected_steps.append(step['step'])
        if prepare_only:
            print(f'prepared {sample}/{gater}', flush=True)
            continue
        pred_path = gdir / 'pred.json'
        api_ok = gdir / 'api_success'
        if not api_ok.exists():
            subprocess.run([
                str(python), '-m', 'src.llm_gate.inference.run_openai',
                '--dataset_path', str(sample_dir.parent), '--sample', sample,
                '--model', 'gpt-6-luna', '--ablation_slug', 'full',
                '--output_path', str(pred_path), '--concurrency', '5',
                '--max_tokens', '32768', '--temperature', '1.0',
                '--api-reasoning-effort', 'medium', '--save-raw',
            ], cwd=ROOT, check=True)
            api_ok.write_text('run_openai exited successfully\n')
        preds = json.loads(pred_path.read_text())
        steps = preds['samples'][sample]['steps']
        if {int(s) for s in steps} != set(expected_steps):
            raise ValueError(f'{sample}/{gater}: missing model steps')
        label_frame = pd.DataFrame({'event_index': frame['event_index']})
        report = {'sample': sample, 'gater': gater, 'model': 'gpt-6-luna',
                  'reasoning_effort': 'medium', 'steps': {}}
        for step in PLAN['steps']:
            number = step['step']
            record = steps[str(number)]
            options = [*step['annotation categories'], 'Unassigned']
            if record['options'] != options:
                raise ValueError(f'{sample}/{gater}/step{number}: options mismatch')
            mask = parent_mask(canonical, step)
            xy = canonical.loc[mask, [step['x_marker'], step['y_marker']]].to_numpy(dtype=np.float32)
            predicted, info = _propagate_step(SimpleNamespace(X_2d=xy, categories=options),
                                              record, 'smallest')
            if info['n_gates_valid'] + info['n_gates_absent'] != info['n_gates_total']:
                raise ValueError(f'{sample}/{gater}/step{number}: incomplete gates: {info}')
            truth = canonical.loc[mask, step['annotation column name']].astype(object)
            truth_arr = truth.fillna('Unassigned').replace('', 'Unassigned').to_numpy(dtype=str)
            pred_arr = np.asarray(predicted, dtype=str)
            if len(pred_arr) != len(truth_arr):
                raise ValueError(f'{sample}/{gater}/step{number}: prediction length mismatch')
            full = pd.Series(pd.Categorical([None] * len(frame), categories=options))
            full.loc[mask] = pd.Categorical(pred_arr, categories=options)
            label_frame[step['annotation column name']] = full
            report['steps'][str(number)] = {
                'n_parent': int(mask.sum()), 'gate_info': info,
                'category_metrics': metrics(truth_arr, pred_arr, options),
            }
        label_frame.to_parquet(gdir / 'predicted_labels.parquet', index=False)
        (gdir / 'metrics.json').write_text(json.dumps(report, indent=2))
        print(f'completed {sample}/{gater}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('LSB_JOBID'):
        raise SystemExit('Run through LSF bsub')
    run(args.sample, prepare_only=args.prepare_only)
