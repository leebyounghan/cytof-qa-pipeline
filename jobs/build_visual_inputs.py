"""Build the six conditions' shared, definition-only inputs on test samples."""

import argparse
import json
from pathlib import Path

from src.bench import BenchmarkLoader
from src.visual_variants import extent_for, render_background
from src.vlm_gate.prompt import format_prompt
from src.vlm_gate.utils.task_input_adapter import _TaskInputDataset, _step_from_task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--root', default='results/visual_luna_medium_definition/inputs')
    args = ap.parse_args()
    root = Path(args.root)
    loader = BenchmarkLoader('benchmark', 'data')
    count = 0
    for ti in loader.iter_tasks(
        datasets=[args.dataset], split='benchmark/splits.json', split_set='test',
    ):
        step_dir = root / ti.dataset / ti.sample / f'step_{ti.step:02d}'
        source = ti._step_dir / 'gate_task.json'
        if not source.is_file():
            raise FileNotFoundError(source)
        task = json.loads(source.read_text())
        prompt = format_prompt(task, note_mode='definition')
        adapter = _TaskInputDataset(ti)
        x, y = adapter.get_xy_vals(_step_from_task(ti))
        extent = extent_for(x, y)
        expected = {'step': ti.step, 'options': task['options'],
                    'x_marker': task['x_marker'], 'y_marker': task['y_marker'],
                    'scatter_extent': extent, 'n_parent_cells': len(x),
                    'note_mode': 'definition'}
        step_dir.mkdir(parents=True, exist_ok=True)
        meta = step_dir / 'meta.json'
        prompt_path = step_dir / 'cells.md'
        if meta.exists() and json.loads(meta.read_text()) != expected:
            raise ValueError(f'Existing metadata differs: {meta}')
        if prompt_path.exists() and prompt_path.read_text() != prompt:
            raise ValueError(f'Existing prompt differs: {prompt_path}')
        meta.write_text(json.dumps(expected, indent=2) + '\n')
        prompt_path.write_text(prompt)
        for style in ('scatter', 'contour', 'density'):
            path = step_dir / f'{style}.png'
            if not path.exists():
                render_background(x, y, path, extent, style)
        count += 1
        if count % 100 == 0:
            print(f'{args.dataset}: built {count} steps', flush=True)
    if count == 0:
        raise RuntimeError(f'No test tasks found for {args.dataset}')
    print(f'{args.dataset}: DONE {count} steps', flush=True)


if __name__ == '__main__':
    main()
