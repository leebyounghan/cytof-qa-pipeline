"""Walk a benchmark directory and assemble per-(sample, step) prompt
groups for the agent-gate runner.

One group = one rendered prompt (``cells.md`` from a ``gate_{slug}/``
ablation dir) + the path to the pre-rendered background scatter
(``scatter_bg.png``) and its data extent + the metadata needed to write
predictions back into the eval JSON.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_gate_tasks(
    benchmark_dir: str,
    ablation_slug: str = 'full',
    sample_filter: set | None = None,
    perturbation_name: str | None = None,
) -> list[dict]:
    """Walk ``benchmark_dir/{sample}/step_NN/`` and collect one group per
    (sample, step). Each group holds the rendered prompt text plus the
    metadata needed to record the model's per-category gates.

    Returns list of groups::

        {
          'prompt_text':     str,
          'options':         list[str],   # categories incl. 'Unassigned'
          'sample':          str,
          'step':            int,
          'ablation':        str,
          'x_marker':        str,
          'y_marker':        str,
          'scatter_bg_path': str | None,  # absolute path or None
          'scatter_extent':  list[float], # [xmin, xmax, ymin, ymax]
        }

    ``scatter_bg_path`` and ``scatter_extent`` come from gate_task.json's
    ``scatter_bg`` and ``scatter_extent`` fields written by
    ``src.vlm_gate.task``. Older gate_task.json without these falls
    through with ``scatter_bg_path=None`` (overlay tool then renders
    boxes on a blank plane) — re-run ``src.vlm_gate.task`` to upgrade.
    """
    base = Path(benchmark_dir)
    all_samples = sorted(d for d in base.iterdir() if d.is_dir())
    if sample_filter:
        missing = sample_filter - {d.name for d in all_samples}
        if missing:
            print(f'  [WARN] samples not found: {missing}')
        selected = [d for d in all_samples if d.name in sample_filter]
    else:
        selected = all_samples

    task_filename = (f'gate_task__{perturbation_name}.json'
                     if perturbation_name else 'gate_task.json')
    prompt_dir_name = (f'gate_{ablation_slug}__{perturbation_name}'
                       if perturbation_name else f'gate_{ablation_slug}')

    print(f'  Samples: {len(selected)} / {len(all_samples)}  ({benchmark_dir})')
    print(f'  Ablation: {prompt_dir_name}/cells.md')
    if perturbation_name:
        print(f'  Perturbation: {perturbation_name}')

    groups: list[dict] = []
    missing_prompt_dirs = 0
    for sample_dir in selected:
        for step_dir in sorted(sample_dir.iterdir()):
            if not step_dir.is_dir():
                continue
            json_path = step_dir / task_filename
            prompt_dir = step_dir / prompt_dir_name
            md_path = prompt_dir / 'cells.md'
            if not json_path.exists():
                continue
            if not md_path.exists():
                if not prompt_dir.is_dir():
                    missing_prompt_dirs += 1
                continue

            data = json.loads(json_path.read_text())
            options = data.get('options', [])
            prompt_text = md_path.read_text()

            scatter_bg_name = data.get('scatter_bg')
            scatter_bg_path = (str((step_dir / scatter_bg_name).resolve())
                               if scatter_bg_name else None)
            scatter_extent = data.get('scatter_extent') or [0.0, 1.0, 0.0, 1.0]

            groups.append({
                'prompt_text':     prompt_text,
                'options':         options,
                'sample':          sample_dir.name,
                'step':            int(data['step']),
                'ablation':        ablation_slug,
                'x_marker':        data.get('x_marker'),
                'y_marker':        data.get('y_marker'),
                'scatter_bg_path': scatter_bg_path,
                'scatter_extent':  scatter_extent,
            })

    if missing_prompt_dirs:
        print(f'  [WARN] {missing_prompt_dirs} step_dir(s) missing '
              f'{prompt_dir_name}/ — run src.vlm_gate.prompt first.')
    print(f'  Found {len(groups)} (sample, step) prompts')
    return groups
