"""Refresh prompts for changed definitions and make a step-only benchmark view."""
import json
from pathlib import Path

from src.llm.c2s_prompt import process_json as grid_prompt
from src.llm_flowsom.c2s_prompt import process_json as flowsom_prompt
from src.llm_gate.prompt import process_json as gate_prompt

ROOT = Path('/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline')
AUDIT = ROOT / 'jobs/definition_rerun_audit_20261027.json'
OUT = ROOT / 'results/_definition_rerun_20261027'
BENCH = OUT / 'benchmark'
reports = json.loads(AUDIT.read_text())
changed_sets = {tuple((cohort, step) for cohort, step, _ in r['changed']) for r in reports.values()}
if len(changed_sets) != 1:
    raise ValueError('Runs disagree on the changed step set')
changed = sorted(changed_sets.pop())
splits = json.loads((ROOT / 'benchmark/splits.json').read_text())['in_panel']
manifest = {}
for cohort, step in changed:
    samples = []
    for sample in splits[cohort]['test']:
        task_dir = ROOT / 'benchmark' / cohort / sample / f'step_{step:02d}'
        if not (task_dir / 'task.json').is_file():
            continue
        for filename in ('gate_task.json', 'c2s_grid.json', 'c2s.json'):
            if not (task_dir / filename).is_file():
                raise FileNotFoundError(task_dir / filename)
        gate_prompt(str(task_dir / 'gate_task.json'), save_md=True, note_mode='definition')
        grid_prompt(str(task_dir / 'c2s_grid.json'), save_md=True, top_hvp_n=10,
                    prompt_subdir='c2s_den_hvp10', note_mode='definition')
        flowsom_prompt(str(task_dir / 'c2s.json'), save_md=True, top_hvp_n=10,
                       prompt_subdir='c2s_clu_hvp10', note_mode='definition')
        for dirname in ('gate_full', 'c2s_den_hvp10', 'c2s_clu_hvp10'):
            if not (task_dir / dirname / 'cells.md').is_file():
                raise FileNotFoundError(task_dir / dirname / 'cells.md')
        link = BENCH / cohort / sample / f'step_{step:02d}'
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != task_dir.resolve():
                raise ValueError(f'Incorrect existing link: {link}')
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(task_dir, target_is_directory=True)
        samples.append(sample)
    expected = next(r for r in reports.values())['tasks_by_cohort'][cohort]
    # Multiple changed steps in one cohort share the same test samples.
    if len(samples) != expected // sum(c == cohort for c, _ in changed):
        raise ValueError(f'Unexpected sample count for {cohort} step {step}: {len(samples)}')
    manifest.setdefault(cohort, {})[str(step)] = samples
    print(f'{cohort} step {step}: {len(samples)} prompts refreshed')
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Prepared', len(changed), 'step definitions in', BENCH)
