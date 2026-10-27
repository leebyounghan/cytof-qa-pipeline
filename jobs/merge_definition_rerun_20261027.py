"""Validate and merge step-only rerun outputs into the existing full runs."""
import copy
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path('/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline')
TEMP = ROOT / 'results/_definition_rerun_20261027'
RUN = sys.argv[1]
manifest = json.loads((TEMP / 'manifest.json').read_text())
datasets = list(manifest)
updates = []
for cohort in datasets:
    partial_path = TEMP / RUN / f'{cohort}.json'
    full_candidates = list((ROOT / 'results' / RUN).glob(f'{cohort}_*.json'))
    if len(full_candidates) != 1:
        raise ValueError(f'Expected one original output for {RUN}/{cohort}: {full_candidates}')
    full_path = full_candidates[0]
    partial = json.loads(partial_path.read_text())
    full = json.loads(full_path.read_text())
    expected = {(sample, step) for step, samples in manifest[cohort].items() for sample in samples}
    observed = {(sample, step) for sample, value in partial['samples'].items()
                for step in value['steps']}
    if observed != expected:
        raise ValueError(f'{RUN}/{cohort}: rerun task set mismatch: missing={expected-observed}, extra={observed-expected}')
    if partial['meta']['model'] != full['meta']['model']:
        raise ValueError(f'{RUN}/{cohort}: model mismatch')
    merged = copy.deepcopy(full)
    for sample, step in expected:
        old = full['samples'][sample]['steps'][step]
        new = partial['samples'][sample]['steps'][step]
        for key in ('x_marker', 'y_marker', 'options'):
            if old[key] != new[key]:
                raise ValueError(f'{RUN}/{cohort}/{sample}/{step}: {key} changed')
        if 'cells' in old:
            old_ids = [(c['cell_id'], c['cell_index']) for c in old['cells']]
            new_ids = [(c['cell_id'], c['cell_index']) for c in new['cells']]
            if old_ids != new_ids:
                raise ValueError(f'{RUN}/{cohort}/{sample}/{step}: sampled cells changed')
        merged['samples'][sample]['steps'][step] = new
    if len(merged['samples']) != len(full['samples']):
        raise ValueError(f'{RUN}/{cohort}: sample count changed')
    updates.append((full_path, merged, len(expected)))

# Complete all validation before changing any existing output.
if len(sys.argv) > 2 and sys.argv[2] == '--validate-only':
    print(f'Validated {sum(n for _, _, n in updates)} updated steps for {RUN}')
    raise SystemExit(0)
for full_path, merged, n in updates:
    backup = TEMP / 'backup' / RUN / full_path.name
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(full_path, backup)
    tmp = full_path.with_suffix(full_path.suffix + '.definition-rerun.tmp')
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, full_path)
    print(f'Merged {n} steps into {full_path}')
