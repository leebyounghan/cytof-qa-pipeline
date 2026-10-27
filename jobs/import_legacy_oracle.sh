#!/usr/bin/env bash
#BSUB -J import_legacy_oracle
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 2000
#BSUB -R "rusage[mem=2000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/import_legacy_oracle.%J.log

set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

.venv/bin/python - <<'PY'
from pathlib import Path
import os
import re

import yaml

path = Path('benchmark/notes.yaml')
legacy_path = Path('../cytof-qa-pipeline_old_20260514/benchmark/notes.yaml')
original = path.read_text()
current = yaml.safe_load(original)
legacy = yaml.safe_load(legacy_path.read_text())

for cohort, steps in current.items():
    if cohort not in legacy:
        raise KeyError(f'Legacy cohort missing: {cohort}')
    for step in steps:
        if step not in legacy[cohort] or not isinstance(legacy[cohort][step], str):
            raise KeyError(f'Legacy note missing: {cohort} step {step}')

output = []
lines = original.splitlines(keepends=True)
cohort = None
step = None
replaced = set()
i = 0
while i < len(lines):
    line = lines[i]
    cohort_match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_-]*):\n', line)
    step_match = re.fullmatch(r'  ([0-9]+):\n', line)
    if cohort_match:
        cohort = cohort_match.group(1)
        step = None
    elif step_match:
        step = int(step_match.group(1))
    if line == '    details: |\n':
        if cohort is None or step is None:
            raise ValueError(f'Details outside a cohort and step at line {i + 1}')
        key = (cohort, step)
        if key in replaced:
            raise ValueError(f'Duplicate details: {key}')
        replaced.add(key)
        note = legacy[cohort][step]
        output.append('    oracle: |\n')
        output.extend(f'      {part}\n' for part in note.rstrip('\n').split('\n'))
        i += 1
        while i < len(lines) and (lines[i].startswith('      ') or not lines[i].strip()):
            i += 1
        continue
    output.append(line)
    i += 1

expected = {(cohort, step) for cohort, steps in current.items() for step in steps}
if replaced != expected:
    raise ValueError(f'Replaced {len(replaced)} of {len(expected)} entries; missing: {expected - replaced}')
updated_text = ''.join(output)
updated = yaml.safe_load(updated_text)
for cohort, step in sorted(expected):
    before = current[cohort][step]
    after = updated[cohort][step]
    if after['definition'] != before['definition'] or after['rule'] != before['rule']:
        raise ValueError(f'Definition or rule changed: {cohort} step {step}')
    if set(after) != {'definition', 'rule', 'oracle'}:
        raise ValueError(f'Unexpected fields: {cohort} step {step}: {set(after)}')
    if after['oracle'] != legacy[cohort][step]:
        raise ValueError(f'Legacy oracle mismatch: {cohort} step {step}')

temporary = path.with_name('notes.yaml.oracle-migration.tmp')
temporary.write_text(updated_text)
os.replace(temporary, path)
print(f'Imported {len(expected)} legacy oracle notes; definition and rule verified unchanged.')
print('Legacy-only cohorts:', sorted(set(legacy) - set(current)))
PY
