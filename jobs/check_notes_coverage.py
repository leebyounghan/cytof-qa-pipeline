"""Every (cohort, step) in data/*/gating_plan.json must resolve in notes.yaml."""
import json
import sys
from pathlib import Path

from src.notes import NOTE_MODES, get_note

missing = []
n = 0
for plan in sorted(Path('data').glob('*/gating_plan.json')):
    cohort = plan.parent.name
    for s in json.loads(plan.read_text())['steps']:
        for mode in NOTE_MODES:
            try:
                get_note(cohort, s['step'], mode)
            except (KeyError, ValueError) as e:
                missing.append(f'{mode}: {e}')
        n += 1
print(f'checked {n} (cohort, step) pairs x {len(NOTE_MODES)} modes')
if missing:
    print('\n'.join(missing))
    sys.exit(f'{len(missing)} lookups failed')
print('all resolved')
