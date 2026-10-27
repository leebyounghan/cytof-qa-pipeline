#!/usr/bin/env bash
#BSUB -J validate_oracle_notes
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 2000
#BSUB -R "rusage[mem=2000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/validate_oracle_notes.%J.log

set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
from src.notes import DEFAULT_NOTE_MODE, NOTE_MODES, get_note, load_notes

legacy = yaml.safe_load(Path('../cytof-qa-pipeline_old_20260514/benchmark/notes.yaml').read_text())
current = load_notes()
assert DEFAULT_NOTE_MODE == 'definition'
assert NOTE_MODES == ('definition', 'definition_oracle')
count = 0
for cohort, steps in current.items():
    for step, entry in steps.items():
        assert set(entry) == {'definition', 'rule', 'oracle'}, (cohort, step)
        assert entry['oracle'] == legacy[cohort][step], (cohort, step)
        assert get_note(cohort, step) == entry['definition'].strip(), (cohort, step)
        assert get_note(cohort, step, 'definition_oracle') == (
            entry['definition'].strip() + '\n' + entry['oracle'].strip()
        ), (cohort, step)
        count += 1
print(f'Validated both note modes and legacy oracle equality for {count} steps.')
PY
