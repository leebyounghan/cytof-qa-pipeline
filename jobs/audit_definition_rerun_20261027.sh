#!/usr/bin/env bash
#BSUB -J audit_def_rerun
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 2000
#BSUB -R "rusage[mem=2000]"
#BSUB -o /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/audit_def_rerun.%J.out
#BSUB -e /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline/jobs/logs/audit_def_rerun.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import yaml

root = Path('.')
current = yaml.safe_load((root / 'benchmark/notes.yaml').read_text())
splits = json.loads((root / 'benchmark/splits.json').read_text())['in_panel']
reports = {}
for path in sorted((root / 'results').glob('*/notes.yaml.used')):
    old = yaml.safe_load(path.read_text())
    changed = []
    counts = {}
    for cohort, steps in current.items():
        for step, note in steps.items():
            previous = old[cohort][step]
            if note['definition'] != previous['definition']:
                count = sum(
                    (root / 'benchmark' / cohort / sample / f'step_{int(step):02d}' / 'task.json').is_file()
                    for sample in splits[cohort]['test']
                )
                changed.append((cohort, int(step), count))
                counts[cohort] = counts.get(cohort, 0) + count
    reports[path.parent.name] = {'changed': changed, 'tasks_by_cohort': counts, 'total_tasks': sum(counts.values())}
    print(path.parent.name, 'steps=', len(changed), 'test_tasks=', sum(counts.values()))
    for cohort, step, count in changed:
        print(f'  {cohort} step {step}: {count} test tasks')
out = root / 'jobs' / 'definition_rerun_audit_20261027.json'
out.write_text(json.dumps(reports, indent=2) + '\n')
print('Wrote', out)
PY
