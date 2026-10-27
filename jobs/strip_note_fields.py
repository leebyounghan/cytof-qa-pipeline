"""Remove non-notes.yaml note text: `note`/`tips` from every gating_plan.json
step (plus top-level `notes`) and from every benchmark task.json.

gating_plan.json originals are copied to jobs/backup/gating_plan_<date>/
first. Each file must round-trip byte-identically through json before it is
rewritten; any mismatch aborts before writing anything.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'jobs' / 'backup' / 'gating_plan_20261020'
STEP_KEYS = ('note', 'tips')


def dump(obj, trailing_newline):
    return json.dumps(obj, indent=2, ensure_ascii=False) + ('\n' if trailing_newline else '')


def plan(path, strip):
    text = path.read_text()
    obj = json.loads(text)
    nl = text.endswith('\n')
    if dump(obj, nl) != text:
        raise SystemExit(f'{path}: does not round-trip through json (indent=2); refusing to rewrite')
    removed = strip(obj)
    return obj, nl, removed


def strip_plan(d):
    n = 0
    if 'notes' in d:
        del d['notes']; n += 1
    for s in d['steps']:
        for k in STEP_KEYS:
            if k in s:
                del s[k]; n += 1
    return n


def strip_task(d):
    n = 0
    for k in STEP_KEYS:
        if k in d:
            del d[k]; n += 1
    return n


plans = sorted((ROOT / 'data').glob('*/gating_plan.json'))
tasks = sorted((ROOT / 'benchmark').glob('*/*/step_*/task.json'))
print(f'gating_plan.json: {len(plans)}  task.json: {len(tasks)}', flush=True)

pending = []
for p in plans:
    pending.append((p, *plan(p, strip_plan)))
for p in tasks:
    pending.append((p, *plan(p, strip_task)))
print('round-trip check passed for all files', flush=True)

BACKUP.mkdir(parents=True, exist_ok=False)
for p in plans:
    dst = BACKUP / p.parent.name / p.name
    dst.parent.mkdir()
    shutil.copy2(p, dst)
print(f'backed up {len(plans)} gating_plan.json → {BACKUP}', flush=True)

n_files = n_keys = 0
for p, obj, nl, removed in pending:
    if removed:
        p.write_text(dump(obj, nl))
        n_files += 1; n_keys += removed
print(f'rewrote {n_files} files, removed {n_keys} keys', flush=True)

leftover = [str(p) for p in plans + tasks
            if any(k in json.loads(p.read_text()) for k in ('note', 'tips', 'notes'))
            or ('steps' in json.loads(p.read_text())
                and any(k in s for s in json.loads(p.read_text())['steps'] for k in STEP_KEYS))]
if leftover:
    sys.exit(f'note/tips still present in {len(leftover)} files, e.g. {leftover[:3]}')
print('verified: no note/tips/notes keys remain')
