"""Check that a C2S output contains every test sample and task for one cohort."""
import json
from pathlib import Path
import sys

ds, filename = sys.argv[1:]
root = Path('/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline')
splits = json.loads((root / 'benchmark/splits.json').read_text())
samples = splits['in_panel'][ds]['test']
output = json.loads(Path(filename).read_text())['samples']
if set(output) != set(samples):
    raise ValueError(f'{ds}: output sample set differs from test split')
count = 0
for sample in samples:
    expected = {int(p.parent.name.split('_')[-1]) for p in (root / 'benchmark' / ds / sample).glob('step_*/task.json')}
    actual = {int(key) for key in output[sample]['steps']}
    if actual != expected:
        raise ValueError(f'{ds}/{sample}: expected {len(expected)} steps, got {len(actual)}')
    count += len(actual)
print(f'Validated {ds}: {len(samples)} samples, {count} steps')
