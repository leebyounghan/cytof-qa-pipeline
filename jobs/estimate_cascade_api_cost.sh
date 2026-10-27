#!/usr/bin/env bash
#BSUB -J estimate_cascade_api_cost
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 1G
#BSUB -R "rusage[mem=1G]"
#BSUB -W 00:05
#BSUB -o jobs/logs/estimate_cascade_api_cost.%J.out
#BSUB -e jobs/logs/estimate_cascade_api_cost.%J.err
set -euo pipefail
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
.venv/bin/python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

rows = json.loads(Path('jobs/logs/usage_sample.json').read_text())
by_model = defaultdict(list)
for row in rows:
    by_model[row['model']].append(row)

n_steps = 11671  # Current 11-cohort test split, documented in docs/experiment_results.md.
prices = {
    'gpt-5.4': {'input': 2.50, 'cached': .25, 'output': 15.00},
    'gpt-6-luna': {'input': .10, 'cached': .01, 'output': .50},
}
for model, model_rows in by_model.items():
    avg = {k: sum(row[k] for row in model_rows) / len(model_rows)
           for k in ('prompt', 'cached', 'completion', 'reasoning')}
    price = prices[model]
    cached_cost = n_steps / 1_000_000 * (
        (avg['prompt'] - avg['cached']) * price['input']
        + avg['cached'] * price['cached'] + avg['completion'] * price['output'])
    uncached_cost = n_steps / 1_000_000 * (
        avg['prompt'] * price['input'] + avg['completion'] * price['output'])
    print(model, 'n_usage_samples', len(model_rows), 'n_calls', n_steps,
          'avg_tokens', {k: round(v, 1) for k, v in avg.items()},
          'estimated_usd_with_observed_cache', round(cached_cost, 2),
          'estimated_usd_without_cache', round(uncached_cost, 2))
PY
