#!/usr/bin/env bash
#BSUB -J "paper_gates_current[1-11]%3"
#BSUB -q i2c2_normal
#BSUB -n 1
#BSUB -M 12G
#BSUB -R "rusage[mem=12G]"
#BSUB -W 04:00
#BSUB -o jobs/logs/paper_gates_current.%J.%I.out
#BSUB -e jobs/logs/paper_gates_current.%J.%I.err
set -euo pipefail
: "${LSB_JOBID:?Submit through LSF}"
: "${LSB_JOBINDEX:?Submit as an LSF job array}"
cd /project/kimlab_storage2NB/jaesik/cytof-qa-pipeline
export PYTHONPATH="$PWD" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DATASETS=(Acute2020 Acute2021 Bjornson Vaccine FR-FCM-Z74D_hc FR-FCM-Z74D_tissue
          FRDR_covid19 Lyoplate_DC Lyoplate_bcell Lyoplate_tcell Lyoplate_treg)
cohort="${DATASETS[$((LSB_JOBINDEX - 1))]}"
out=results/gate_gpt-5.4_paper_gates_current_benchmark
mkdir -p "$out/logs"
.venv/bin/python -m src.llm_gate.postprocess.propagate \
    --eval_path "results/gate_gpt-5.4_full_old/$cohort/pred.json" \
    --dataset "$cohort" --benchmark benchmark --data-dir data \
    --method gate_gpt-5.4_paper_gates_current_benchmark \
    --output_dir "$out/predictions" --tiebreak smallest \
    > "$out/logs/$cohort.propagate.log" 2>&1
.venv/bin/python -m src.eval \
    --predictions "$out/predictions" --benchmark benchmark --data-dir data \
    --datasets "$cohort" --split benchmark/splits.json --split-set test \
    > "$out/logs/$cohort.eval_gt.log" 2>&1
.venv/bin/python -m src.eval \
    --predictions "$out/predictions" --gt-labels results/rect1_oracle \
    --benchmark benchmark --data-dir data --datasets "$cohort" \
    --split benchmark/splits.json --split-set test \
    > "$out/logs/$cohort.eval_rect.log" 2>&1
.venv/bin/python - "$cohort" "$out" <<'PY'
import json
import sys
from pathlib import Path

cohort, out = sys.argv[1], Path(sys.argv[2])
expected = json.loads((Path('results/gate_gpt-5.4_full_old') / cohort / 'eval_summary.json').read_text())['n_tasks']
raw_path = Path('results/gate_gpt-5.4_full_old') / cohort / 'pred.json'
raw = json.loads(raw_path.read_text())
raw_steps = [(sample, int(step)) for sample, record in raw['samples'].items()
             for step in record['steps']]
if len(raw_steps) != expected:
    raise ValueError(f'{cohort}: raw gate count {len(raw_steps)} != old evaluation {expected}')
missing_tasks = [(sample, step) for sample, step in raw_steps
                 if not (Path('benchmark') / cohort / sample / f'step_{step:02d}' / 'task.json').exists()]
expected_current = expected - len(missing_tasks)
base = out / 'predictions' / cohort
gt = json.loads((base / 'eval_summary.json').read_text())
rect = json.loads((base / 'eval_summary__gt_rect1_oracle.json').read_text())
if gt['n_tasks'] != expected_current or rect['n_tasks'] != expected_current:
    raise ValueError(f'{cohort}: expected {expected_current} available tasks, got GT {gt["n_tasks"]}, rect {rect["n_tasks"]}')
print(cohort, expected_current, 'missing task.json', len(missing_tasks),
      'GT', gt['overall']['f1_macro_mean'],
      gt['overall']['balanced_accuracy_mean'], gt['overall']['iou_hull_macro_mean'],
      'rect', rect['overall']['f1_macro_mean'],
      rect['overall']['balanced_accuracy_mean'], rect['overall']['iou_hull_macro_mean'])
PY
