"""Summarize four full-test C2S runs with explicit medium reasoning effort."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline')
RESULTS = ROOT / 'results'
SPLITS = json.loads((ROOT / 'benchmark/splits.json').read_text())['in_panel']
COHORTS = ('Acute2020', 'Acute2021', 'Vaccine', 'Bjornson', 'FR-FCM-Z74D_hc',
           'FR-FCM-Z74D_tissue', 'FRDR_covid19', 'Lyoplate_tcell',
           'Lyoplate_treg', 'Lyoplate_bcell', 'Lyoplate_DC')
RUNS = {
    'C2S density-grid · GPT-6 luna medium': 'c2s_flowdensity_gpt-6-luna-medium',
    'C2S density-grid · GPT-6 sol medium': 'c2s_flowdensity_gpt-6-sol-medium',
    'C2S FlowSOM · GPT-6 luna medium': 'c2s_flowsom_gpt-6-luna-medium',
    'C2S FlowSOM · GPT-6 sol medium': 'c2s_flowsom_gpt-6-sol-medium',
}
METRICS = ('f1_macro_mean', 'balanced_accuracy_mean', 'iou_hull_macro_mean')
notes = (ROOT / 'benchmark/notes.yaml').read_bytes()
notes_sha = hashlib.sha256(notes).hexdigest()
records = {}
for label, name in RUNS.items():
    run_dir = RESULTS / name
    rows = {}
    for cohort in COHORTS:
        gt = json.loads((run_dir / 'predictions' / cohort / 'eval_summary.json').read_text())['overall']
        rect = json.loads((run_dir / 'predictions' / cohort / 'eval_summary__gt_rect1_oracle.json').read_text())['overall']
        expected_n = sum(len(list((ROOT / 'benchmark' / cohort / sample).glob('step_*/task.json')))
                         for sample in SPLITS[cohort]['test'])
        if gt['n_tasks'] != expected_n or rect['n_tasks'] != expected_n:
            raise ValueError(f'{name}/{cohort}: expected {expected_n} tasks, got GT {gt["n_tasks"]}, rect {rect["n_tasks"]}')
        rows[cohort] = {'n_tasks': expected_n,
                        'gt': {key: gt[key] for key in METRICS},
                        'rect_gt': {key: rect[key] for key in METRICS}}
    total = sum(row['n_tasks'] for row in rows.values())
    if total != 11671:
        raise ValueError(f'{name}: {total} tasks, expected 11671')
    mean = {kind: {key: sum(rows[c][kind][key] for c in COHORTS) / 11 for key in METRICS}
            for kind in ('gt', 'rect_gt')}
    records[name] = {'label': label, 'effort': 'medium', 'n_tasks': total,
                     'notes_sha256': notes_sha, 'cohorts': rows, 'equal_cohort_mean': mean}
    shutil.copyfile(ROOT / 'benchmark/notes.yaml', run_dir / 'notes.yaml.used')
    (run_dir / 'run_meta.json').write_text(json.dumps({
        'model': 'gpt-6-luna' if 'luna' in name else 'gpt-6-sol',
        'reasoning_effort': 'medium', 'split': 'benchmark/splits.json:test',
        'paradigm': 'density-grid' if 'flowdensity' in name else 'FlowSOM',
        'n_tasks': total, 'notes_sha256': notes_sha}, indent=2) + '\n')

(RESULTS / 'c2s_medium_full_summary.json').write_text(json.dumps(records, indent=2) + '\n')
lines = ['# LLM-C2S GPT-6 medium, full test split', '',
         'All rows use definition-only notes, 11,671 test tasks, and equal-weight averaging over 11 cohorts.',
         '', '| Method | GT F1 | GT BA | GT Hull IoU | rect GT F1 | rect GT BA | rect GT Hull IoU |',
         '|---|---:|---:|---:|---:|---:|---:|']
for name, record in records.items():
    mean = record['equal_cohort_mean']
    vals = [mean[kind][metric] for kind in ('gt', 'rect_gt') for metric in METRICS]
    lines.append('| ' + record['label'] + ' | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
lines += ['', '## Per-cohort GT results', '',
          '| Method | Cohort | Tasks | F1 | BA | Hull IoU |',
          '|---|---|---:|---:|---:|---:|']
for name, record in records.items():
    for cohort in COHORTS:
        row = record['cohorts'][cohort]
        vals = [row['gt'][metric] for metric in METRICS]
        lines.append(f'| {record["label"]} | {cohort} | {row["n_tasks"]} | '
                     + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
(RESULTS / 'c2s_medium_full_summary.md').write_text('\n'.join(lines) + '\n')
print('Wrote results/c2s_medium_full_summary.md and .json')
