"""Refresh published comparison tables after the 14-step definition rerun."""
import subprocess
import sys
from pathlib import Path

ROOT = Path('/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline')
DOC = ROOT / 'docs/experiment_results.md'
SUMMARY = ROOT / 'results/vs_gt_rect1_summary.md'
GATE_SUMMARY = ROOT / 'results/gate_full_test_summary.md'

summary_text = subprocess.check_output(
    [sys.executable, str(ROOT / 'results/summarize_vs_gt_rect1.py')], text=True, cwd=ROOT)
SUMMARY.write_text(summary_text)

def fields(line):
    return [part.strip() for part in line.strip().strip('|').split('|')]

summary_rows = {}
cohort_rows = {}
in_cohorts = False
for line in summary_text.splitlines():
    if line == 'F1 vs rect GT per cohort':
        in_cohorts = True
        continue
    if line == 'hull IoU vs GT per cohort':
        in_cohorts = False
    if not line.startswith('| '):
        continue
    parts = fields(line)
    if in_cohorts and parts[0] not in ('cohort', '---') and not parts[0].startswith('---'):
        cohort_rows[parts[0]] = parts[1:]
    elif not in_cohorts and len(parts) == 7 and parts[0] not in ('method', '---'):
        summary_rows[parts[0]] = parts[1:]

names = {
    'C2S grid · GPT-5.4': 'C2S grid gpt-5.4',
    'C2S grid · GPT-6 luna': 'C2S grid luna',
    'C2S FlowSOM · GPT-5.4': 'C2S FlowSOM gpt-5.4',
    'C2S FlowSOM · GPT-6 luna': 'C2S FlowSOM luna',
    'LLM Gate · GPT-5.4 (현재 정의)': 'Gate gpt-5.4',
    'LLM Gate · GPT-6 luna 기본값': 'Gate luna (default)',
    'LLM Gate · GPT-6 luna none': 'Gate luna none',
    'LLM Gate · GPT-6 luna low': 'Gate luna low',
    'LLM Gate · GPT-6 luna medium': 'Gate luna medium',
    'LLM Gate · GPT-6 luna high': 'Gate luna high',
    'LLM Gate · GPT-6 sol medium': 'Gate sol medium',
}
lines = DOC.read_text().splitlines()
section = None
main_seen = set()
cohort_seen = set()
for i, line in enumerate(lines):
    if line.startswith('## '):
        section = line
    if not line.startswith('| '):
        continue
    parts = fields(line)
    if section == '## 전체 결과' and parts[0] in names:
        key = names[parts[0]]
        if key not in summary_rows:
            raise KeyError(key)
        lines[i] = '| ' + ' | '.join([parts[0], *summary_rows[key]]) + ' |'
        main_seen.add(parts[0])
    elif section == '## 코호트별 F1 (rect GT 기준)' and parts[0] in cohort_rows:
        if len(parts) != 14 or len(cohort_rows[parts[0]]) != 11:
            raise ValueError(f'Unexpected cohort row shape: {parts[0]}')
        updated = parts[:2] + cohort_rows[parts[0]][:4] + [parts[6]] + cohort_rows[parts[0]][4:]
        lines[i] = '| ' + ' | '.join(updated) + ' |'
        cohort_seen.add(parts[0])
if main_seen != set(names) or cohort_seen != set(cohort_rows):
    raise ValueError(f'Missing table rows: main={set(names)-main_seen}, cohort={set(cohort_rows)-cohort_seen}')
DOC.write_text('\n'.join(lines) + '\n')

comparison = subprocess.check_output(
    [sys.executable, str(ROOT / 'results/compare_fulltest.py')], text=True, cwd=ROOT)
header = '''# LLM-Gate full test split — gpt-5.4 vs gpt-6-luna (current definitions)

- Date: 2026-10-27 selective rerun of 14 changed definitions (328 test tasks per model); other predictions retained from 2026-10-20.
- Notes: commit 19551bbd, definition mode; gpt-5.4 temperature 0, gpt-6-luna temperature 1.0, max_tokens 32768.
- Split: benchmark/splits.json test (386 samples, 11,671 steps).
- Old 5.4 = results/gate_gpt-5.4_full_old (earlier note text); paired on the tasks present in all three.
- Overall rows average the 11 cohort means with equal weight; `n` is the number of paired steps.

'''
GATE_SUMMARY.write_text(header + comparison)
print('Updated', SUMMARY, DOC, GATE_SUMMARY, sep='\n')
