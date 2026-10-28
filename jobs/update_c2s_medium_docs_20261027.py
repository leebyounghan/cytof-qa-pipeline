"""Add completed C2S medium runs to the manuscript-facing experiment summary."""
import json
from pathlib import Path

ROOT = Path('/project/kimlab_storage2NB/jaesik/cytof-qa-pipeline')
summary = json.loads((ROOT / 'results/c2s_medium_full_summary.json').read_text())
doc = ROOT / 'docs/experiment_results.md'
text = doc.read_text()
rows = [
    ('C2S grid · GPT-6 luna', 'C2S grid · GPT-6 luna medium', 'c2s_flowdensity_gpt-6-luna-medium'),
    ('C2S grid · GPT-6 luna medium', 'C2S grid · GPT-6 sol medium', 'c2s_flowdensity_gpt-6-sol-medium'),
    ('C2S FlowSOM · GPT-6 luna', 'C2S FlowSOM · GPT-6 luna medium', 'c2s_flowsom_gpt-6-luna-medium'),
    ('C2S FlowSOM · GPT-6 luna medium', 'C2S FlowSOM · GPT-6 sol medium', 'c2s_flowsom_gpt-6-sol-medium'),
]
for preceding, label, dirname in rows:
    if dirname not in summary or summary[dirname]['n_tasks'] != 11671:
        raise ValueError(f'Incomplete summary: {dirname}')
    mean = summary[dirname]['equal_cohort_mean']
    values = [mean[k][m] for k in ('gt', 'rect_gt')
              for m in ('f1_macro_mean', 'balanced_accuracy_mean', 'iou_hull_macro_mean')]
    anchor = f'| {preceding} |'
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if line.startswith(anchor)]
    if len(matches) != 1:
        raise ValueError(f'Expected one anchor for {preceding}, found {len(matches)}')
    new_line = '| ' + label + ' | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |'
    lines.insert(matches[0] + 1, new_line)
    text = '\n'.join(lines) + '\n'
needle = '현재 정의 모델 행은 [11개 방법의 동일 split 집계](../results/vs_gt_rect1_summary.md)에서 가져왔다.'
replacement = (needle + ' 새 C2S medium 네 행은 [별도 전체 test 집계]'
               '(../results/c2s_medium_full_summary.md)에서 가져왔으며, 기존 GPT-6 luna 기본 설정 행과 구분한다.')
if text.count(needle) != 1:
    raise ValueError('Main-table provenance sentence missing or duplicated')
text = text.replace(needle, replacement)
doc.write_text(text)
print(f'Updated {doc} with four C2S medium rows')
