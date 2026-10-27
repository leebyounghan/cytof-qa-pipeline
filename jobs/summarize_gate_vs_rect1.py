"""Part B table (test split): LLM-Gate macro-F1 vs annotation GT and vs rect-1 GT, next to the rect-1 ceiling."""
import json

DS = ['Acute2020', 'Acute2021', 'Bjornson', 'Vaccine', 'FR-FCM-Z74D_hc', 'FR-FCM-Z74D_tissue',
      'FRDR_covid19', 'Lyoplate_DC', 'Lyoplate_bcell', 'Lyoplate_tcell', 'Lyoplate_treg']
MODELS = ('gpt-5.4', 'gpt-6-luna')
rows, N = [], 0
for ds in DS:
    o = json.load(open(f'results/rect1_oracle/{ds}/eval_summary.json'))
    n = o['n_tasks']
    row = {'dataset': ds, 'n_steps': n, 'rect1_ceiling': o['overall']['f1_macro_mean']}
    for m in MODELS:
        a = json.load(open(f'results/gate_{m}_full/predictions/{ds}/eval_summary.json'))
        b = json.load(open(f'results/gate_{m}_full/predictions/{ds}/eval_summary__gt_rect1_oracle.json'))
        if a['n_tasks'] != n or b['n_tasks'] != n:
            raise ValueError(f'{ds} {m}: step counts differ (oracle {n}, vs-gt {a["n_tasks"]}, vs-rect {b["n_tasks"]})')
        row[f'{m}_vs_gt'] = a['overall']['f1_macro_mean']
        row[f'{m}_vs_rect1'] = b['overall']['f1_macro_mean']
    N += n
    rows.append(row)
keys = ['rect1_ceiling'] + [f'{m}_{suffix}' for m in MODELS for suffix in ('vs_gt', 'vs_rect1')]
out = {'per_cohort': rows, 'all': {'n_steps': N, 'aggregation': 'equal_cohort_mean',
       **{k: sum(row[k] for row in rows) / len(DS) for k in keys}}}
json.dump(out, open('results/rect1_oracle/gate_vs_rect1.json', 'w'), indent=1)
print(json.dumps(out['all'], indent=1))
