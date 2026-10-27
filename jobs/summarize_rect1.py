"""Summarise the rect-1 oracle run: per-cohort eval metrics, fit/eval agreement, absent categories."""
import glob
import json

import numpy as np

DS = ['Acute2020', 'Acute2021', 'Bjornson', 'Vaccine', 'FR-FCM-Z74D_hc', 'FR-FCM-Z74D_tissue',
      'FRDR_covid19', 'Lyoplate_DC', 'Lyoplate_bcell', 'Lyoplate_tcell', 'Lyoplate_treg']
rows, N = [], 0
for ds in DS:
    o = json.load(open(f'results/rect1_oracle/{ds}/eval_summary.json'))['overall']
    fd = json.load(open(f'results/flowdensity_rule/{ds}/eval_summary.json'))['overall']
    n = o['n_tasks']; N += n
    rows.append(dict(dataset=ds, n_steps=n, **{k: o[k] for k in ('f1_macro_mean', 'f1_weighted_mean',
                     'balanced_accuracy_mean', 'population_proportion_error_mean')},
                     flowdensity_rule_f1_macro=fd['f1_macro_mean']))
fit, absent, nrect, mism = [], 0, [], 0
for f in glob.glob('results/rect1_oracle/*/*/step_*/prediction.json'):
    info = json.load(open(f))['info']
    if info['n_parent'] == 0:
        continue
    fit.append(info['f1_macro_fit']); absent += len(info['absent']); nrect.append(len(info['gates']))
keys = ('f1_macro_mean', 'f1_weighted_mean', 'balanced_accuracy_mean', 'population_proportion_error_mean')
summary = dict(per_cohort=rows, all=dict(n_steps=N, aggregation='equal_cohort_mean',
               **{k: sum(row[k] for row in rows) / len(DS) for k in keys},
               flowdensity_rule_f1_macro=sum(row['flowdensity_rule_f1_macro'] for row in rows) / len(DS)),
               fit=dict(n=len(fit), mean=float(np.mean(fit)), quantiles_5_25_50_75=np.quantile(fit, [.05, .25, .5, .75]).tolist(),
                        absent_categories=absent, rects_total=int(np.sum(nrect))))
json.dump(summary, open('results/rect1_oracle/summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
