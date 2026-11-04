"""Compare LLM-Gate with each rater and raters with one another."""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .run_sample import DATA, GATERS, META, OUT, PLAN


def add_counts(dst: dict, truth: np.ndarray, pred: np.ndarray, categories: list[str]) -> None:
    for category in categories:
        t = truth == category
        p = pred == category
        c = dst[category]
        c['tp'] += int(np.count_nonzero(t & p))
        c['fp'] += int(np.count_nonzero(~t & p))
        c['fn'] += int(np.count_nonzero(t & ~p))


def score(c: dict) -> float | None:
    d = 2 * c['tp'] + c['fp'] + c['fn']
    return 2 * c['tp'] / d if d else None


def macro(counts: dict, steps: list[dict] = PLAN['steps']) -> float:
    vals = [score(counts[(str(s['step']), cat)])
            for s in steps for cat in s['annotation categories']]
    vals = [v for v in vals if v is not None]
    if not vals:
        raise ValueError('No evaluable categories')
    return float(np.mean(vals))


def clean_labels(series: pd.Series) -> np.ndarray:
    values = series.astype(object).fillna('Unassigned').replace('', 'Unassigned')
    return values.to_numpy(dtype=str)


def main() -> None:
    completed = json.loads((OUT / 'summary.json').read_text())
    if completed['missing'] or completed['completed'] != completed['expected_sample_gaters']:
        raise ValueError('LLM-Gate run is incomplete')
    samples = [item['sample'] for item in json.loads(
        (DATA / 'dataset_manifest.json').read_text())['files']]
    # Ordered references permit comparing each rater-specific LLM run with
    # another human on exactly the same shared-parent events.
    paired = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}))
    n_shared = defaultdict(int)
    for i, sample in enumerate(samples, 1):
        source = DATA / 'parquet' / f'{sample}.parquet'
        gt_cols = [c for m in META['gater_columns'].values() for c in m.values()]
        gt = pd.read_parquet(source, columns=gt_cols)
        predictions = {}
        for rater in GATERS:
            path = OUT / sample / rater / 'predicted_labels.parquet'
            predictions[rater] = pd.read_parquet(path)
            if len(predictions[rater]) != len(gt):
                raise ValueError(f'{sample}/{rater}: label row count mismatch')
            if not np.array_equal(predictions[rater]['event_index'].to_numpy(),
                                  np.arange(len(gt))):
                raise ValueError(f'{sample}/{rater}: event_index mismatch')
        for step in PLAN['steps']:
            number = str(step['step'])
            canonical = step['annotation column name']
            masks = {}
            labels = {}
            llm = {}
            for rater in GATERS:
                mapping = META['gater_columns'][rater]
                if step['parent'] == 'ALL':
                    mask = np.ones(len(gt), dtype=bool)
                else:
                    parent_col, parent_cat = (x.strip() for x in step['parent'].split(' == ', 1))
                    mask = gt[mapping[parent_col]].to_numpy() == parent_cat
                masks[rater] = mask
                labels[rater] = clean_labels(gt[mapping[canonical]])
                llm[rater] = clean_labels(predictions[rater][canonical])
            for ref in GATERS:
                for other in GATERS:
                    if ref == other:
                        continue
                    common = masks[ref] & masks[other]
                    n = int(np.count_nonzero(common))
                    n_shared[(ref, other, number)] += n
                    if not n:
                        continue
                    truth = labels[ref][common]
                    model_pred = llm[ref][common]
                    human_pred = labels[other][common]
                    for name, pred in [('llm', model_pred), ('human', human_pred)]:
                        add_counts(paired[(ref, other, name, number)], truth, pred,
                                   step['annotation categories'])
        if i % 25 == 0 or i == len(samples):
            print(f'compared {i}/{len(samples)} samples', flush=True)

    # Reduce ordered-pair counts to per-reference and overall counts.
    overall = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    by_ref = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    pair_scores = {}
    for ref in GATERS:
        for other in GATERS:
            if ref == other:
                continue
            pair_counts = {}
            for name in ('llm', 'human'):
                for step in PLAN['steps']:
                    number = str(step['step'])
                    for cat in step['annotation categories']:
                        c = paired[(ref, other, name, number)][cat]
                        pair_counts[(number, cat)] = c
                        for target in (overall[(name, number, cat)],
                                       by_ref[(ref, name, number, cat)]):
                            for k in ('tp', 'fp', 'fn'):
                                target[k] += c[k]
                pair_scores[(ref, other, name)] = macro(pair_counts)

    full = {}
    for ref in GATERS:
        vals = [completed['gaters'][ref][str(s['step'])][cat]['f1']
                for s in PLAN['steps'] for cat in s['annotation categories']]
        full[ref] = float(np.mean([v for v in vals if v is not None]))

    reference = {}
    for ref in GATERS:
        reference[ref] = {
            'llm_full_parent_macro_f1': full[ref],
            'llm_shared_parent_macro_f1': macro({
                (str(s['step']), cat): by_ref[(ref, 'llm', str(s['step']), cat)]
                for s in PLAN['steps'] for cat in s['annotation categories']}),
            'other_raters_shared_parent_macro_f1': macro({
                (str(s['step']), cat): by_ref[(ref, 'human', str(s['step']), cat)]
                for s in PLAN['steps'] for cat in s['annotation categories']}),
        }

    pairs = {}
    for a, b in combinations(GATERS, 2):
        pairs[f'{a}__{b}'] = {
            'rater_pair_shared_parent_macro_f1': pair_scores[(a, b, 'human')],
            'llm_vs_a_shared_parent_macro_f1': pair_scores[(a, b, 'llm')],
            'llm_vs_b_shared_parent_macro_f1': pair_scores[(b, a, 'llm')],
        }

    category = {}
    for step in PLAN['steps']:
        number = str(step['step'])
        for cat in step['annotation categories']:
            category[f'{number}:{cat}'] = {
                name: {'f1': score(overall[(name, number, cat)]),
                       **overall[(name, number, cat)]}
                for name in ('llm', 'human')}
    overall_scores = {
        name: macro({(str(s['step']), cat): overall[(name, str(s['step']), cat)]
                     for s in PLAN['steps'] for cat in s['annotation categories']})
        for name in ('llm', 'human')}
    report = {
        'n_samples': len(samples), 'metric': 'named-category macro F1',
        'protocol': 'GT-parent step tasks; shared-parent comparisons use intersection of two raters parent masks for each sample and step',
        'full_parent_llm_vs_reference': full,
        'shared_parent_overall': overall_scores,
        'shared_parent_by_reference': reference,
        'shared_parent_pairs': pairs,
        'shared_parent_by_category': category,
        'shared_parent_event_counts': {'|'.join(k): v for k, v in n_shared.items()},
    }
    path = OUT / 'comparison_named_category.json'
    path.write_text(json.dumps(report, indent=2))

    lines = [
        '# LLM-Gate vs four raters, and interrater agreement', '',
        'Metric: macro F1 across the nine named gate categories (Unassigned excluded). '
        'Each category F1 pools TP/FP/FN across 194 samples before averaging.',
        '',
        'The LLM-Gate task uses each reference rater’s GT parent population. '
        'For the paired comparison, both the LLM and another rater are scored '
        'against the reference rater on the intersection of their GT parent populations. '
        'These paired scores use exactly the same events for both predictors. '
        'The full-parent LLM score uses all events in that reference parent.', '',
        '| Reference rater | LLM, full parent | LLM, shared parent | Other raters, shared parent | LLM minus raters |',
        '|---|---:|---:|---:|---:|',
    ]
    for ref, d in reference.items():
        gap = d['llm_shared_parent_macro_f1'] - d['other_raters_shared_parent_macro_f1']
        lines.append(f"| {ref} | {d['llm_full_parent_macro_f1']:.3f} | "
                     f"{d['llm_shared_parent_macro_f1']:.3f} | "
                     f"{d['other_raters_shared_parent_macro_f1']:.3f} | {gap:+.3f} |")
    lines += ['', f"Overall shared-parent macro F1: LLM **{overall_scores['llm']:.3f}**, "
              f"other raters **{overall_scores['human']:.3f}** "
              f"(difference **{overall_scores['llm'] - overall_scores['human']:+.3f}**).", '',
              '## Rater pairs', '',
              '| Pair | Rater–rater F1 | LLM vs first | LLM vs second |',
              '|---|---:|---:|---:|']
    for pair, d in pairs.items():
        lines.append(f"| {pair.replace('__', '–')} | {d['rater_pair_shared_parent_macro_f1']:.3f} | "
                     f"{d['llm_vs_a_shared_parent_macro_f1']:.3f} | "
                     f"{d['llm_vs_b_shared_parent_macro_f1']:.3f} |")
    lines += ['', '## Named gate categories', '',
              '| Step | Category | LLM F1 | Rater–rater F1 | Difference |',
              '|---:|---|---:|---:|---:|']
    for step in PLAN['steps']:
        for cat in step['annotation categories']:
            d = category[f"{step['step']}:{cat}"]
            l, h = d['llm']['f1'], d['human']['f1']
            lines.append(f"| {step['step']} | {cat} | {l:.3f} | {h:.3f} | {l-h:+.3f} |")
    lines += ['', 'Interpretation: the LLM receives the reference rater’s GT parent '
              'population, whereas the second human supplied an independent gating '
              'hierarchy. The shared-parent restriction isolates agreement within '
              'a common population; it does not measure end-to-end cascade errors.', '']
    (OUT / 'comparison_named_category.md').write_text('\n'.join(lines))
    print(f'Wrote {path} and {OUT / "comparison_named_category.md"}')


if __name__ == '__main__':
    main()
