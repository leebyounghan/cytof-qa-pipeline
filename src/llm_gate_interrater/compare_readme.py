"""README-compatible per-task macro F1, then equal-weight five-step mean."""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .run_sample import DATA, GATERS, META, OUT, PLAN


def clean(series: pd.Series) -> np.ndarray:
    return series.astype(object).fillna('Unassigned').replace('', 'Unassigned').to_numpy(dtype=str)


def encode(series: pd.Series, categories: list[str]) -> np.ndarray:
    if isinstance(series.dtype, pd.CategoricalDtype):
        lookup = np.full(len(series.cat.categories) + 1, -1, dtype=np.int16)
        target = {name: i for i, name in enumerate(categories)}
        lookup[0] = target['Unassigned']  # null, including outside-parent rows
        for i, name in enumerate(series.cat.categories, 1):
            lookup[i] = target.get('Unassigned' if name == '' else name, -1)
        return lookup[series.cat.codes.to_numpy(dtype=np.int32) + 1]
    return pd.Categorical(clean(series), categories=categories).codes.astype(np.int16)


def task_macro_f1(truth: np.ndarray, pred: np.ndarray, n_categories: int) -> float:
    """Match src.eval's per-task macro F1 over explicit categories."""
    if len(truth) != len(pred) or not len(truth):
        raise ValueError('Task arrays must have equal nonzero length')
    cm = np.bincount(truth * n_categories + pred,
                     minlength=n_categories * n_categories).reshape(n_categories, n_categories)
    tp = np.diag(cm).astype(np.float64)
    gt_count = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    denom = gt_count + pred_count
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_categories), where=denom > 0)
    return round(float(np.mean(f1)), 4)


def balanced(values: dict, key_prefix: tuple) -> float:
    per_step = []
    for step in PLAN['steps']:
        key = (*key_prefix, str(step['step']))
        if not values[key]:
            raise ValueError(f'No tasks for {key}')
        per_step.append(float(np.mean(values[key])))
    return float(np.mean(per_step))


def main() -> None:
    manifest = json.loads((DATA / 'dataset_manifest.json').read_text())
    samples = [item['sample'] for item in manifest['files']]
    scores = defaultdict(list)
    n_empty = defaultdict(int)
    for index, sample in enumerate(samples, 1):
        gt_cols = [column for m in META['gater_columns'].values() for column in m.values()]
        gt = pd.read_parquet(DATA / 'parquet' / f'{sample}.parquet', columns=gt_cols)
        predictions = {}
        sample_metrics = {}
        for rater in GATERS:
            rdir = OUT / sample / rater
            predictions[rater] = pd.read_parquet(rdir / 'predicted_labels.parquet')
            sample_metrics[rater] = json.loads((rdir / 'metrics.json').read_text())
            if len(predictions[rater]) != len(gt):
                raise ValueError(f'{sample}/{rater}: row count mismatch')
        for step in PLAN['steps']:
            number = str(step['step'])
            canonical = step['annotation column name']
            categories = [*step['annotation categories'], 'Unassigned']
            masks, labels, llm = {}, {}, {}
            for rater in GATERS:
                mapping = META['gater_columns'][rater]
                if step['parent'] == 'ALL':
                    mask = np.ones(len(gt), dtype=bool)
                else:
                    parent_col, parent_cat = (p.strip() for p in step['parent'].split(' == ', 1))
                    mask = gt[mapping[parent_col]].to_numpy() == parent_cat
                masks[rater] = mask
                labels[rater] = encode(gt[mapping[canonical]], categories)
                llm[rater] = encode(predictions[rater][canonical], categories)
                if np.any(labels[rater] < 0) or np.any(llm[rater] < 0):
                    raise ValueError(f'{sample}/{rater}/step{number}: unknown label')
                per_category = sample_metrics[rater]['steps'][number]['category_metrics']
                if set(per_category) != set(categories):
                    raise ValueError(f'{sample}/{rater}/step{number}: category mismatch')
                # src.eval maps an absent category's undefined F1 to zero.
                f1 = float(np.mean([per_category[c]['f1'] or 0.0 for c in categories]))
                scores[('full', rater, number)].append(round(f1, 4))
            for ref in GATERS:
                for other in GATERS:
                    if ref == other:
                        continue
                    common = masks[ref] & masks[other]
                    if not common.any():
                        n_empty[(ref, other, number)] += 1
                        continue
                    truth = labels[ref][common]
                    for method, pred in [('llm', llm[ref][common]),
                                         ('human', labels[other][common])]:
                        scores[('shared', ref, other, method, number)].append(
                            task_macro_f1(truth, pred, len(categories)))
        if index % 25 == 0 or index == len(samples):
            print(f'compared {index}/{len(samples)} samples', flush=True)

    reference = {}
    by_step = {}
    for ref in GATERS:
        per_step_full = {s: float(np.mean(scores[('full', ref, s)]))
                         for s in [str(x['step']) for x in PLAN['steps']]}
        per_step_llm = {}
        per_step_human = {}
        for step in PLAN['steps']:
            s = str(step['step'])
            per_step_llm[s] = float(np.mean([
                v for other in GATERS if other != ref
                for v in scores[('shared', ref, other, 'llm', s)]]))
            per_step_human[s] = float(np.mean([
                v for other in GATERS if other != ref
                for v in scores[('shared', ref, other, 'human', s)]]))
        reference[ref] = {
            'llm_full_parent': float(np.mean(list(per_step_full.values()))),
            'llm_shared_parent': float(np.mean(list(per_step_llm.values()))),
            'other_raters_shared_parent': float(np.mean(list(per_step_human.values()))),
            'by_step': {'full_llm': per_step_full, 'shared_llm': per_step_llm,
                        'shared_human': per_step_human},
        }

    pairs = {}
    for a, b in combinations(GATERS, 2):
        pairs[f'{a}__{b}'] = {
            'human_shared_parent': balanced(scores, ('shared', a, b, 'human')),
            'llm_vs_first_shared_parent': balanced(scores, ('shared', a, b, 'llm')),
            'llm_vs_second_shared_parent': balanced(scores, ('shared', b, a, 'llm')),
        }
    overall = {}
    for method in ('llm', 'human'):
        step_means = {}
        for step in PLAN['steps']:
            s = str(step['step'])
            vals = [v for ref in GATERS for other in GATERS if ref != other
                    for v in scores[('shared', ref, other, method, s)]]
            step_means[s] = float(np.mean(vals))
        overall[method] = {'by_step': step_means,
                           'balanced_f1': float(np.mean(list(step_means.values())))}

    # The all-event diagnostic uses a different cell universe. Keep named-only
    # category F1 there, but balance the five steps in the displayed score.
    full_event = json.loads((OUT / 'rater_full_event.json').read_text())
    all_event_pairs = {}
    for a, b in combinations(GATERS, 2):
        pair = f'{a}__{b}'
        step_f1 = {}
        for step in PLAN['steps']:
            vals = [full_event['categories'][f"{step['step']}:{cat}"][pair]['f1']
                    for cat in step['annotation categories']]
            step_f1[str(step['step'])] = float(np.mean(vals))
        all_event_pairs[pair] = {'by_step': step_f1,
                                 'step_balanced_named_f1': float(np.mean(list(step_f1.values())))}

    report = {
        'n_samples': len(samples),
        'metric': 'per-sample per-step macro F1 including Unassigned, then equal-weight mean of five steps',
        'full_parent_llm_by_reference': reference,
        'shared_parent_overall': overall,
        'shared_parent_pairs': pairs,
        'all_event_named_category_diagnostic_pairs': all_event_pairs,
        'empty_shared_parent_tasks': {'|'.join(k): v for k, v in n_empty.items()},
    }
    report_json = json.dumps(report, indent=2)
    (OUT / 'comparison_readme.json').write_text(report_json)
    (OUT / 'comparison.json').write_text(report_json)

    lines = [
        '# LLM-Gate vs four raters: README-compatible F1', '',
        'For each sample and step, macro F1 includes `Unassigned`; absent categories '
        'score zero, as in `src.eval`. Task scores are averaged within each of the '
        'five steps, then the five step means receive equal weight.', '',
        'LLM-Gate uses each reference rater’s GT parent. Shared-parent comparisons '
        'score the LLM and another rater against that reference on exactly the same '
        'events: the intersection of the two raters’ GT parents.', '',
        '| Reference | LLM, full parent | LLM, shared parent | Other raters, shared parent | Difference |',
        '|---|---:|---:|---:|---:|',
    ]
    for ref, d in reference.items():
        l, h = d['llm_shared_parent'], d['other_raters_shared_parent']
        lines.append(f"| {ref} | {d['llm_full_parent']:.3f} | {l:.3f} | {h:.3f} | {l-h:+.3f} |")
    l, h = overall['llm']['balanced_f1'], overall['human']['balanced_f1']
    lines += ['', f'**Overall shared-parent F1:** LLM **{l:.3f}**, other raters **{h:.3f}**; difference **{l-h:+.3f}**.', '',
              '## By step, shared parent', '',
              '| Step | LLM | Other raters | Difference |', '|---:|---:|---:|---:|']
    for step in PLAN['steps']:
        s = str(step['step'])
        l, h = overall['llm']['by_step'][s], overall['human']['by_step'][s]
        lines.append(f'| {s} | {l:.3f} | {h:.3f} | {l-h:+.3f} |')
    lines += ['', '## Rater pairs', '',
              '| Pair | Rater–rater, shared parent | Rater–rater, all events* | LLM vs first | LLM vs second |',
              '|---|---:|---:|---:|---:|']
    for pair, d in pairs.items():
        lines.append(f"| {pair.replace('__', '–')} | {d['human_shared_parent']:.3f} | "
                     f"{all_event_pairs[pair]['step_balanced_named_f1']:.3f} | "
                     f"{d['llm_vs_first_shared_parent']:.3f} | "
                     f"{d['llm_vs_second_shared_parent']:.3f} |")
    lines += ['', '*All-events scores use named categories only. They include '
              'upstream parent disagreements and use a different cell universe, '
              'so they are a separate diagnostic rather than a direct LLM comparison.', '',
              'The LLM was given the reference rater’s GT parent; these scores '
              'do not measure end-to-end cascade errors.', '']
    report_md = '\n'.join(lines)
    (OUT / 'comparison_readme.md').write_text(report_md)
    (OUT / 'comparison.md').write_text(report_md)
    print(f'Wrote {OUT / "comparison_readme.md"}')


if __name__ == '__main__':
    main()
