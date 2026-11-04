"""Joint, nonoverlapping one-rectangle-per-label GT-informed fitting.

The objective is the step macro-F1 over *all* named categories and
Unassigned. Multi-start grid search repositions one or two complete category
rectangles at a time, then exact cell-rank coordinate refinement is applied.
This is a best-found solution, not a certificate of global optimality.
"""

from itertools import combinations

import numpy as np

from src.baselines.rect_oracle.fit_rect1 import Exact, Step, UNASSIGNED


def candidate_boxes(st, k, gt_codes):
    """Diverse complete-rectangle proposals for a category on this grid."""
    mask = gt_codes == k
    if not mask.any():
        return []
    xb, yb = st.xb[mask], st.yb[mask]
    boxes = set()

    def add_subset(select, qlo=0.0, qhi=1.0):
        if not np.any(select):
            return
        xx, yy = xb[select], yb[select]
        x0 = int(np.quantile(xx, qlo)); x1 = int(np.quantile(xx, qhi)) + 1
        y0 = int(np.quantile(yy, qlo)); y1 = int(np.quantile(yy, qhi)) + 1
        boxes.add((x0, x1, y0, y1))

    all_cells = np.ones(len(xb), dtype=bool)
    for lo, hi in [(0, 1), (0.01, 0.99), (0.05, 0.95), (0.1, 0.9)]:
        add_subset(all_cells, lo, hi)
    for axis in (xb, yb):
        for cut in np.unique(np.quantile(axis, [0.25, 0.5, 0.75]).astype(int)):
            for select in (axis <= cut, axis > cut):
                if select.sum() >= max(2, 0.05 * len(axis)):
                    add_subset(select)
                    add_subset(select, 0.05, 0.95)
    h = st.H[k]
    for flat in np.argsort(h, axis=None)[-4:][::-1]:
        i, j = np.unravel_index(int(flat), h.shape)
        if h[i, j] == 0:
            continue
        for radius in (0, 2, 8):
            boxes.add((max(0, i - radius), min(st.nx, i + radius + 1),
                       max(0, j - radius), min(st.ny, j + radius + 1)))
    boxes.add((0, st.nx, 0, st.ny))
    return sorted(boxes)


def ranked_insertion(st, fixed, k, pool, limit):
    feasible = [r for r in pool if st.feasible(k, r, fixed)]
    if not feasible:
        return []
    a = np.asarray(feasible, dtype=np.int64)
    counts = st.counts(a[:, 0], a[:, 1], a[:, 2], a[:, 3])
    scores = st.score(fixed, (k, counts))
    order = np.argsort(scores)[::-1][:limit]
    return [(feasible[i], float(scores[i])) for i in order]


def replace_block(st, current, block, pools, first_keep=4, second_keep=2):
    """Reposition complete rectangles, choosing by global macro-F1."""
    fixed = {k: r for k, r in current.items() if k not in block}
    best = st.ascend(fixed)
    orders = [block] if len(block) == 1 else [block, block[::-1]]
    for order in orders:
        first = order[0]
        first_choices = ranked_insertion(st, fixed, first, pools[first], first_keep)
        for r1, _ in first_choices:
            partial = {**fixed, first: r1}
            if len(order) == 1:
                trial = st.ascend(partial)
                if trial[1] > best[1] + 1e-12:
                    best = trial
                continue
            second = order[1]
            second_choices = ranked_insertion(st, partial, second, pools[second], second_keep)
            for r2, _ in second_choices:
                trial = st.ascend({**partial, second: r2})
                if trial[1] > best[1] + 1e-12:
                    best = trial
    return best


def joint_refine(st, seed, pools, max_passes=2):
    current, score = st.ascend(seed)
    blocks = [(k,) for k in st.named]
    blocks += list(combinations(st.named, 2))
    for _ in range(max_passes):
        improved = False
        for block in blocks:
            trial, value = replace_block(st, current, block, pools)
            if value > score + 1e-12:
                current, score, improved = trial, value, True
        if not improved:
            break
    return current, score


def old_exact_rects(ex, gates):
    rects = {}
    for category, gate in gates.items():
        if category == UNASSIGNED:
            continue
        if isinstance(gate, list):
            raise ValueError(f'Old rect-1 gate has multiple rectangles: {category}')
        k = ex.categories.index(category)
        r = (int(np.searchsorted(ex.ux, gate['x_min'], side='left')),
             int(np.searchsorted(ex.ux, gate['x_max'], side='right')),
             int(np.searchsorted(ex.uy, gate['y_min'], side='left')),
             int(np.searchsorted(ex.uy, gate['y_max'], side='right')))
        if r[0] >= r[1] or r[2] >= r[3]:
            raise ValueError(f'Empty old rectangle: {category} {r}')
        rects[k] = r
    return rects


def fit_task_joint(ti, old_prediction, grid_bins=(64, 128), max_passes=2):
    xy = ti.X_2d.astype(float)
    x, y = xy[:, 0], xy[:, 1]
    categories = list(ti.categories)
    code = {c: i for i, c in enumerate(categories)}
    gt_codes = np.fromiter((code[v] for v in ti.y), dtype=np.int64, count=len(ti.y))
    ex = Exact(x, y, gt_codes, categories)
    old_rects = old_exact_rects(ex, old_prediction['info']['gates'])
    old_labels = np.asarray(old_prediction['predicted_labels'], dtype=object)
    if not np.array_equal(ex.labels(old_rects), old_labels):
        raise AssertionError(f'Old gates do not reproduce saved labels: {ti.task_id}')
    old_score = ex.total({k: ex.counts(r) for k, r in old_rects.items()})
    saved_score = old_prediction['info']['f1_macro_fit']
    if abs(old_score - saved_score) > 1e-8:
        raise AssertionError(f'Old score mismatch: {ti.task_id} {old_score} vs {saved_score}')
    best_rects, best_score = old_rects, old_score
    for bins in grid_bins:
        st = Step(x, y, np.asarray(ti.y, dtype=object), categories, bins)
        pools = {k: candidate_boxes(st, k, gt_codes) for k in st.named}
        order = sorted(st.named, key=lambda k: -st.G[k])
        seeds = [st.seed_modal(order), st.seed_quantile(order, gt_codes),
                 st.seed_modal(order[::-1]), st.seed_quantile(order[::-1], gt_codes)]
        for seed in seeds:
            grid_rects, _ = joint_refine(st, seed, pools, max_passes=max_passes)
            exact_rects, exact_score = ex.ascend(ex.from_grid(st, grid_rects))
            if exact_score > best_score + 1e-12:
                best_rects, best_score = exact_rects, exact_score
    labels = ex.labels(best_rects)
    gates = {categories[k]: ex.bounds(r, x, y) for k, r in best_rects.items()}
    return labels, {'f1_macro_fit': best_score, 'rect1_original_f1': old_score,
                    'gates': gates, 'absent': [categories[k] for k in ex.named if k not in best_rects],
                    'grid_bins': list(grid_bins), 'max_passes': max_passes,
                    'objective': 'all-category macro-F1 including Unassigned',
                    'optimization': 'multi-start single/pair complete-rectangle replacement plus exact refinement'}
