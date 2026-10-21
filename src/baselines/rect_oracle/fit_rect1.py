"""
fit_rect1.py — rectangle oracle ("rect-1") for LLM-Gate-style gating.

For every step, fits at most one axis-aligned rectangle per named category
to the ground-truth labels of the step's parent population, maximising the
step's macro-F1 exactly as ``src.eval.compute_metrics`` computes it (mean F1
over ``task.json`` categories, ``Unassigned`` included). Rectangles of
different categories never overlap; cells outside every rectangle are
``Unassigned``. A category may get no rectangle.

The result is the best score any single-rectangle-per-category gating can
reach on that step (up to the optimiser and the grid), and its labels serve
as a rectangle-shaped ground truth.

Two stages:

1. Grid seeds. For each grid size in ``--grid-bins`` (default 128, 256,
   512), each axis is cut into that many equal-count bins and rectangle
   edges move on the grid by coordinate ascent (every feasible position of
   one edge at a time), from two starts (per-category modal bin;
   per-category 5–95 % quantile box) — six seeds in total.
2. Exact refinement. Every grid solution is refined by the same coordinate
   ascent at cell resolution: an edge may sit between any two neighbouring
   distinct values on its axis. Because rectangles never overlap, moving one
   edge changes only that category's and Unassigned's F1, so every candidate
   position is scored at once from per-label cumulative counts. The better
   refined solution is kept.

Edges always sit halfway between neighbouring distinct values, so no cell
lies on an edge.

Output per step: results/rect1_oracle/{dataset}/{sample}/step_NN/prediction.json

Usage:
    python -m src.baselines.rect_oracle.fit_rect1 --output-dir results/rect1_oracle \\
        --datasets Bjornson --split benchmark/splits.json --split-set test
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.bench import BenchmarkLoader

METHOD = 'rect1_oracle'
UNASSIGNED = 'Unassigned'


def axis_edges(v: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior bin edges halfway between neighbouring sorted values."""
    s = np.sort(v)
    pos = np.unique(np.linspace(0, len(s), n_bins + 1)[1:-1].astype(int))
    pos = pos[(pos > 0) & (pos < len(s))]
    edges = (s[pos - 1] + s[pos]) / 2.0
    edges = edges[s[pos - 1] < s[pos]]          # drop edges inside tied runs
    return np.unique(edges)


class Step:
    def __init__(self, x, y, gt, categories, n_bins):
        self.categories = list(categories)
        if UNASSIGNED not in self.categories:
            raise ValueError(f'{UNASSIGNED!r} missing from categories {categories}')
        unknown = set(np.unique(gt)) - set(self.categories)
        if unknown:
            raise ValueError(f'GT labels {sorted(unknown)} not in categories {categories}')
        self.ex, self.ey = axis_edges(x, n_bins), axis_edges(y, n_bins)
        self.xb = np.searchsorted(self.ex, x)
        self.yb = np.searchsorted(self.ey, y)
        self.nx, self.ny = len(self.ex) + 1, len(self.ey) + 1
        L = len(self.categories)
        code = {c: i for i, c in enumerate(self.categories)}
        g = np.fromiter((code[v] for v in gt), dtype=np.int64, count=len(gt))
        H = np.zeros((L, self.nx, self.ny), dtype=np.float64)
        np.add.at(H, (g, self.xb, self.yb), 1.0)
        self.H = H
        self.S = np.zeros((L, self.nx + 1, self.ny + 1))
        self.S[:, 1:, 1:] = H.cumsum(1).cumsum(2)
        self.G = H.sum((1, 2))
        self.N = float(len(gt))
        self.u = code[UNASSIGNED]
        self.named = [i for i in range(L) if i != self.u]

    def counts(self, x0, x1, y0, y1):
        """Per-label counts in half-open bin rect(s); args may be arrays (→ (P, L))."""
        S = self.S
        return (S[:, x1, y1] - S[:, x0, y1] - S[:, x1, y0] + S[:, x0, y0]).T

    def score(self, rects, override=None):
        """Macro-F1 for rects {label_idx: (x0,x1,y0,y1)}; override=(k, cand_counts (P,L))."""
        L = len(self.categories)
        base = {k: self.counts(*r) for k, r in rects.items() if override is None or k != override[0]}
        P = 1 if override is None else override[1].shape[0]
        R = np.zeros((P, L, L))                   # R[p, pred_label, gt_label]
        for k, c in base.items():
            R[:, k, :] = c
        if override is not None:
            R[:, override[0], :] = override[1]
        tp = np.einsum('pkk->pk', R).copy()
        pred = R.sum(2)
        in_rect = R.sum(1)                        # per gt label, cells inside some rect
        tp[:, self.u] = self.G[self.u] - in_rect[:, self.u]
        pred[:, self.u] = self.N - pred.sum(1) + pred[:, self.u]
        gt = self.G[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            prec = np.where(pred > 0, tp / pred, 0.0)
            rec = np.where(gt > 0, tp / gt, 0.0)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        return f1.mean(1)

    @staticmethod
    def overlap(a, b):
        return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]

    def feasible(self, k, r, rects):
        return all(not self.overlap(r, o) for j, o in rects.items() if j != k)

    def edge_candidates(self, k, rects, e):
        """All feasible rects that move edge e (0..3) of rects[k]."""
        r = list(rects[k])
        n = self.nx if e < 2 else self.ny
        cands = []
        for p in range(n + 1):
            q = r.copy(); q[e] = p
            if q[0] < q[1] and q[2] < q[3] and self.feasible(k, tuple(q), rects):
                cands.append(tuple(q))
        return cands

    def ascend(self, rects):
        rects = dict(rects)
        best = float(self.score(rects)[0])
        improved = True
        while improved:
            improved = False
            for k in list(rects):
                if k not in rects:
                    continue
                without = {j: r for j, r in rects.items() if j != k}
                s = float(self.score(without)[0])
                if s > best + 1e-12:
                    rects, best, improved = without, s, True
                    continue
                for e in range(4):
                    cands = self.edge_candidates(k, rects, e)
                    if not cands:
                        continue
                    arr = np.array(cands)
                    sc = self.score(rects, (k, self.counts(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])))
                    i = int(np.argmax(sc))
                    if sc[i] > best + 1e-12:
                        rects[k], best, improved = cands[i], float(sc[i]), True
        return rects, best

    def seed_modal(self, order):
        rects = {}
        majority = self.H.argmax(0)
        for k in order:
            if self.G[k] == 0:
                continue
            m = np.where(majority == k, self.H[k], 0)
            for flat in np.argsort(m, axis=None)[::-1]:
                i, j = np.unravel_index(flat, m.shape)
                if m[i, j] == 0:
                    break
                r = (int(i), int(i) + 1, int(j), int(j) + 1)
                if self.feasible(k, r, rects):
                    rects[k] = r
                    break
        return rects

    def seed_quantile(self, order, gt_codes):
        rects = {}
        for k in order:
            mask = gt_codes == k
            if not mask.any():
                continue
            xq = np.quantile(self.xb[mask], [0.05, 0.95])
            yq = np.quantile(self.yb[mask], [0.05, 0.95])
            r = (int(xq[0]), int(xq[1]) + 1, int(yq[0]), int(yq[1]) + 1)
            if self.feasible(k, r, rects):
                rects[k] = r
        modal = self.seed_modal([k for k in order if k not in rects])
        for k, r in modal.items():
            if self.feasible(k, r, rects):
                rects[k] = r
        return rects

    def fit(self, gt_codes):
        order = sorted(self.named, key=lambda k: -self.G[k])
        runs = [self.ascend(self.seed_modal(order)),
                self.ascend(self.seed_quantile(order, gt_codes))]
        return max(runs, key=lambda t: t[1])

    def bounds(self, r, x, y):
        """Rect in data units; open sides become the parent's data min/max."""
        lo_x = self.ex[r[0] - 1] if r[0] > 0 else float(x.min())
        hi_x = self.ex[r[1] - 1] if r[1] < self.nx else float(x.max())
        lo_y = self.ey[r[2] - 1] if r[2] > 0 else float(y.min())
        hi_y = self.ey[r[3] - 1] if r[3] < self.ny else float(y.max())
        return {'x_min': float(lo_x), 'x_max': float(hi_x), 'y_min': float(lo_y), 'y_max': float(hi_y)}

    def labels(self, rects):
        out = np.full(len(self.xb), UNASSIGNED, dtype=object)
        for k, (x0, x1, y0, y1) in rects.items():
            m = (self.xb >= x0) & (self.xb < x1) & (self.yb >= y0) & (self.yb < y1)
            if (out[m] != UNASSIGNED).any():
                raise AssertionError('rectangles overlap')
            out[m] = self.categories[k]
        return out


class Exact:
    """Coordinate ascent at cell resolution; rects are half-open rank cuts."""

    def __init__(self, x, y, g, categories):
        self.categories = list(categories)
        self.L = len(self.categories)
        self.u = self.categories.index(UNASSIGNED)
        self.named = [i for i in range(self.L) if i != self.u]
        self.ux, self.xi = np.unique(x, return_inverse=True)
        self.uy, self.yi = np.unique(y, return_inverse=True)
        self.g = g
        self.G = np.bincount(g, minlength=self.L).astype(float)
        self.N = float(len(g))

    def cut(self, edge_values, axis_values, v):
        """Grid bin index → rank cut (number of distinct values below the edge)."""
        if v == 0:
            return 0
        if v == len(edge_values) + 1:
            return len(axis_values)
        return int(np.searchsorted(axis_values, edge_values[v - 1]))

    def from_grid(self, st, rects):
        return {k: (self.cut(st.ex, self.ux, r[0]), self.cut(st.ex, self.ux, r[1]),
                    self.cut(st.ey, self.uy, r[2]), self.cut(st.ey, self.uy, r[3]))
                for k, r in rects.items()}

    def mask(self, r):
        return (self.xi >= r[0]) & (self.xi < r[1]) & (self.yi >= r[2]) & (self.yi < r[3])

    def counts(self, r):
        return np.bincount(self.g[self.mask(r)], minlength=self.L).astype(float)

    def f1(self, tp, pred, gt):
        with np.errstate(divide='ignore', invalid='ignore'):
            prec = np.where(pred > 0, tp / pred, 0.0)
            rec = np.where(gt > 0, tp / gt, 0.0)
            return np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

    def total(self, cnt):
        """Macro-F1 from per-rect counts {k: (L,)}."""
        f = 0.0
        in_u = sum((c[self.u] for c in cnt.values()), 0.0)
        pred_named = sum((c.sum() for c in cnt.values()), 0.0)
        for k, c in cnt.items():
            f += float(self.f1(c[k], c.sum(), self.G[k]))
        f += float(self.f1(self.G[self.u] - in_u, self.N - pred_named, self.G[self.u]))
        return f / self.L

    @staticmethod
    def overlap_1d(a0, a1, b0, b1):
        return a0 < b1 and b0 < a1

    def edge_scan(self, k, rects, cnt, e):
        """Best feasible position for edge e of rect k → (score, new_rect)."""
        x0, x1, y0, y1 = rects[k]
        on_x = e < 2
        lo_o, hi_o = (y0, y1) if on_x else (x0, x1)
        a0, a1 = (x0, x1) if on_x else (y0, y1)
        n_axis = len(self.ux) if on_x else len(self.uy)
        lo, hi = 0, n_axis
        for j, o in rects.items():
            if j == k:
                continue
            p0, p1, q0, q1 = (o[0], o[1], o[2], o[3]) if on_x else (o[2], o[3], o[0], o[1])
            if not self.overlap_1d(lo_o, hi_o, q0, q1):
                continue
            if p1 <= a0:
                lo = max(lo, p1)
            elif p0 >= a1:
                hi = min(hi, p0)
        ai = self.xi if on_x else self.yi
        oi = self.yi if on_x else self.xi
        sel = (oi >= lo_o) & (oi < hi_o)
        others = {j: c for j, c in cnt.items() if j != k}
        const = sum(float(self.f1(c[j], c.sum(), self.G[j])) for j, c in others.items())
        in_u = sum((c[self.u] for c in others.values()), 0.0)
        pred_o = sum((c.sum() for c in others.values()), 0.0)
        if e % 2 == 1:            # move the upper edge: candidates a0+1 .. hi
            m = sel & (ai >= a0) & (ai < hi)
            W = hi - a0
            H = np.bincount(self.g[m] * W + (ai[m] - a0), minlength=self.L * W).reshape(self.L, W)
            C = H.cumsum(1).T.astype(float)                      # C[i] → cut a0+i+1
            pos = np.arange(a0 + 1, hi + 1)
        else:                     # move the lower edge: candidates lo .. a1-1
            m = sel & (ai >= lo) & (ai < a1)
            W = a1 - lo
            H = np.bincount(self.g[m] * W + (ai[m] - lo), minlength=self.L * W).reshape(self.L, W)
            C = H[:, ::-1].cumsum(1)[:, ::-1].T.astype(float)    # C[i] → cut lo+i
            pos = np.arange(lo, a1)
        tp_k, pred_k = C[:, k], C.sum(1)
        f = (const + self.f1(tp_k, pred_k, self.G[k])
             + self.f1(self.G[self.u] - in_u - C[:, self.u], self.N - pred_o - pred_k,
                       self.G[self.u])) / self.L
        i = int(np.argmax(f))
        new = list(rects[k]); new[e] = int(pos[i])
        return float(f[i]), tuple(new)

    def ascend(self, rects):
        rects = dict(rects)
        cnt = {k: self.counts(r) for k, r in rects.items()}
        best = self.total(cnt)
        improved = True
        while improved:
            improved = False
            for k in list(rects):
                without = {j: c for j, c in cnt.items() if j != k}
                s = self.total(without)
                if s > best + 1e-12:
                    del rects[k]; cnt = without; best = s; improved = True
                    continue
                for e in range(4):
                    s, r = self.edge_scan(k, rects, cnt, e)
                    if s > best + 1e-12:
                        rects[k] = r; cnt[k] = self.counts(r); best = s; improved = True
        return rects, best

    @staticmethod
    def _edge(u, c, data_end):
        """Cut c on distinct values u → data value (open side → data_end)."""
        if c == 0 or c == len(u):
            return float(data_end)
        return float((u[c - 1] + u[c]) / 2.0)

    def bounds(self, r, x, y):
        return {'x_min': self._edge(self.ux, r[0], x.min()), 'x_max': self._edge(self.ux, r[1], x.max()),
                'y_min': self._edge(self.uy, r[2], y.min()), 'y_max': self._edge(self.uy, r[3], y.max())}

    def labels(self, rects):
        out = np.full(len(self.g), UNASSIGNED, dtype=object)
        for k, r in rects.items():
            m = self.mask(r)
            if (out[m] != UNASSIGNED).any():
                raise AssertionError('rectangles overlap')
            out[m] = self.categories[k]
        return out


def fit_task(ti, grid_bins):
    X = ti.X_2d
    x, y = X[:, 0].astype(float), X[:, 1].astype(float)
    gt = np.asarray(ti.y, dtype=object)
    ex = None
    runs = []
    for n_bins in grid_bins:
        st = Step(x, y, gt, ti.categories, n_bins)
        code = {c: i for i, c in enumerate(st.categories)}
        gt_codes = np.fromiter((code[v] for v in gt), dtype=np.int64, count=len(gt))
        if ex is None:
            ex = Exact(x, y, gt_codes, st.categories)
        order = sorted(st.named, key=lambda k: -st.G[k])
        for seed in (st.seed_modal(order), st.seed_quantile(order, gt_codes)):
            runs.append(ex.ascend(ex.from_grid(st, st.ascend(seed)[0])))
    rects, best = max(runs, key=lambda t: t[1])
    labels = ex.labels(rects)
    gates = {ex.categories[k]: ex.bounds(r, x, y) for k, r in rects.items()}
    return labels, {'f1_macro_fit': best, 'gates': gates,
                    'absent': [ex.categories[k] for k in ex.named if k not in rects],
                    'grid_bins': list(grid_bins)}


def main():
    parser = argparse.ArgumentParser(description='rect-1 oracle: best non-overlapping single rectangle per category')
    parser.add_argument('--output-dir', type=Path, default=Path('results/rect1_oracle'))
    parser.add_argument('--grid-bins', type=int, nargs='+', default=[128, 256, 512])
    BenchmarkLoader.add_cli_args(parser)
    args = parser.parse_args()

    loader = BenchmarkLoader.from_cli(args)
    n = 0
    for ti in loader.iter_tasks_from_cli(args):
        out = args.output_dir / ti.dataset / ti.sample / ti.step_dir_name
        out.mkdir(parents=True, exist_ok=True)
        if len(ti.y) == 0:
            pred = {'task_id': ti.task_id, 'method': METHOD, 'predicted_labels': [],
                    'info': {'n_parent': 0}}
        else:
            labels, info = fit_task(ti, args.grid_bins)
            pred = {'task_id': ti.task_id, 'method': METHOD,
                    'predicted_labels': labels.tolist(),
                    'info': {'n_parent': int(len(labels)), **info}}
        (out / 'prediction.json').write_text(json.dumps(pred, ensure_ascii=False))
        n += 1
    print(f'fit_rect1: {n} steps written under {args.output_dir}')


if __name__ == '__main__':
    main()
