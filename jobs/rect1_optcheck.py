"""Optimiser sanity check: grid resolution and random restarts on a few steps."""
import argparse
import numpy as np
from src.bench import BenchmarkLoader
from src.baselines.rect_oracle.fit_rect1 import Step

p = argparse.ArgumentParser(); BenchmarkLoader.add_cli_args(p); args = p.parse_args()
loader = BenchmarkLoader.from_cli(args, verbose=False)
rng = np.random.default_rng(0)
for ti in loader.iter_tasks_from_cli(args):
    X = ti.X_2d; x, y = X[:, 0].astype(float), X[:, 1].astype(float); gt = np.asarray(ti.y, dtype=object)
    row = []
    for b in (128, 256, 512):
        st = Step(x, y, gt, ti.categories, b)
        code = {c: i for i, c in enumerate(st.categories)}
        g = np.fromiter((code[v] for v in gt), dtype=np.int64, count=len(gt))
        row.append(st.fit(g)[1])
    st = Step(x, y, gt, ti.categories, 256)
    best_rand = 0.0
    for _ in range(20):
        order = list(rng.permutation(st.named))
        rects = {}
        for k in order:
            a = int(rng.integers(0, st.nx)); b2 = int(rng.integers(a + 1, st.nx + 1))
            c = int(rng.integers(0, st.ny)); d = int(rng.integers(c + 1, st.ny + 1))
            r = (a, b2, c, d)
            if st.feasible(k, r, rects): rects[k] = r
        best_rand = max(best_rand, st.ascend(rects)[1])
    print(f'{ti.dataset} {ti.sample} step{ti.task["step"]}: bins128 {row[0]:.4f} bins256 {row[1]:.4f} bins512 {row[2]:.4f} | 20 random restarts @256 {best_rand:.4f}', flush=True)
