"""How much do extra seeds (grid sizes, random restarts) improve the refined optimum?"""
import argparse, time
import numpy as np
from src.bench import BenchmarkLoader
from src.baselines.rect_oracle.fit_rect1 import Step, Exact

p = argparse.ArgumentParser(); BenchmarkLoader.add_cli_args(p); args = p.parse_args()
loader = BenchmarkLoader.from_cli(args, verbose=False)
rng = np.random.default_rng(0)
for ti in loader.iter_tasks_from_cli(args):
    X = ti.X_2d; x, y = X[:, 0].astype(float), X[:, 1].astype(float); gt = np.asarray(ti.y, dtype=object)
    res = {}
    t0 = time.time()
    for b in (128, 256, 512):
        st = Step(x, y, gt, ti.categories, b)
        code = {c: i for i, c in enumerate(st.categories)}
        g = np.fromiter((code[v] for v in gt), dtype=np.int64, count=len(gt))
        order = sorted(st.named, key=lambda k: -st.G[k])
        ex = Exact(x, y, g, st.categories)
        for name, seed in (('modal', st.seed_modal(order)), ('quant', st.seed_quantile(order, g))):
            res[f'{b}_{name}'] = ex.ascend(ex.from_grid(st, st.ascend(seed)[0]))[1]
        if b == 256:
            rand = []
            for _ in range(10):
                rects = {}
                for k in rng.permutation(st.named):
                    a = int(rng.integers(0, st.nx)); b2 = int(rng.integers(a + 1, st.nx + 1))
                    c = int(rng.integers(0, st.ny)); d = int(rng.integers(c + 1, st.ny + 1))
                    if st.feasible(k, (a, b2, c, d), rects): rects[k] = (a, b2, c, d)
                rand.append(ex.ascend(ex.from_grid(st, st.ascend(rects)[0]))[1])
            res['256_rand10'] = max(rand)
    cur = max(res['256_modal'], res['256_quant'])
    print(f"{ti.dataset} step{ti.task['step']}: current {cur:.4f} | all-grid {max(v for k,v in res.items() if 'rand' not in k):.4f} | +rand {max(res.values()):.4f} | {time.time()-t0:.1f}s", flush=True)
