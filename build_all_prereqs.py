#!/usr/bin/env python3
"""Pre-build Setting 2 in-panel-hard prereqs (gate_task__{slug}.json +
gate_full__{slug}/cells.md) for the test split of each HID.

PREREQ-ONLY. No LLM inference. Pair with a separate experiment runner
(e.g. scripts/run_hard_*.sh) to actually call the model.

Two phases:
  Phase A — task.py per HID, N_PARALLEL in parallel.
  Phase B — prompt.py per (HID, test_sample), N_PARALLEL in parallel.

Skip-existing on both phases. Logs per HID/sample under
results/sweep/_logs/prereq/.

Filters
-------
  --cohorts Acute2020 Acute2021 Vaccine    # keep HIDs whose dataset is in list
  --modes   depletion shift                 # keep HIDs of these modes
  --hids    hiv_acute2020_step29 ...        # explicit allowlist (overrides --cohorts/--modes)
  --phase   A | B | both (default both)

Env:
  N_PARALLEL=96
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

N_PARALLEL = int(os.environ.get('N_PARALLEL', '96'))
LOG_DIR = ROOT / 'results' / 'sweep' / '_logs' / 'prereq'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def collect_hids(
    *,
    cohorts: list[str] | None = None,
    modes: list[str] | None = None,
    hids_filter: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Return (mode, hid, dataset) triples after applying filters.

    Precedence: hids_filter overrides cohorts/modes when set.
    """
    out: list[tuple[str, str, str]] = []
    for path, mode in [('benchmark/hard_depletion.yaml', 'depletion'),
                       ('benchmark/hard_shift.yaml', 'shift')]:
        if modes and mode not in modes:
            continue
        with open(path) as f:
            d = yaml.safe_load(f)
        for t in d['tasks']:
            out.append((mode, t['hard_id'], t['dataset']))

    if hids_filter:
        wanted = set(hids_filter)
        out = [t for t in out if t[1] in wanted]
        missing = wanted - {t[1] for t in out}
        if missing:
            _log(f'[warn] --hids not found in yamls: {sorted(missing)}')
    elif cohorts:
        wanted = set(cohorts)
        out = [t for t in out if t[2] in wanted]
    return out


def collect_test_samples() -> dict[str, list[str]]:
    splits = json.loads(Path('benchmark/splits.json').read_text())['in_panel']
    return {ds: v.get('test', []) for ds, v in splits.items()}


def run_task(args: tuple[str, str]) -> tuple[str, int]:
    mode, hid = args
    flag = '--hard-depletion' if mode == 'depletion' else '--hard-shift'
    log_path = LOG_DIR / f'task_{mode}_{hid}.log'
    cmd = [
        'python3', '-m', 'src.llm_gate.task',
        '--benchmark', 'benchmark/', '--data-dir', 'data/',
        '--split', 'benchmark/splits.json', '--split-set', 'test',
        '--skip-existing', '--n-workers', '1',
        flag, hid,
    ]
    with open(log_path, 'w') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    return f'{mode}/{hid}', rc


def run_prompt(args: tuple[str, str, str, str]) -> tuple[str, int]:
    mode, hid, ds, sample = args
    flag = '--hard-depletion' if mode == 'depletion' else '--hard-shift'
    safe = sample.replace('/', '_').replace(' ', '_')
    log_path = LOG_DIR / f'prompt_{mode}_{hid}_{safe}.log'
    sample_dir = f'benchmark/{ds}/{sample}'
    cmd = [
        'python3', '-m', 'src.llm_gate.prompt',
        '--batch', sample_dir, '--save_md',
        flag, hid,
    ]
    with open(log_path, 'w') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    return f'{mode}/{hid}/{sample}', rc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohorts', nargs='+', default=None,
                   help='Dataset allowlist (e.g. Acute2020 Acute2021 Vaccine).')
    p.add_argument('--modes', nargs='+', default=None,
                   choices=['depletion', 'shift'],
                   help='Modes to build (default: both).')
    p.add_argument('--hids', nargs='+', default=None,
                   help='Explicit HID allowlist (overrides --cohorts / --modes).')
    p.add_argument('--phase', choices=['A', 'B', 'both'], default='both',
                   help='A=task.py only, B=prompt.py only, both=A then B.')
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    hids = collect_hids(cohorts=args.cohorts, modes=args.modes, hids_filter=args.hids)
    test_samples = collect_test_samples()
    n_dep = sum(1 for h in hids if h[0] == 'depletion')
    n_shi = sum(1 for h in hids if h[0] == 'shift')
    _log(f'Selected {len(hids)} hard HIDs ({n_dep} depletion + {n_shi} shift)')
    if args.cohorts:
        _log(f'  --cohorts={args.cohorts}')
    if args.modes:
        _log(f'  --modes={args.modes}')
    if args.hids:
        _log(f'  --hids={len(args.hids)} explicit')
    _log(f'  --phase={args.phase}')
    if not hids:
        _log('Nothing to build — exiting.')
        return 0

    fails_a: list[tuple[str, int]] = []
    fails_b: list[tuple[str, int]] = []

    # ── Phase A: task.py ──
    if args.phase in ('A', 'both'):
        _log(f'Phase A: task.py × {len(hids)} HIDs (parallel={N_PARALLEL})')
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
            futs = {ex.submit(run_task, (m, h)): (m, h) for m, h, _ in hids}
            done = 0
            for f in as_completed(futs):
                tag, rc = f.result()
                done += 1
                if rc != 0:
                    fails_a.append((tag, rc))
                if done % 10 == 0 or done == len(hids):
                    _log(f'  Phase A: {done}/{len(hids)} done')
        _log(f'Phase A finished in {time.time()-t0:.0f}s ({len(fails_a)} failures)')
        if fails_a:
            for tag, rc in fails_a[:10]:
                _log(f'  [FAIL] task.py {tag} rc={rc} (see {LOG_DIR}/task_*.log)')

    # ── Phase B: prompt.py per (hid, sample) ──
    if args.phase in ('B', 'both'):
        jobs_b: list[tuple[str, str, str, str]] = []
        for mode, hid, ds in hids:
            for sample in test_samples.get(ds, []):
                jobs_b.append((mode, hid, ds, sample))
        _log(f'Phase B: prompt.py × {len(jobs_b)} (HID, sample) pairs (parallel={N_PARALLEL})')
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
            futs = {ex.submit(run_prompt, j): j for j in jobs_b}
            done = 0
            for f in as_completed(futs):
                tag, rc = f.result()
                done += 1
                if rc != 0:
                    fails_b.append((tag, rc))
                if done % 100 == 0 or done == len(jobs_b):
                    _log(f'  Phase B: {done}/{len(jobs_b)} done')
        _log(f'Phase B finished in {time.time()-t0:.0f}s ({len(fails_b)} failures)')
        if fails_b:
            for tag, rc in fails_b[:10]:
                _log(f'  [FAIL] prompt.py {tag} rc={rc} (see {LOG_DIR}/prompt_*.log)')

    _log('Prereq build complete.')
    return 0 if not (fails_a or fails_b) else 1


if __name__ == '__main__':
    sys.exit(main())
