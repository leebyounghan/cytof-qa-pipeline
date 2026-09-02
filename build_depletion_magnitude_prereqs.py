#!/usr/bin/env python3
"""Pre-build cells.md for the depletion magnitude sweep:
   N HIDs × {fraction = 0.0, 0.9, 0.99} combos.

   Output slugs: <HID>_frac0 / <HID>_frac90 / <HID>_frac99
   Files written:
     benchmark/<DS>/<sample>/step_NN/gate_task__<HID>_frac{NN}.json
     benchmark/<DS>/<sample>/step_NN/gate_full__<HID>_frac{NN}/cells.md

   Test split only. 96-way parallel.

Filters
-------
  --cohorts Acute2020 Acute2021 Vaccine    # keep HIDs whose dataset is in list
  --hids    hiv_acute2020_step29 ...        # explicit HID allowlist (overrides --cohorts)
  --fractions 0.0 0.9 0.99                  # override default fraction set
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
LOG_DIR = ROOT / 'results' / 'sweep' / '_logs' / 'prereq_magnitude'
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_FRACTIONS = [0.0, 0.9, 0.99]


def _log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def collect_hids(
    *,
    cohorts: list[str] | None = None,
    hids_filter: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Return (hid, dataset) after applying filters.

    Precedence: hids_filter overrides cohorts when set.
    """
    out: list[tuple[str, str]] = []
    with open('benchmark/hard_depletion.yaml') as f:
        d = yaml.safe_load(f)
    for t in d['tasks']:
        out.append((t['hard_id'], t['dataset']))

    if hids_filter:
        wanted = set(hids_filter)
        kept = [t for t in out if t[0] in wanted]
        missing = wanted - {t[0] for t in kept}
        if missing:
            _log(f'[warn] --hids not found in yaml: {sorted(missing)}')
        return kept
    if cohorts:
        wanted = set(cohorts)
        return [t for t in out if t[1] in wanted]
    return out


def collect_test_samples() -> dict[str, list[str]]:
    splits = json.loads(Path('benchmark/splits.json').read_text())['in_panel']
    return {ds: v.get('test', []) for ds, v in splits.items()}


def run_task(args: tuple[str, float]) -> tuple[str, int]:
    hid, frac = args
    pct = int(round(frac * 100))
    log_path = LOG_DIR / f'task_{hid}_frac{pct}.log'
    cmd = [
        'python3', '-m', 'src.llm_gate.task',
        '--benchmark', 'benchmark/', '--data-dir', 'data/',
        '--split', 'benchmark/splits.json', '--split-set', 'test',
        '--skip-existing', '--n-workers', '1',
        '--hard-depletion', hid,
        '--hard-depletion-fraction', str(frac),
    ]
    with open(log_path, 'w') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    return f'{hid}@frac{pct}', rc


def run_prompt(args: tuple[str, float, str, str]) -> tuple[str, int]:
    hid, frac, ds, sample = args
    pct = int(round(frac * 100))
    safe = sample.replace('/', '_').replace(' ', '_')
    log_path = LOG_DIR / f'prompt_{hid}_frac{pct}_{safe}.log'
    sample_dir = f'benchmark/{ds}/{sample}'
    cmd = [
        'python3', '-m', 'src.llm_gate.prompt',
        '--batch', sample_dir, '--save_md',
        '--hard-depletion', hid,
        '--hard-depletion-fraction', str(frac),
    ]
    with open(log_path, 'w') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
    return f'{hid}@frac{pct}/{sample}', rc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cohorts', nargs='+', default=None,
                   help='Dataset allowlist (e.g. Acute2020 Acute2021 Vaccine).')
    p.add_argument('--hids', nargs='+', default=None,
                   help='Explicit HID allowlist (overrides --cohorts).')
    p.add_argument('--fractions', nargs='+', type=float, default=None,
                   help=f'Fractions to build (default: {DEFAULT_FRACTIONS}).')
    p.add_argument('--phase', choices=['A', 'B', 'both'], default='both',
                   help='A=task.py only, B=prompt.py only, both=A then B.')
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    fractions = args.fractions if args.fractions else DEFAULT_FRACTIONS
    hids = collect_hids(cohorts=args.cohorts, hids_filter=args.hids)
    test_samples = collect_test_samples()
    _log(f'{len(hids)} depletion HIDs × {len(fractions)} fractions = {len(hids)*len(fractions)} combos')
    if args.cohorts:
        _log(f'  --cohorts={args.cohorts}')
    if args.hids:
        _log(f'  --hids={len(args.hids)} explicit')
    _log(f'  --fractions={fractions}')
    _log(f'  --phase={args.phase}')
    if not hids:
        _log('Nothing to build — exiting.')
        return 0

    fails_a: list[tuple[str, int]] = []
    fails_b: list[tuple[str, int]] = []

    # ── Phase A: task.py ──
    if args.phase in ('A', 'both'):
        task_jobs = [(h, f) for h, _ in hids for f in fractions]
        _log(f'Phase A: task.py × {len(task_jobs)} (parallel={N_PARALLEL})')
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
            futs = {ex.submit(run_task, j): j for j in task_jobs}
            done = 0
            for f in as_completed(futs):
                tag, rc = f.result()
                done += 1
                if rc != 0: fails_a.append((tag, rc))
                if done % 20 == 0 or done == len(task_jobs):
                    _log(f'  Phase A: {done}/{len(task_jobs)} done')
        _log(f'Phase A finished in {time.time()-t0:.0f}s ({len(fails_a)} failures)')
        for tag, rc in fails_a[:10]:
            _log(f'  [FAIL] task.py {tag} rc={rc}')

    # ── Phase B: prompt.py per (hid, fraction, sample) ──
    if args.phase in ('B', 'both'):
        prompt_jobs = []
        for hid, ds in hids:
            for sample in test_samples.get(ds, []):
                for frac in fractions:
                    prompt_jobs.append((hid, frac, ds, sample))
        _log(f'Phase B: prompt.py × {len(prompt_jobs)} jobs (parallel={N_PARALLEL})')
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
            futs = {ex.submit(run_prompt, j): j for j in prompt_jobs}
            done = 0
            for f in as_completed(futs):
                tag, rc = f.result()
                done += 1
                if rc != 0: fails_b.append((tag, rc))
                if done % 200 == 0 or done == len(prompt_jobs):
                    _log(f'  Phase B: {done}/{len(prompt_jobs)} done')
        _log(f'Phase B finished in {time.time()-t0:.0f}s ({len(fails_b)} failures)')
        for tag, rc in fails_b[:10]:
            _log(f'  [FAIL] prompt.py {tag} rc={rc}')

    _log('Magnitude prereq build complete.')
    return 0 if not (fails_a or fails_b) else 1


if __name__ == '__main__':
    sys.exit(main())
