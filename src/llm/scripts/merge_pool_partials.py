#!/usr/bin/env python3
"""Reduce step in the pool/shard map-reduce: merge per-shard partial JSONs
back into one pred.json per cohort.

Each pool shard writes ``{output_dir}/_partial/{dataset}__shard{i}of{N}.json``
holding only the (sample, step) groups assigned to that shard. This script
deep-merges those partials at the (sample, step) level — each shard
contributes a disjoint set of (sample, step) keys for any given cohort, so
no value-level conflict resolution is needed.

Output schema matches the single-cohort runner exactly, so downstream
``anchor_metric`` / ``propagate`` / ``src.eval`` work unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


_PARTIAL_RE = re.compile(r'^(?P<ds>.+)__shard(?P<i>\d+)of(?P<n>\d+)\.json$')


def _scan_partials(partial_dir: Path) -> dict[str, dict[int, Path]]:
    """{dataset: {shard_idx: path}} from filenames matching the partial pattern."""
    by_ds: dict[str, dict[int, Path]] = defaultdict(dict)
    for p in sorted(partial_dir.iterdir()):
        m = _PARTIAL_RE.match(p.name)
        if not m:
            continue
        by_ds[m['ds']][int(m['i'])] = p
    return by_ds


def _merge_one_cohort(
    paths: dict[int, Path],
    expected_n_shards: int | None = None,
) -> tuple[dict, int]:
    """Deep-merge per-shard partials for one cohort. Returns (merged, n_cells)."""
    merged_samples: dict = {}
    meta = None
    seen_n = set()
    for shard_idx in sorted(paths):
        data = json.loads(paths[shard_idx].read_text())
        if meta is None:
            meta = dict(data['meta'])
        for sample_name, sample in data.get('samples', {}).items():
            tgt = merged_samples.setdefault(sample_name, {'steps': {}})
            for step_key, step in sample.get('steps', {}).items():
                if step_key in tgt['steps']:
                    raise ValueError(
                        f'duplicate (sample={sample_name}, step={step_key}) '
                        f'across shards in cohort partials'
                    )
                tgt['steps'][step_key] = step
        seen_n.add(int(paths[shard_idx].stem.rsplit('of', 1)[-1]))

    if len(seen_n) > 1:
        raise ValueError(f'inconsistent N in shard filenames: {sorted(seen_n)}')
    if expected_n_shards is not None and seen_n and expected_n_shards not in seen_n:
        raise ValueError(
            f'expected n_shards={expected_n_shards} but partials say {seen_n}'
        )

    assert meta is not None
    meta['n_samples'] = len(merged_samples)
    step_set = {int(s) for sample in merged_samples.values()
                for s in sample['steps'].keys()}
    meta['n_steps'] = len(step_set)
    out = {'meta': meta, 'samples': dict(sorted(merged_samples.items()))}
    n_cells = sum(len(step['cells'])
                  for sample in merged_samples.values()
                  for step in sample['steps'].values())
    return out, n_cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', required=True,
                        help='Pool output dir (contains _partial/ subdir).')
    parser.add_argument('--n_shards', type=int, default=0,
                        help='Expected number of shards per cohort. '
                             '0 = infer from filenames; non-zero = enforce.')
    parser.add_argument('--name_template',
                        default='{dataset}.json',
                        help='Output filename template per cohort. '
                             "Tokens: {dataset}. Default: '{dataset}.json'.")
    parser.add_argument('--keep_partials', action='store_true',
                        help='Keep _partial/ files after successful merge.')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    partial_dir = out_dir / '_partial'
    if not partial_dir.is_dir():
        print(f'[ERR] partial dir not found: {partial_dir}', file=sys.stderr)
        return 1

    by_ds = _scan_partials(partial_dir)
    if not by_ds:
        print(f'[ERR] no partial files in {partial_dir}', file=sys.stderr)
        return 1

    expected = args.n_shards or None
    rc = 0
    for ds, paths in sorted(by_ds.items()):
        present = sorted(paths)
        n_seen = len(present)
        # Infer expected from any filename when not supplied
        if expected is None:
            inferred = int(next(iter(paths.values())).stem.rsplit('of', 1)[-1])
        else:
            inferred = expected
        missing = [i for i in range(inferred) if i not in paths]
        if missing:
            print(f'[ERR] {ds}: missing shard partials {missing} '
                  f'(have {present}, expected {inferred})', file=sys.stderr)
            rc = 1
            continue
        merged, n_cells = _merge_one_cohort(paths, expected_n_shards=inferred)
        out_path = out_dir / args.name_template.format(dataset=ds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        print(f'Merged {ds}: {n_seen} shards → {out_path}  '
              f'({merged["meta"]["n_samples"]} samples, '
              f'{merged["meta"]["n_steps"]} steps, {n_cells:,} cells)')
        if not args.keep_partials:
            for p in paths.values():
                p.unlink()

    if not args.keep_partials and rc == 0:
        try:
            partial_dir.rmdir()
        except OSError:
            pass

    return rc


if __name__ == '__main__':
    sys.exit(main())
