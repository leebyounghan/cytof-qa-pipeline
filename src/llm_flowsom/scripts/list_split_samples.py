#!/usr/bin/env python3
"""Print sample names for a (dataset, split) combo from benchmark/splits.json.

Used by the sweep launchers to restrict a run to e.g. test-only samples.

Exits 0 with a space-separated list on stdout, or 1 if the split is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', default='benchmark/splits.json')
    parser.add_argument('--dataset', required=True,
                        help='e.g. Acute2020, Acute2021')
    parser.add_argument('--split', required=True,
                        help='e.g. test, val, train, test_samples, dev_samples')
    parser.add_argument('--group', default='in_panel')
    args = parser.parse_args()

    data = json.loads(Path(args.splits).read_text())
    ds = data.get(args.group, {}).get(args.dataset)
    if ds is None:
        print(f'[ERR] dataset not found: {args.group}/{args.dataset}',
              file=sys.stderr)
        return 1
    samples = ds.get(args.split)
    if samples is None:
        print(f'[ERR] split {args.split!r} not found for {args.dataset}. '
              f'available: {sorted(ds.keys())}', file=sys.stderr)
        return 1
    # Newline-separated so callers can `mapfile -t` safely — sample names
    # may contain spaces (e.g. 'Clean cells_T cells', 'lot 1349_D4_D04').
    print('\n'.join(samples))
    return 0


if __name__ == '__main__':
    sys.exit(main())
