"""Pool prompts across cohorts and shard them evenly for data-parallel
vLLM inference.

Map-reduce layout:
  map    : ``build_pool`` flattens (sample, step) groups from N cohorts,
           tagging each group with its ``dataset`` name.
  shard  : ``shard_pool`` partitions the pool deterministically across
           ``n_shards`` workers. ``balance='lpt'`` (Longest Processing
           Time first) packs longest prompts into the least-loaded shard
           — keeps GPU wall-clock close to even when cohort sizes differ.
  reduce : a separate merger (``src.llm.scripts.merge_pool_partials``)
           regroups per-shard partial outputs back into one
           pred.json per cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.llm.inference.data_loader import load_c2s


def _resolve_split_filter(
    split: str | None,
    splits_path: str | Path,
    datasets: list[str],
    group: str = 'in_panel',
) -> dict[str, set[str]] | None:
    """Read splits.json → {dataset: set(sample_names)} for the given split.

    Returns ``None`` when ``split`` is empty (no filtering). Raises if
    a requested dataset is missing from ``splits.json`` — surfacing the
    mismatch loudly is safer than silently running on all samples.
    """
    if not split:
        return None
    data = json.loads(Path(splits_path).read_text())
    panel = data.get(group, {})
    out: dict[str, set[str]] = {}
    for ds in datasets:
        ds_entry = panel.get(ds)
        if ds_entry is None:
            raise KeyError(f'splits.json missing {group}/{ds}')
        names = ds_entry.get(split)
        if names is None:
            raise KeyError(
                f'splits.json {group}/{ds} has no split {split!r} '
                f'(available: {sorted(ds_entry.keys())})'
            )
        out[ds] = set(names)
    return out


def build_pool(
    benchmark: str,
    datasets: list[str],
    ablation_slug: str,
    split: str | None = None,
    splits_path: str | Path = 'benchmark/splits.json',
    perturbation_name: str | None = None,
) -> list[dict]:
    """Build a flat pool of (sample, step) groups across multiple cohorts.

    Each group dict gets a ``dataset`` field added. Order is deterministic:
    cohorts in input order, then load_c2s's intrinsic ordering (sample
    name asc, step asc). Two workers loading the same args will produce
    identical pools — required for shard agreement.
    """
    sample_filters = _resolve_split_filter(split, splits_path, datasets)
    pool: list[dict] = []
    for ds in datasets:
        ds_path = f"{str(benchmark).rstrip('/')}/{ds}"
        ds_groups = load_c2s(
            benchmark_dir=ds_path,
            ablation_slug=ablation_slug,
            sample_filter=(sample_filters or {}).get(ds),
            perturbation_name=perturbation_name,
        )
        for g in ds_groups:
            g['dataset'] = ds
            g['dataset_path'] = ds_path
        pool.extend(ds_groups)
    return pool


def shard_pool(
    groups: list[dict],
    shard_idx: int,
    n_shards: int,
    balance: str = 'lpt',
) -> tuple[list[dict], list[int]]:
    """Return (groups_for_this_shard, original_indices_into_pool).

    ``balance``:
      - ``'lpt'`` — Longest Processing Time first by ``len(prompt_text)``.
        Char length is a proxy for compute; with thinking enabled, output
        tokens dominate so this is approximate but typically within ~10%
        of optimal across reasonable cohort mixes.
      - ``'rr'``  — round-robin (count-balanced, ignores prompt length).
    """
    if not 0 <= shard_idx < n_shards:
        raise ValueError(f'shard_idx={shard_idx} out of [0,{n_shards})')
    n = len(groups)
    if n_shards == 1:
        return list(groups), list(range(n))

    if balance == 'rr':
        idxs = list(range(shard_idx, n, n_shards))
        return [groups[i] for i in idxs], idxs

    if balance != 'lpt':
        raise ValueError(f'unknown balance mode: {balance!r}')

    # LPT: sort by (-length, orig_idx) for deterministic tie-break, then
    # greedy-assign each group to the currently-least-loaded shard.
    items = sorted(
        enumerate(groups),
        key=lambda kv: (-len(kv[1].get('prompt_text', '')), kv[0]),
    )
    bins: list[list[int]] = [[] for _ in range(n_shards)]
    loads = [0] * n_shards
    for orig_i, g in items:
        target = min(range(n_shards), key=lambda k: (loads[k], k))
        bins[target].append(orig_i)
        loads[target] += len(g['prompt_text'])

    shard_idxs = sorted(bins[shard_idx])
    return [groups[i] for i in shard_idxs], shard_idxs


def shard_summary(groups: list[dict], n_shards: int, balance: str = 'lpt') -> str:
    """Human-readable summary of how a pool would shard. Useful in launcher
    logs to verify balance before kicking off N workers.
    """
    lines = [f'pool: {len(groups):,} groups across {n_shards} shards (balance={balance})']
    for k in range(n_shards):
        sub, _ = shard_pool(groups, k, n_shards, balance=balance)
        chars = sum(len(g.get('prompt_text', '')) for g in sub)
        per_ds: dict[str, int] = {}
        for g in sub:
            per_ds[g['dataset']] = per_ds.get(g['dataset'], 0) + 1
        ds_str = ', '.join(f'{d}:{n}' for d, n in sorted(per_ds.items()))
        lines.append(
            f'  shard {k}/{n_shards}: {len(sub):>5} groups  {chars:>12,} chars  [{ds_str}]'
        )
    return '\n'.join(lines)
