"""Parquet → cyMAE tensor adapter.

Stages that consume this module:
    pretrain.py      -- concat all-train-cell tensor (capped per sample) for MAE.
    train_heads.py   -- normalizes parent-pop cells from each train task, feeds frozen encoder.
    inference.py     -- normalizes parent-pop cells from each test task, feeds frozen encoder.

Channel selection, deterministic marker order, and per-channel z-score statistics are
fitted *once* on the cohort's train split (in pretrain.py) and persisted alongside
the encoder. Every later stage MUST re-load and reuse the same artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

INCLUDED_KINDS: tuple[str, ...] = (
    "protein",
    "livedead",
    "scatter",
    "dna",
    "bead",
    "gaussian",
)


def select_channels(meta_channels: dict, kinds: Iterable[str] = INCLUDED_KINDS) -> list[str]:
    """Pick channels from meta.json["channels"] in deterministic kind-then-name order.

    Tier order matches `kinds`. Within a tier, channel keys are sorted alphabetically.
    Channels with kinds not in the include list (typically `tech`, `background`)
    are dropped — `tech` is uncorrelated with biology, `background` is empty noise.
    """
    kinds_tuple = tuple(kinds)
    by_kind: dict[str, list[str]] = {k: [] for k in kinds_tuple}
    for ch, info in meta_channels.items():
        kind = info[1] if len(info) > 1 else "unknown"
        if kind in by_kind:
            by_kind[kind].append(ch)
    ordered: list[str] = []
    for k in kinds_tuple:
        ordered.extend(sorted(by_kind[k]))
    return ordered


def fit_zscore(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std fitted on `arr` (N, M). σ floored at 1e-6 to avoid divide-by-zero
    on constant channels."""
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr - mean) / std).astype(np.float32)


def save_channel_artifacts(
    out_dir: Path,
    cohort: str,
    channel_order: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    channel_marker_map: dict,
    marker_kinds: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "marker_order.json").write_text(
        json.dumps(
            {
                "cohort": cohort,
                "channels": channel_order,
                "display_names": [channel_marker_map.get(c, c) for c in channel_order],
                "kinds": [marker_kinds.get(c, "unknown") for c in channel_order],
            },
            indent=2,
        )
    )
    np.savez(out_dir / "channel_stats.npz", mean=mean, std=std)


def load_channel_artifacts(in_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    order = json.loads((in_dir / "marker_order.json").read_text())["channels"]
    stats = np.load(in_dir / "channel_stats.npz")
    return order, stats["mean"].astype(np.float32), stats["std"].astype(np.float32)


def _read_parquet_cells(
    parquet_path: Path,
    channels: list[str],
    max_cells: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=channels)
    arr = df[channels].to_numpy(dtype=np.float32, na_value=0.0)
    if max_cells is not None and len(arr) > max_cells:
        idx = rng.choice(len(arr), size=max_cells, replace=False)
        arr = arr[idx]
    return arr


def load_concatenated_cells(
    data_dir: Path,
    cohort: str,
    samples: list[str],
    channels: list[str],
    max_cells_per_sample: int | None = 10_000,
    seed: int = 0,
) -> np.ndarray:
    """Concatenate cells across `samples`, channel-ordered, optionally capped per sample.

    Used for pretrain (label-free): returns a (N_total, M) float32 array.
    Same `seed` => identical subsample across re-runs.
    """
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for sample in samples:
        path = data_dir / cohort / "parquet" / f"{sample}.parquet"
        if not path.exists():
            print(f"[WARN] missing parquet, skipping: {path}")
            continue
        parts.append(_read_parquet_cells(path, channels, max_cells_per_sample, rng))
    if not parts:
        raise RuntimeError(f"No parquets found for cohort={cohort}, samples={samples}")
    return np.concatenate(parts, axis=0)


def subsample_array(arr: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    if len(arr) <= max_n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arr), size=max_n, replace=False)
    return arr[idx]


def cohort_train_samples(
    splits_path: Path, cohort: str, split_set: str = "train"
) -> list[str]:
    """Read splits.json and return cohort sample list for the given split,
    excluding blacklist."""
    splits = json.loads(splits_path.read_text())
    info = splits.get("in_panel", {}).get(cohort, {})
    samples = list(info.get(split_set, []))
    blacklist = set(info.get("blacklist", []))
    return [s for s in samples if s not in blacklist]
