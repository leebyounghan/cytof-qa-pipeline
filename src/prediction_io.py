"""Read per-step prediction files, written as ``prediction.json.gz`` or the legacy uncompressed ``prediction.json``."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

PREDICTION_NAMES = ("prediction.json.gz", "prediction.json")


def prediction_path(step_dir: Path) -> Path | None:
    """Return the prediction file in ``step_dir``, preferring the compressed one, or None."""
    for name in PREDICTION_NAMES:
        path = Path(step_dir) / name
        if path.exists():
            return path
    return None


def has_prediction(step_dir: Path) -> bool:
    return prediction_path(step_dir) is not None


def read_prediction(step_dir: Path) -> dict:
    path = prediction_path(step_dir)
    if path is None:
        raise FileNotFoundError(f"No prediction.json.gz or prediction.json in {step_dir}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)
