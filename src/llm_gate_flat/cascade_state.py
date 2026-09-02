"""Cascade state — tracks predicted Step columns through a sample's gating walk.

The pristine sample DataFrame holds GT ``Step*`` columns from the parquet.
``synthetic_df()`` returns a copy with feature columns + every Step column the
cascade has predicted so far, deliberately *omitting* the GT Step columns —
parent expressions for downstream steps must resolve against predictions only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CascadeState:
    def __init__(self, df_full: pd.DataFrame, dataset: str, sample: str):
        self._df_full = df_full
        self._n = len(df_full)
        self._predicted_cols: dict[str, np.ndarray] = {}
        self._feature_cols = [c for c in df_full.columns if not c.startswith("Step")]
        self.dataset = dataset
        self.sample = sample

    @property
    def n(self) -> int:
        return self._n

    def feature_values(self, channel: str) -> np.ndarray:
        return self._df_full[channel].values

    def gt_labels(self, annotation_col: str) -> np.ndarray | None:
        if annotation_col in self._df_full.columns:
            return self._df_full[annotation_col].values
        return None

    def set(self, col_name: str, full_length_labels: np.ndarray) -> None:
        if len(full_length_labels) != self._n:
            raise ValueError(
                f"predicted column {col_name} has length {len(full_length_labels)}, "
                f"expected {self._n}"
            )
        self._predicted_cols[col_name] = np.asarray(full_length_labels, dtype=object)

    def predicted_cols(self) -> dict[str, np.ndarray]:
        return dict(self._predicted_cols)

    def synthetic_df(self) -> pd.DataFrame:
        df = self._df_full[self._feature_cols].copy()
        for col, vals in self._predicted_cols.items():
            df[col] = vals
        return df

    def to_predicted_parquet(self) -> pd.DataFrame:
        """DataFrame with predicted Step columns only (length = N_full).

        Cells outside a step's predicted parent are filled with ``Unassigned``.
        """
        if not self._predicted_cols:
            return pd.DataFrame(index=range(self._n))
        return pd.DataFrame(self._predicted_cols)
