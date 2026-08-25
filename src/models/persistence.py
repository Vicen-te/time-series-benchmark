"""Naive persistence forecaster: repeat the last observed value.

This is the reference a forecaster has to beat before any of its other
qualities matter. On a strongly autocorrelated hourly series it is a
surprisingly strong opponent, and quoting model errors without it leaves the
reader no way to tell skill from inertia.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..config import TARGET_COLUMN
from .base import BaseForecaster


class PersistenceForecaster(BaseForecaster):
    """Predicts ``target[t]`` as ``target[t - horizon]``."""

    name = "persistence"

    def __init__(self, target: str = TARGET_COLUMN, horizon: int = 1) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1.")
        self.target = target
        self.horizon = horizon

    @property
    def context_rows(self) -> int:
        return self.horizon

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **_: Any,
    ) -> "PersistenceForecaster":
        """Nothing to learn -- the rule is fixed."""

        if self.target not in train_df.columns:
            raise KeyError(f"Target column '{self.target}' not found in DataFrame.")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        context: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        full, _ = self._with_context(df, context)
        y_pred = full[self.target].astype(float).shift(self.horizon).reindex(df.index)
        y_true = df[self.target].astype(float)
        valid = y_pred.notna()
        return (
            y_true.loc[valid].rename("y_true"),
            y_pred.loc[valid].rename("y_pred"),
        )

    def get_params(self) -> Dict[str, Any]:
        return {"target": self.target, "horizon": self.horizon, "trained": False}

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata.json").write_text(
            json.dumps({"target": self.target, "horizon": self.horizon}, indent=2)
        )
        return output_dir
