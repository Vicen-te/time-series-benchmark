"""Common interface implemented by every forecaster in this benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Minimal contract: fit, predict, and report which hyperparameters were used."""

    name: str = "base"

    @property
    def context_rows(self) -> int:
        """How many rows of history immediately before a frame ``predict`` needs.

        Warm-up taken from inside the evaluated frame silently shortens the
        window a model is scored on, and by a different amount per model. Each
        forecaster declares its requirement here so the caller can hand it the
        preceding split instead.
        """

        return 0

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(
        self,
        df: pd.DataFrame,
        context: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Return ``(y_true, y_pred)`` for the rows of ``df``, aligned on its index.

        ``context`` holds the rows directly preceding ``df``; the last
        ``context_rows`` of it are used as warm-up and never scored.
        """

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Hyperparameters logged to MLflow."""

    def save(self, output_dir: Path) -> Path:
        """Persist the fitted model to ``output_dir`` and return the path.

        Default implementation does nothing useful -- concrete subclasses with
        actual trainable parameters override this to drop weights to disk.
        Zero-shot models can keep the no-op behaviour.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _with_context(
        self,
        df: pd.DataFrame,
        context: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, int]:
        """Prepend the needed tail of ``context`` to ``df``.

        Returns the combined frame and how many warm-up rows were prepended.
        """

        rows = self.context_rows
        if context is None or rows <= 0 or context.empty:
            return df, 0

        tail = context.tail(rows)
        if not tail.index.max() < df.index.min():
            raise ValueError("context must end strictly before df begins.")

        combined = pd.concat([tail[df.columns], df])
        return combined, len(tail)

    @staticmethod
    def align(y_true: pd.Series, y_pred: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        common = y_true.index.intersection(y_pred.index)
        return (
            y_true.loc[common].to_numpy(dtype=np.float64),
            y_pred.loc[common].to_numpy(dtype=np.float64),
        )
