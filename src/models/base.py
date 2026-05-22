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

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Return ``(y_true, y_pred)`` aligned on the same DatetimeIndex."""

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Hyperparameters logged to MLflow."""

    def save(self, output_dir: Path) -> Path:
        """Persist the fitted model to ``output_dir`` and return the path.

        Default implementation does nothing useful — concrete subclasses with
        actual trainable parameters override this to drop weights to disk.
        Zero-shot models can keep the no-op behaviour.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def align(y_true: pd.Series, y_pred: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        common = y_true.index.intersection(y_pred.index)
        return (
            y_true.loc[common].to_numpy(dtype=np.float64),
            y_pred.loc[common].to_numpy(dtype=np.float64),
        )
