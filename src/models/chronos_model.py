"""Zero-shot Chronos foundation-model forecaster.

Chronos (Amazon, 2024) is a family of pre-trained foundation models for time
series. We do **no training** here — we just run inference for each test point
using a rolling context window of past observations.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..config import CONFIG, TARGET_COLUMN, ChronosConfig
from .base import BaseForecaster


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class ChronosForecaster(BaseForecaster):
    name = "chronos"

    def __init__(
        self,
        target: str = TARGET_COLUMN,
        horizon: int = 1,
        config: Optional[ChronosConfig] = None,
    ) -> None:
        self.target = target
        self.horizon = horizon
        self.config = config or CONFIG.chronos
        self.device = _resolve_device(self.config.device)
        self.pipeline_ = None

    @property
    def context_rows(self) -> int:
        return self.config.context_length

    def _load_pipeline(self):
        if self.pipeline_ is not None:
            return self.pipeline_
        from pathlib import Path as _Path
        from chronos import BaseChronosPipeline
        from .. import config as _cfg

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Prefer the local snapshot to avoid re-downloading and to dodge SSL
        # issues with strict corporate networks.
        local_path = _Path(_cfg.PROJECT_ROOT) / self.config.local_model_path
        source = str(local_path) if local_path.exists() else self.config.model_name

        self.pipeline_ = BaseChronosPipeline.from_pretrained(
            source,
            device_map=self.device,
            dtype=dtype,
        )
        return self.pipeline_

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **_: Any,
    ) -> "ChronosForecaster":
        """No training -- we only make sure the weights are resident."""

        if self.target not in train_df.columns:
            raise KeyError(f"Target column '{self.target}' not found in DataFrame.")
        self._load_pipeline()
        return self

    def predict(
        self,
        df: pd.DataFrame,
        context: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        if self.pipeline_ is None:
            raise RuntimeError("Call .fit() before .predict().")

        target_values = df[self.target].to_numpy(dtype=np.float32)
        target_index = df.index

        full, history_offset = self._with_context(df, context)
        full_series = full[self.target].to_numpy(dtype=np.float32)
        context_length = self.config.context_length
        horizon = self.horizon

        contexts: list[torch.Tensor] = []
        ground_truth: list[float] = []
        truth_index: list[pd.Timestamp] = []

        for i in range(len(target_values)):
            end = history_offset + i
            start = max(0, end - context_length)
            window = full_series[start:end]
            if len(window) < 8:
                continue
            contexts.append(torch.tensor(window, dtype=torch.float32))
            ground_truth.append(float(target_values[i]))
            truth_index.append(target_index[i])

        if not contexts:
            empty = pd.Series([], dtype=np.float64)
            return empty, empty

        predictions: list[float] = []
        batch_size = self.config.batch_size
        iterator = range(0, len(contexts), batch_size)
        for start in tqdm(iterator, desc="chronos", leave=False):
            batch = contexts[start : start + batch_size]
            forecasts = self.pipeline_.predict(batch, prediction_length=horizon)
            # ``forecasts`` shape varies by model: bolt returns
            # (batch, quantiles, horizon); base returns (batch, samples, horizon).
            # We collapse the middle axis with a median which is the standard
            # point estimate for either family.
            forecast_np = forecasts.float().cpu().numpy()
            if forecast_np.ndim == 3:
                point = np.median(forecast_np, axis=1)[:, -1]
            elif forecast_np.ndim == 2:
                point = forecast_np[:, -1]
            else:
                raise RuntimeError(f"Unexpected forecast shape: {forecast_np.shape}")
            predictions.extend(point.tolist())

        idx = pd.Index(truth_index)
        y_true = pd.Series(ground_truth, index=idx, name="y_true")
        y_pred = pd.Series(predictions, index=idx, name="y_pred")
        return y_true, y_pred

    def get_params(self) -> Dict[str, Any]:
        params = asdict(self.config)
        params.update(
            {
                "target": self.target,
                "horizon": self.horizon,
                "device": self.device,
                "trained": False,
            }
        )
        return params

    def save(self, output_dir: Path) -> Path:
        """Chronos is zero-shot; we record which checkpoint was used.

        We deliberately don't copy the 183 MB weights into every MLflow run —
        the manifest points to the cached snapshot so anyone reproducing the
        run can recover the exact model.
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_name": self.config.model_name,
            "local_model_path": self.config.local_model_path,
            "context_length": self.config.context_length,
            "horizon": self.horizon,
            "device": self.device,
            "trained": False,
            "note": (
                "Zero-shot foundation model. Weights are NOT bundled with the run. "
                "Re-load with chronos.BaseChronosPipeline.from_pretrained(model_name)."
            ),
        }
        (output_dir / "chronos_manifest.json").write_text(json.dumps(manifest, indent=2))
        return output_dir
