"""LightGBM forecaster with temporal feature engineering."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import CONFIG, TARGET_COLUMN, LightGBMConfig
from ..features import build_feature_matrix
from .base import BaseForecaster


class LightGBMForecaster(BaseForecaster):
    """Direct h-step forecaster: predicts y[t+horizon] from past lags & rollings."""

    name = "lightgbm"

    def __init__(
        self,
        target: str = TARGET_COLUMN,
        horizon: int = 1,
        config: Optional[LightGBMConfig] = None,
    ) -> None:
        self.target = target
        self.horizon = horizon
        self.config = config or CONFIG.lightgbm
        self.model_: Optional[lgb.LGBMRegressor] = None
        self.feature_cols_: list[str] = []
        self.best_iteration_: Optional[int] = None

    @property
    def context_rows(self) -> int:
        cfg = CONFIG.features
        deepest = max(max(cfg.lag_hours), max(cfg.rolling_windows))
        return self.horizon - 1 + deepest

    def _prepare(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
        return build_feature_matrix(df, target=self.target, horizon=self.horizon)

    def _prepare_scored(
        self,
        df: pd.DataFrame,
        context: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Feature matrix for ``df`` alone, warmed up from ``context``."""

        full, _ = self._with_context(df, context)
        X, y, _ = self._prepare(full)
        scored = X.index.isin(df.index)
        return X.loc[scored], y.loc[scored]

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **_: Any,
    ) -> "LightGBMForecaster":
        X_train, y_train, feature_cols = self._prepare(train_df)
        self.feature_cols_ = feature_cols

        params = asdict(self.config)
        early_stopping = params.pop("early_stopping_rounds")
        random_state = params.pop("random_state")

        self.model_ = lgb.LGBMRegressor(
            random_state=random_state,
            objective="regression",
            metric="rmse",
            verbosity=-1,
            **params,
        )

        callbacks = [lgb.log_evaluation(period=0)]
        eval_set = None
        if val_df is not None and not val_df.empty:
            X_val, y_val = self._prepare_scored(val_df, train_df)
            eval_set = [(X_val[self.feature_cols_], y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping, verbose=False))

        self.model_.fit(
            X_train[self.feature_cols_],
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        return self

    def predict(
        self,
        df: pd.DataFrame,
        context: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        if self.model_ is None:
            raise RuntimeError("Call .fit() before .predict().")
        X, y = self._prepare_scored(df, context)
        preds = self.model_.predict(X[self.feature_cols_])
        return y.rename("y_true"), pd.Series(preds, index=y.index, name="y_pred")

    def feature_importance(self, top_k: int = 25) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Model not trained yet.")
        booster = self.model_.booster_
        gains = booster.feature_importance(importance_type="gain")
        splits = booster.feature_importance(importance_type="split")
        names = booster.feature_name()
        return (
            pd.DataFrame({"feature": names, "gain": gains, "split": splits})
            .sort_values("gain", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )

    def get_params(self) -> Dict[str, Any]:
        params = asdict(self.config)
        params.update({"target": self.target, "horizon": self.horizon})
        if self.best_iteration_ is not None:
            params["best_iteration"] = self.best_iteration_
        return params

    def save(self, output_dir: Path) -> Path:
        if self.model_ is None:
            raise RuntimeError("Nothing to save — call .fit() first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Native LightGBM text format — portable across languages/versions.
        booster_path = output_dir / "lightgbm_booster.txt"
        self.model_.booster_.save_model(str(booster_path))

        # Sklearn wrapper for Python-side reuse via joblib/pickle.
        pickle_path = output_dir / "lightgbm_sklearn.pkl"
        with pickle_path.open("wb") as f:
            pickle.dump(self.model_, f)

        meta = {
            "target": self.target,
            "horizon": self.horizon,
            "feature_cols": self.feature_cols_,
            "best_iteration": self.best_iteration_,
            "config": asdict(self.config),
        }
        (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return output_dir
