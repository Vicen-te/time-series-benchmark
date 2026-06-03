"""High-level training/evaluation orchestration with MLflow tracking."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import pandas as pd

from .config import CONFIG, MLFLOW_EXPERIMENT, MLRUNS_DIR, RESULTS_DIR
from .data import TimeSeriesSplit, load_etth1, temporal_split
from .evaluation import Metrics, compute_metrics
from .models.base import BaseForecaster
from .plotting import plot_predictions, plot_residuals, plot_scatter


def configure_mlflow(experiment: str = MLFLOW_EXPERIMENT) -> None:
    """Point MLflow to ./mlruns/ and create the experiment if needed."""

    # MLflow 3.x deprecates file-based tracking stores and raises by default;
    # opt in explicitly so the workflow runs out-of-the-box without a DB backend.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(experiment)


def _log_metrics_block(prefix: str, metrics: Metrics) -> None:
    for key, value in metrics.to_dict().items():
        if key == "n":
            mlflow.log_metric(f"{prefix}_{key}", value)
            continue
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            continue
        mlflow.log_metric(f"{prefix}_{key}", float(value))


@dataclass
class TrainResult:
    model_name: str
    train_metrics: Metrics
    val_metrics: Optional[Metrics]
    test_metrics: Metrics
    y_test_true: pd.Series
    y_test_pred: pd.Series
    fit_seconds: float
    predict_seconds: float
    extra: Dict[str, Any]


def train_and_evaluate(
    model: BaseForecaster,
    splits: TimeSeriesSplit,
    run_name: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
    log_to_mlflow: bool = True,
) -> TrainResult:
    """Fit ``model`` on the train split and evaluate on val + test."""

    run_name = run_name or model.name
    artifacts_dir = artifacts_dir or RESULTS_DIR / "figures"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    fit_start = time.perf_counter()
    model.fit(splits.train, splits.val)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_train_true, y_train_pred = model.predict(splits.train)
    y_val_true, y_val_pred = (
        model.predict(splits.val) if not splits.val.empty else (pd.Series(dtype=float), pd.Series(dtype=float))
    )
    y_test_true, y_test_pred = model.predict(splits.test)
    predict_seconds = time.perf_counter() - predict_start

    train_metrics = compute_metrics(y_train_true, y_train_pred)
    val_metrics = compute_metrics(y_val_true, y_val_pred) if len(y_val_true) else None
    test_metrics = compute_metrics(y_test_true, y_test_pred)

    pred_plot = plot_predictions(
        y_test_true, y_test_pred, model_name=model.name,
        index=y_test_true.index,
        output_path=artifacts_dir / f"predictions_{model.name}.png",
    )
    scatter_plot = plot_scatter(
        y_test_true, y_test_pred, model_name=model.name,
        output_path=artifacts_dir / f"scatter_{model.name}.png",
    )
    residual_plot = plot_residuals(
        y_test_true, y_test_pred, model_name=model.name,
        output_path=artifacts_dir / f"residuals_{model.name}.png",
    )

    extra: Dict[str, Any] = {
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_test": len(splits.test),
    }

    if log_to_mlflow:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(_jsonable(model.get_params()))
            mlflow.log_param("split_train_n", len(splits.train))
            mlflow.log_param("split_val_n", len(splits.val))
            mlflow.log_param("split_test_n", len(splits.test))

            _log_metrics_block("train", train_metrics)
            if val_metrics is not None:
                _log_metrics_block("val", val_metrics)
            _log_metrics_block("test", test_metrics)
            mlflow.log_metric("fit_seconds", fit_seconds)
            mlflow.log_metric("predict_seconds", predict_seconds)

            mlflow.log_artifact(str(pred_plot), artifact_path="figures")
            mlflow.log_artifact(str(scatter_plot), artifact_path="figures")
            mlflow.log_artifact(str(residual_plot), artifact_path="figures")

            preds_path = artifacts_dir.parent / "tables" / f"predictions_{model.name}.csv"
            preds_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"y_true": y_test_true.values, "y_pred": y_test_pred.values},
                index=y_test_true.index,
            ).to_csv(preds_path)
            mlflow.log_artifact(str(preds_path), artifact_path="predictions")

            if hasattr(model, "feature_importance"):
                try:
                    importance_df = model.feature_importance()
                    imp_path = artifacts_dir.parent / "tables" / f"feature_importance_{model.name}.csv"
                    importance_df.to_csv(imp_path, index=False)
                    mlflow.log_artifact(str(imp_path), artifact_path="feature_importance")
                except Exception:  # pragma: no cover - best-effort logging
                    pass

            # Log the FITTED model itself (booster / state_dict / manifest).
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    saved_dir = model.save(Path(tmp_dir))
                    for f in Path(saved_dir).rglob("*"):
                        if f.is_file():
                            mlflow.log_artifact(str(f), artifact_path="model")
            except Exception as exc:  # pragma: no cover - best-effort logging
                mlflow.set_tag("model_save_error", str(exc)[:200])

    return TrainResult(
        model_name=model.name,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        y_test_true=y_test_true,
        y_test_pred=y_test_pred,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        extra=extra,
    )


def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    """MLflow params must be primitives -> serialise tuples/lists/etc to strings."""

    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v)
            except TypeError:
                out[k] = str(v)
    return out


def load_default_splits() -> TimeSeriesSplit:
    df = load_etth1()
    return temporal_split(df)
