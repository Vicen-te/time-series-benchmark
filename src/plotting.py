"""Plot helpers used by every model and the benchmark script."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # safe on headless CI machines

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR


def _ensure_index(values, index):
    if index is None:
        return pd.RangeIndex(len(values))
    if isinstance(index, pd.Index):
        return index
    return pd.Index(index)


def plot_predictions(
    y_true,
    y_pred,
    model_name: str,
    index=None,
    output_path: Optional[Path] = None,
    n_points: int = 500,
) -> Path:
    """Line plot of the last ``n_points`` predictions against the truth."""

    index = _ensure_index(y_true, index)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    n_points = min(n_points, len(y_true))
    sl = slice(len(y_true) - n_points, len(y_true))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(index[sl], y_true[sl], label="actual", linewidth=1.4, color="#222")
    ax.plot(index[sl], y_pred[sl], label="predicted", linewidth=1.2, color="#d6336c", alpha=0.85)
    ax.set_title(f"{model_name} - predictions vs actual (last {n_points} points)")
    ax.set_ylabel("OT (oil temperature)")
    ax.set_xlabel("date")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()

    output_path = output_path or FIGURES_DIR / f"predictions_{model_name.lower()}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def plot_scatter(
    y_true,
    y_pred,
    model_name: str,
    output_path: Optional[Path] = None,
) -> Path:
    """Predicted vs actual scatter, with the y=x reference line."""

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.35, color="#1c7ed6")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#212529", linestyle="--", linewidth=1)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(f"{model_name} - predicted vs actual")
    ax.grid(alpha=0.25)

    output_path = output_path or FIGURES_DIR / f"scatter_{model_name.lower()}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def plot_residuals(
    y_true,
    y_pred,
    model_name: str,
    output_path: Optional[Path] = None,
) -> Path:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=60, color="#495057", alpha=0.85)
    ax.axvline(0, color="#e03131", linestyle="--", linewidth=1)
    ax.set_title(f"{model_name} - residuals (actual - predicted)")
    ax.set_xlabel("residual")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)

    output_path = output_path or FIGURES_DIR / f"residuals_{model_name.lower()}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def plot_metrics_bar(
    metrics_by_model: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
) -> Path:
    """One bar chart per metric (MAE, RMSE, R2) with one bar per model."""

    metric_keys = ("mae", "rmse", "r2")
    models = list(metrics_by_model.keys())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    palette = ["#1c7ed6", "#37b24d", "#f08c00", "#d6336c", "#7048e8"]

    for ax, metric in zip(axes, metric_keys):
        values = [metrics_by_model[m].get(metric, np.nan) for m in models]
        colors = [palette[i % len(palette)] for i in range(len(models))]
        bars = ax.bar(models, values, color=colors)
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Test-set metrics by model", fontsize=12)
    output_path = output_path or FIGURES_DIR / "metrics_comparison.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def plot_overlay(
    y_true,
    predictions: Dict[str, Iterable[float]],
    index=None,
    output_path: Optional[Path] = None,
    n_points: int = 400,
) -> Path:
    """Overlay every model's predictions on the truth on the same axes."""

    index = _ensure_index(y_true, index)
    y_true = np.asarray(y_true).ravel()
    n_points = min(n_points, len(y_true))
    sl = slice(len(y_true) - n_points, len(y_true))

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(index[sl], y_true[sl], label="actual", color="#212529", linewidth=1.6)
    palette = ["#1c7ed6", "#37b24d", "#f08c00", "#d6336c", "#7048e8"]
    for i, (name, preds) in enumerate(predictions.items()):
        preds = np.asarray(preds).ravel()
        ax.plot(index[sl], preds[sl], label=name, color=palette[i % len(palette)], alpha=0.85, linewidth=1.1)

    ax.set_title(f"All models - predictions vs actual (last {n_points} points)")
    ax.set_ylabel("OT")
    ax.set_xlabel("date")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()

    output_path = output_path or FIGURES_DIR / "predictions_overlay.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path
