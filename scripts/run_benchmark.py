"""End-to-end benchmark: persistence vs LightGBM vs LSTM vs Chronos with MLflow tracking.

Usage
-----
    python scripts/run_benchmark.py                # all four models
    python scripts/run_benchmark.py --models lightgbm lstm
    python scripts/run_benchmark.py --no-chronos   # skip the heavyweight one
    python scripts/run_benchmark.py --device cpu   # pin to CPU (reproducible baselines)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import CONFIG, FIGURES_DIR, RESULTS_DIR
from src.evaluation import metrics_to_row, results_to_dataframe
from src.models import (
    ChronosForecaster,
    LightGBMForecaster,
    LSTMForecaster,
    PersistenceForecaster,
)
from src.plotting import plot_metrics_bar, plot_overlay
from src.significance import p_values_against, pairwise_significance
from src.train import configure_mlflow, load_default_splits, train_and_evaluate


# The row every other row is judged against.
BASELINE_MODEL = "persistence"

MODEL_FACTORIES = {
    "persistence": PersistenceForecaster,
    "lightgbm": LightGBMForecaster,
    "lstm": LSTMForecaster,
    "chronos": ChronosForecaster,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_FACTORIES.keys()),
        default=list(MODEL_FACTORIES.keys()),
        help="Subset of models to run.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=CONFIG.horizon,
        help=(
            "Steps ahead to forecast, in hours. The default of 1 is the easiest "
            "horizon and the least informative one: hourly oil temperature is "
            "close to a random walk, so persistence is nearly optimal there. "
            "Runs with a horizon other than 1 write under results/tables/h<N>/ "
            "so the canonical tables are never overwritten."
        ),
    )
    parser.add_argument("--no-chronos", action="store_true", help="Skip the Chronos run.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Where the torch-backed models run. Pin to cpu for numbers that are "
            "reproducible on a CI runner; CUDA_VISIBLE_DEVICES is not reliable "
            "for this on every platform."
        ),
    )
    return parser.parse_args()


def build_model(name: str, device: str, horizon: int):
    """Instantiate ``name`` at ``horizon``, pinning the device where there is one."""

    if name == "lstm":
        return MODEL_FACTORIES[name](
            horizon=horizon, device=None if device == "auto" else device
        )
    if name == "chronos":
        return MODEL_FACTORIES[name](
            config=replace(CONFIG.chronos, device=device), horizon=horizon
        )
    return MODEL_FACTORIES[name](horizon=horizon)


def main() -> int:
    args = parse_args()
    horizon: int = args.horizon
    if horizon < 1:
        raise SystemExit("--horizon must be >= 1.")
    selected: List[str] = list(args.models)
    if args.no_chronos and "chronos" in selected:
        selected.remove("chronos")

    if not args.no_mlflow:
        configure_mlflow()

    # Horizon 1 owns the canonical artefact paths that the README, the figures
    # and the CI gate read; every other horizon writes beside them, never over.
    tables_dir = RESULTS_DIR / "tables"
    figures_dir = FIGURES_DIR
    if horizon != 1:
        tables_dir = tables_dir / f"h{horizon}"
        figures_dir = figures_dir / f"h{horizon}"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    splits = load_default_splits()
    print("Temporal split:")
    print(splits.describe())
    print()

    rows: List[Dict[str, float]] = []
    predictions: Dict[str, pd.Series] = {}
    y_test_truth: pd.Series | None = None

    for name in selected:
        print(f"=== Training {name} ===")
        model = build_model(name, args.device, horizon)
        result = train_and_evaluate(
            model,
            splits,
            run_name=name if horizon == 1 else f"{name}-h{horizon}",
            artifacts_dir=figures_dir,
            log_to_mlflow=not args.no_mlflow,
        )
        print(
            f"  fit  : {result.fit_seconds:8.2f}s | "
            f"predict: {result.predict_seconds:7.2f}s | "
            f"test RMSE = {result.test_metrics.rmse:.4f}, "
            f"MAE = {result.test_metrics.mae:.4f}, "
            f"R2 = {result.test_metrics.r2:.4f}"
        )

        row = metrics_to_row(name, result.test_metrics)
        row["fit_seconds"] = round(result.fit_seconds, 3)
        row["predict_seconds"] = round(result.predict_seconds, 3)
        rows.append(row)

        predictions[name] = result.y_test_pred
        if y_test_truth is None or len(result.y_test_true) > len(y_test_truth):
            y_test_truth = result.y_test_true

    summary = results_to_dataframe(rows)

    # A ranking to four decimals says nothing about whether the gaps are real.
    pairs = pd.DataFrame()
    if y_test_truth is not None and len(predictions) > 1:
        pairs = pairwise_significance(y_test_truth, predictions)
        pairs.to_csv(tables_dir / "significance.csv", index=False)
        p_map = p_values_against(BASELINE_MODEL, y_test_truth, predictions)
        if p_map:
            summary[f"p_vs_{BASELINE_MODEL}"] = summary["model"].map(p_map)

    print("\n=== Test-set summary ===")
    print(summary.to_string(index=False))

    if not pairs.empty:
        print("\n=== Diebold-Mariano, pairwise ===")
        print(pairs.drop(columns=["lags"]).to_string(index=False))

    summary_path = tables_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary_md = tables_dir / "benchmark_summary.md"
    summary_md.write_text(summary.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {summary_md}")
    if not pairs.empty:
        print(f"Saved: {tables_dir / 'significance.csv'}")

    if y_test_truth is not None and len(predictions) > 1:
        common_idx = y_test_truth.index
        for s in predictions.values():
            common_idx = common_idx.intersection(s.index)
        aligned_truth = y_test_truth.loc[common_idx]
        aligned_preds = {k: v.loc[common_idx] for k, v in predictions.items()}

        metrics_by_model = {
            row["model"]: {k: v for k, v in row.items() if k != "model"} for row in rows
        }
        plot_metrics_bar(metrics_by_model, output_path=figures_dir / "metrics_comparison.png")
        plot_overlay(
            aligned_truth,
            aligned_preds,
            index=aligned_truth.index,
            output_path=figures_dir / "predictions_overlay.png",
        )
        print(f"Saved comparison figures to {figures_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
