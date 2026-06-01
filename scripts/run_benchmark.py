"""End-to-end benchmark: LightGBM vs LSTM vs Chronos with MLflow tracking.

Usage
-----
    python scripts/run_benchmark.py                # all three models
    python scripts/run_benchmark.py --models lightgbm lstm
    python scripts/run_benchmark.py --no-chronos   # skip the heavyweight one
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import RESULTS_DIR
from src.evaluation import metrics_to_row, results_to_dataframe
from src.models import ChronosForecaster, LightGBMForecaster, LSTMForecaster
from src.plotting import plot_metrics_bar, plot_overlay
from src.train import configure_mlflow, load_default_splits, train_and_evaluate


MODEL_FACTORIES = {
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
    parser.add_argument("--no-chronos", action="store_true", help="Skip the Chronos run.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected: List[str] = list(args.models)
    if args.no_chronos and "chronos" in selected:
        selected.remove("chronos")

    if not args.no_mlflow:
        configure_mlflow()

    splits = load_default_splits()
    print("Temporal split:")
    print(splits.describe())
    print()

    rows: List[Dict[str, float]] = []
    predictions: Dict[str, pd.Series] = {}
    y_test_truth: pd.Series | None = None

    for name in selected:
        print(f"=== Training {name} ===")
        model = MODEL_FACTORIES[name]()
        result = train_and_evaluate(
            model,
            splits,
            run_name=name,
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
    print("\n=== Test-set summary ===")
    print(summary.to_string(index=False))

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary_md = tables_dir / "benchmark_summary.md"
    summary_md.write_text(summary.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {summary_md}")

    if y_test_truth is not None and len(predictions) > 1:
        common_idx = y_test_truth.index
        for s in predictions.values():
            common_idx = common_idx.intersection(s.index)
        aligned_truth = y_test_truth.loc[common_idx]
        aligned_preds = {k: v.loc[common_idx] for k, v in predictions.items()}

        metrics_by_model = {
            row["model"]: {k: v for k, v in row.items() if k != "model"} for row in rows
        }
        plot_metrics_bar(metrics_by_model)
        plot_overlay(aligned_truth, aligned_preds, index=aligned_truth.index)
        print("Saved comparison figures to results/figures/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
