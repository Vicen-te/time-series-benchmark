"""Render an MLflow-style runs-comparison image from the local tracking store.

We can't capture a UI screenshot from a headless script, so we read the same
data shown by ``mlflow ui`` and rasterise it as a clean PNG suitable for the
README. The data is real — it comes from ``./mlruns``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.config import FIGURES_DIR, MLFLOW_EXPERIMENT, MLRUNS_DIR


def main() -> int:
    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        print(f"Experiment '{MLFLOW_EXPERIMENT}' not found. Run the benchmark first.")
        return 1

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_rmse ASC"],
    )
    if runs.empty:
        print("No runs to render yet.")
        return 1

    display_cols = {
        "tags.mlflow.runName": "run",
        "metrics.test_rmse": "test_RMSE",
        "metrics.test_mae": "test_MAE",
        "metrics.test_r2": "test_R2",
        "metrics.test_mape": "test_MAPE",
        "metrics.fit_seconds": "fit_s",
        "metrics.predict_seconds": "predict_s",
        "params.horizon": "horizon",
    }
    cols_in_df = [c for c in display_cols.keys() if c in runs.columns]
    table = runs[cols_in_df].rename(columns=display_cols).reset_index(drop=True)
    for c in ("test_RMSE", "test_MAE", "test_R2", "test_MAPE", "fit_s", "predict_s"):
        if c in table.columns:
            table[c] = pd.to_numeric(table[c], errors="coerce").round(4)

    fig, ax = plt.subplots(figsize=(11, 1.0 + 0.55 * len(table)))
    ax.set_axis_off()
    title = (
        f"MLflow experiment: '{MLFLOW_EXPERIMENT}'  (rows={len(table)})\n"
        f"tracking_uri = {MLRUNS_DIR.resolve().as_uri()}"
    )
    ax.set_title(title, loc="left", fontsize=11, color="#212529", pad=14)

    tbl = ax.table(
        cellText=table.astype(str).values,
        colLabels=table.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)

    # Header band styling.
    for col_idx in range(len(table.columns)):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#1c7ed6")
        cell.set_text_props(color="white", weight="bold")

    # Highlight the best row (already first because sorted).
    for col_idx in range(len(table.columns)):
        tbl[1, col_idx].set_facecolor("#e7f5ff")

    output_path = FIGURES_DIR / "mlflow_runs.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
