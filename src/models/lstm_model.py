"""PyTorch LSTM forecaster with mini-batch training and early stopping."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import CONFIG, TARGET_COLUMN, LSTMConfig
from .base import BaseForecaster


def _to_windows(
    series: np.ndarray,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding window: X[i] = series[i:i+window], y[i] = series[i+window+horizon-1]."""

    n = len(series) - window - horizon + 1
    if n <= 0:
        return np.empty((0, window), dtype=np.float32), np.empty((0,), dtype=np.float32)

    indexer = np.arange(window)[None, :] + np.arange(n)[:, None]
    X = series[indexer].astype(np.float32)
    y = series[window + horizon - 1 : window + horizon - 1 + n].astype(np.float32)
    return X, y


class _LSTMNet(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    name = "lstm"

    def __init__(
        self,
        target: str = TARGET_COLUMN,
        horizon: int = 1,
        config: Optional[LSTMConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.target = target
        self.horizon = horizon
        self.config = config or CONFIG.lstm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: Optional[_LSTMNet] = None
        self.train_mean_: float = 0.0
        self.train_std_: float = 1.0
        self.train_history_: list[Dict[str, float]] = []
        self.best_epoch_: Optional[int] = None

    @property
    def context_rows(self) -> int:
        return self.config.input_window + self.horizon - 1

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.train_mean_) / max(self.train_std_, 1e-8)

    def _destandardise(self, x: np.ndarray) -> np.ndarray:
        return x * self.train_std_ + self.train_mean_

    def _build_loaders(
        self,
        train_series: np.ndarray,
        val_series: Optional[np.ndarray],
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        cfg = self.config
        X_train, y_train = _to_windows(self._standardise(train_series), cfg.input_window, self.horizon)
        Xt = torch.from_numpy(X_train).unsqueeze(-1)
        yt = torch.from_numpy(y_train)

        train_loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = None
        if val_series is not None and len(val_series) > cfg.input_window + self.horizon:
            X_val, y_val = _to_windows(self._standardise(val_series), cfg.input_window, self.horizon)
            Xv = torch.from_numpy(X_val).unsqueeze(-1)
            yv = torch.from_numpy(y_val)
            val_loader = DataLoader(
                TensorDataset(Xv, yv),
                batch_size=cfg.batch_size,
                shuffle=False,
            )
        return train_loader, val_loader

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        **_: Any,
    ) -> "LSTMForecaster":
        cfg = self.config
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)

        train_series = train_df[self.target].to_numpy(dtype=np.float64)
        self.train_mean_ = float(train_series.mean())
        self.train_std_ = float(train_series.std() + 1e-8)

        val_series = (
            val_df[self.target].to_numpy(dtype=np.float64) if val_df is not None else None
        )
        train_loader, val_loader = self._build_loaders(train_series, val_series)

        self.model_ = _LSTMNet(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.model_.parameters(), lr=cfg.learning_rate)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        patience = cfg.early_stopping_patience
        bad_epochs = 0

        for epoch in range(1, cfg.epochs + 1):
            self.model_.train()
            running = 0.0
            n_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimiser.zero_grad()
                preds = self.model_(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimiser.step()
                running += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)
            train_loss = running / max(n_seen, 1)

            val_loss = float("nan")
            if val_loader is not None:
                self.model_.eval()
                running = 0.0
                n_seen = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)
                        preds = self.model_(xb)
                        loss = loss_fn(preds, yb)
                        running += float(loss.item()) * xb.size(0)
                        n_seen += xb.size(0)
                val_loss = running / max(n_seen, 1)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                    self.best_epoch_ = epoch
                    bad_epochs = 0
                else:
                    bad_epochs += 1

            self.train_history_.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )

            if val_loader is not None and bad_epochs >= patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(
        self,
        df: pd.DataFrame,
        context: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        if self.model_ is None:
            raise RuntimeError("Call .fit() before .predict().")

        cfg = self.config
        full, _ = self._with_context(df, context)
        series = full[self.target].to_numpy(dtype=np.float64)
        X, y = _to_windows(self._standardise(series), cfg.input_window, self.horizon)
        if X.size == 0:
            empty = pd.Series([], dtype=np.float64)
            return empty, empty

        Xt = torch.from_numpy(X).unsqueeze(-1).to(self.device)

        self.model_.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(Xt), cfg.batch_size):
                batch = Xt[start : start + cfg.batch_size]
                preds.append(self.model_(batch).cpu().numpy())
        y_pred = np.concatenate(preds)
        y_pred = self._destandardise(y_pred)

        # The first valid target is series[input_window + horizon - 1].
        target_indices = full.index[cfg.input_window + self.horizon - 1 :]
        target_indices = target_indices[: len(y_pred)]
        y_true = pd.Series(self._destandardise(y), index=target_indices, name="y_true")
        y_pred_series = pd.Series(y_pred, index=target_indices, name="y_pred")

        scored = target_indices.isin(df.index)
        return y_true[scored], y_pred_series[scored]

    def get_params(self) -> Dict[str, Any]:
        params = asdict(self.config)
        params.update(
            {
                "target": self.target,
                "horizon": self.horizon,
                "device": self.device,
                "best_epoch": self.best_epoch_,
                "train_mean": round(self.train_mean_, 6),
                "train_std": round(self.train_std_, 6),
            }
        )
        return params

    def save(self, output_dir: Path) -> Path:
        if self.model_ is None:
            raise RuntimeError("Nothing to save — call .fit() first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model_.state_dict(), output_dir / "lstm_state_dict.pt")

        meta = {
            "target": self.target,
            "horizon": self.horizon,
            "device": self.device,
            "best_epoch": self.best_epoch_,
            "train_mean": self.train_mean_,
            "train_std": self.train_std_,
            "config": asdict(self.config),
        }
        (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        if self.train_history_:
            history_path = output_dir / "training_history.csv"
            pd.DataFrame(self.train_history_).to_csv(history_path, index=False)
        return output_dir
