from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class ReturnPanel:
    dates: pd.DatetimeIndex
    commodities: list[str]
    raw: np.ndarray  # (T, N) log-returns
    standardized: np.ndarray  # (T, N) window-style z-scores (full sample here)

    @property
    def n_obs(self) -> int:
        return self.raw.shape[0]

    @property
    def n_assets(self) -> int:
        return self.raw.shape[1]


def load_return_panel(data_dir: Path | None = None) -> ReturnPanel:
    """Load aligned commodity log-returns from prices.csv."""
    data_dir = data_dir or DATA_DIR
    prices = pd.read_csv(data_dir / "prices.csv", parse_dates=["date"], index_col="date")
    prices = prices.sort_index()
    raw = np.log(prices / prices.shift(1)).dropna(how="any")
    commodities = raw.columns.tolist()
    raw_values = raw.to_numpy(dtype=np.float64)
    means = raw_values.mean(axis=0, keepdims=True)
    stds = raw_values.std(axis=0, ddof=0, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)
    standardized = (raw_values - means) / stds
    return ReturnPanel(
        dates=raw.index,
        commodities=commodities,
        raw=raw_values,
        standardized=standardized,
    )


def zscore_window(window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-commodity z-score inside a rolling window."""
    means = window.mean(axis=0, keepdims=True)
    stds = window.std(axis=0, ddof=0, keepdims=True)
    stds = np.where(stds == 0, 1.0, stds)
    return (window - means) / stds, means, stds
