"""Commodity return panel loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
COMMODITIES_DIR = DATA_DIR / "commodities"

# --------------------------------------------------------------------------- #
# Canonical panel definition (single source of truth).
#
# The following series were dropped because they are not continuously
# price-discovered daily futures: they carry stale (unchanged) settlement marks
# on a large fraction of days, which biases the AE factor extraction, attenuates
# the CCA canonical correlations, and mechanically inflates forecast benchmarks.
# Zero-return fraction / longest stale run measured on the raw price panel:
#   Lithium 69.7% (57d)  HRCSteel 37.6% (16d)  SGXIronOre 23.8% (19d)
#   Methanol 19.4% (22d)  ThermalCoal 8.4% (4d)  Diesel 6.3% (2d)
# The retained 21 series all sit below ~2% zero-returns (clean daily discovery).
EXCLUDED_COMMODITIES: tuple[str, ...] = (
    "Diesel",
    "ThermalCoal",
    "Methanol",
    "HRCSteel",
    "SGXIronOre",
    "Lithium",
)

# Sector partition of the retained panel. Used by the sector-wise analysis.
SECTORS: dict[str, list[str]] = {
    "agriculture": [
        "Coffee",
        "SoybeanOil",
        "Corn",
        "Cotton",
        "HRWWheat",
        "LeanHogs",
        "LiveCattle",
        "Soybeans",
        "Sugar",
    ],
    "energy": ["Brent", "WTI", "Gasoline", "HeatingOil", "NaturalGas"],
    "metals": ["Gold", "Silver", "Copper", "Aluminium", "Nickel", "Zinc", "Platinum"],
}


def sector_of(commodity: str) -> str | None:
    """Return the sector name a commodity belongs to (or ``None``)."""
    for sector, members in SECTORS.items():
        if commodity in members:
            return sector
    return None


@dataclass(frozen=True)
class ReturnPanel:
    dates: pd.DatetimeIndex
    commodities: list[str]
    raw: np.ndarray  # (T, N) log-returns
    standardized: np.ndarray  # (T, N) full-sample z-scores

    @property
    def n_obs(self) -> int:
        return self.raw.shape[0]

    @property
    def n_assets(self) -> int:
        return self.raw.shape[1]


def load_return_panel(
    data_dir: Path | None = None,
    exclude: tuple[str, ...] | list[str] | None = EXCLUDED_COMMODITIES,
    commodities: list[str] | None = None,
) -> ReturnPanel:
    """Load aligned commodity log-returns from ``data/commodities/prices.csv``.

    ``exclude`` drops the poisonous stale-priced series (defaults to
    :data:`EXCLUDED_COMMODITIES`; the raw file no longer contains them, so this
    is a defensive no-op that also guards against a stale panel being restored).
    ``commodities`` optionally restricts the panel to an explicit subset (used by
    the sector-wise analysis to build per-sector sub-panels).
    """
    data_dir = data_dir or COMMODITIES_DIR
    prices_path = data_dir / "prices.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"Commodity prices not found at {prices_path}")
    prices = pd.read_csv(prices_path, parse_dates=["date"], index_col="date")
    prices = prices.sort_index()
    if exclude:
        prices = prices.drop(columns=[c for c in prices.columns if c in set(exclude)])
    if commodities is not None:
        missing = [c for c in commodities if c not in prices.columns]
        if missing:
            raise KeyError(f"requested commodities not in panel: {missing}")
        prices = prices[commodities]
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
