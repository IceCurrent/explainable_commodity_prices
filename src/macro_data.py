"""Generic macro panel loader.

Expected inputs (placed in data/macro/ before running):
  macro_panel_clean.csv  -- wide panel: date column + one column per macro series.
                            Series can be at any frequency (daily, weekly, monthly).
                            Values should be the raw level; transforms are applied here.
  manifest.csv           -- optional metadata table with columns:
                              name      : matches a column in macro_panel_clean.csv
                              transform : one of {level, diff, log_change}
                              group     : arbitrary label (FX, Rates, etc.)
                            If absent, all series are treated as already stationary
                            (transform = "level").

Output: a MacroPanel dataclass with:
  stationary  -- transformed + forward-filled daily DataFrame aligned to target_index
  raw         -- untransformed levels
  manifest    -- per-series metadata

Transform conventions:
  level      -- series is already stationary (spreads, vol, diffusion indices, YoY rates)
  diff       -- first difference (interest-rate yields, inventory levels)
  log_change -- log first difference (FX, price indices, freight, IP)
               Falls back to pct_change when values are non-positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MACRO_DIR = PROJECT_ROOT / "data" / "macro"
CLEAN_PANEL = MACRO_DIR / "macro_panel_clean.csv"
MANIFEST = MACRO_DIR / "manifest.csv"


@dataclass
class MacroPanel:
    """Container for the loaded macro panel (raw levels + stationary transforms + metadata)."""

    raw: pd.DataFrame         # untransformed levels, date-indexed
    stationary: pd.DataFrame  # transformed + aligned to target index
    manifest: pd.DataFrame    # per-series metadata


def _apply_transform(series: pd.Series, transform: str) -> pd.Series:
    series = series.dropna()
    if transform == "level":
        return series
    if transform == "diff":
        return series.diff()
    if transform == "log_change":
        if (series <= 0).any():
            return series.pct_change()
        return np.log(series).diff()
    raise ValueError(f"unknown transform {transform!r}; expected level/diff/log_change")


def load_macro_panel(
    target_index: pd.DatetimeIndex | None = None,
    drop_groups: list[str] | None = None,
    include_groups: list[str] | None = None,
    exclude_series: list[str] | None = None,
) -> MacroPanel:
    """Load, transform, and align the macro panel.

    Parameters
    ----------
    target_index : DatetimeIndex, optional
        Daily dates to align to (forward-fill). Pass the factor as-of dates
        so the macro panel is 1-to-1 with the aligned factor panel.
    drop_groups : list[str], optional
        Drop series whose manifest ``group`` is in this list.
    include_groups : list[str], optional
        Keep only series whose manifest ``group`` is in this list.
    exclude_series : list[str], optional
        Additional series names to drop before transforming.
    """
    if not CLEAN_PANEL.exists():
        raise FileNotFoundError(
            f"{CLEAN_PANEL} not found.\n"
            "Run `python data/clean_macro_panel.py --input <your_file>` first,\n"
            "or place a pre-cleaned wide CSV at that path directly."
        )

    raw = (
        pd.read_csv(CLEAN_PANEL, parse_dates=["date"])
        .set_index("date")
        .sort_index()
    )

    # Build or load manifest.
    if MANIFEST.exists():
        manifest = pd.read_csv(MANIFEST)
        # Keep only series that exist in the panel.
        manifest = manifest[manifest["name"].isin(raw.columns)].copy()
    else:
        manifest = pd.DataFrame({
            "name": raw.columns.tolist(),
            "transform": "level",
            "group": "unknown",
        })

    # Apply group / series filters.
    if drop_groups:
        manifest = manifest[~manifest["group"].isin(drop_groups)]
    if include_groups:
        manifest = manifest[manifest["group"].isin(include_groups)]
    if exclude_series:
        manifest = manifest[~manifest["name"].isin(exclude_series)]

    raw = raw[manifest["name"].tolist()]

    # Apply transforms per series at native frequency.
    cols: dict[str, pd.Series] = {}
    for _, row in manifest.iterrows():
        transform = row.get("transform", "level")
        cols[row["name"]] = _apply_transform(raw[row["name"]], transform)

    stationary = pd.DataFrame(cols).sort_index()

    # Forward-fill onto a daily grid (or the caller's target_index).
    if target_index is not None:
        grid = pd.DatetimeIndex(target_index)
    else:
        grid = pd.bdate_range(stationary.index.min(), stationary.index.max())

    full_idx = stationary.index.union(grid)
    stationary = stationary.reindex(full_idx).ffill().reindex(grid)
    stationary = stationary.dropna(how="any")

    # Drop zero-variance columns that would break standardization.
    nunique = stationary.nunique()
    degenerate = nunique[nunique <= 1].index.tolist()
    if degenerate:
        stationary = stationary.drop(columns=degenerate)
        manifest = manifest[~manifest["name"].isin(degenerate)]

    return MacroPanel(
        raw=raw,
        stationary=stationary,
        manifest=manifest.reset_index(drop=True),
    )
