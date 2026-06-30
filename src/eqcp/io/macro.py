"""Macro panel loaders and calendar helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_macro_stationary(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    """Load the convenience stationary macro panel (manifest column order)."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    cols = [c for c in manifest["series_id"] if c in df.columns]
    return df[cols]


def build_preferred_panel(
    levels_path: Path,
    manifest: pd.DataFrame,
    factor_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reindex aligned levels to factor dates, transform on that calendar."""
    lv = pd.read_csv(levels_path, parse_dates=["date"], index_col="date").sort_index()
    cols = [c for c in manifest["series_id"] if c in lv.columns]
    lv = lv[cols].reindex(factor_index.union(lv.index)).sort_index()
    tmap = dict(zip(manifest["series_id"], manifest["transform"]))
    out: dict[str, pd.Series] = {}
    for c in cols:
        s = lv[c]
        t = tmap.get(c, "level")
        if t == "log_return":
            out[c] = np.log(s / s.shift(1))
        elif t == "diff":
            out[c] = s.diff()
        else:
            out[c] = s
    return pd.DataFrame(out).reindex(factor_index).dropna(how="any")


def load_regime_labels(path: Path, index: pd.DatetimeIndex) -> pd.Series:
    reg = pd.read_csv(path, parse_dates=["start", "end"])
    labels = pd.Series(index=index, dtype=object)
    for _, row in reg.iterrows():
        m = (index >= row["start"]) & (index <= row["end"])
        labels[m] = row["regime"]
    return labels
