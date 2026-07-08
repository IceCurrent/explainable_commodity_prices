"""Shared helpers for the deep-analysis predictive scans."""

from __future__ import annotations

import numpy as np


def nw_tstat(x: np.ndarray, y: np.ndarray, lags: int) -> float:
    """Newey-West t-stat of the slope in y = a + b x."""
    x = x - x.mean()
    y = y - y.mean()
    sxx = float(x @ x)
    if sxx <= 0:
        return 0.0
    b = float(x @ y) / sxx
    u = y - b * x
    z = x * u
    T = len(z)
    v = float(z @ z) / T
    for lag in range(1, min(lags, T - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        v += 2.0 * w * float(z[:-lag] @ z[lag:]) / T
    se = np.sqrt(v / T) / (sxx / T)
    return b / se if se > 0 else 0.0


def h_fwd_returns(r: np.ndarray, h: int) -> np.ndarray:
    """Forward h-day cumulative return starting at t+1: out[t] = sum(r[t+1..t+h])."""
    T, N = r.shape
    c = np.cumsum(r, axis=0)
    out = np.full((T, N), np.nan)
    out[: T - h] = c[h:] - c[:-h]
    return out


def tmat(m_arr: np.ndarray, y: np.ndarray, lags: int) -> np.ndarray:
    """(J, N) matrix of NW t-stats for y_fwd ~ each macro column."""
    valid = ~np.isnan(y).any(axis=1)
    J, N = m_arr.shape[1], y.shape[1]
    out = np.zeros((J, N))
    for j in range(J):
        xj = m_arr[valid, j]
        for i in range(N):
            out[j, i] = nw_tstat(xj, y[valid, i], lags=lags)
    return out


# Approximate official settlement/close times in ET (hours after midnight).
# Sources: CME/ICE/LME published settlement windows; used only for the
# cross-sectional overlap fingerprint, not for any return computation.
SETTLE_ET: dict[str, float] = {
    "WTI": 14.5, "Brent": 14.5, "Gasoline": 14.5, "HeatingOil": 14.5, "NaturalGas": 14.5,
    "Gold": 13.5, "Silver": 13.42, "Copper": 13.0, "Platinum": 13.08,
    "Aluminium": 12.0, "Nickel": 12.0, "Zinc": 12.0,  # LME closing evaluations ~17:00 London
    "Corn": 14.33, "HRWWheat": 14.33, "Soybeans": 14.33, "SoybeanOil": 14.33,
    "LeanHogs": 14.08, "LiveCattle": 14.08,
    "Coffee": 13.5, "Sugar": 13.0, "Cotton": 14.33,
}

# Macro series grouped by when their official daily value is determined (ET).
# AFTER = after the earliest commodity settles (>= ~13:00 ET): overlap possible.
# BEFORE = determined before US commodity settlements: no same-day overlap.
MACRO_CLOSE_GROUP: dict[str, str] = {
    **{s: "after" for s in [
        "fx_dxy", "fx_bbdxy", "fx_eurusd", "fx_usdcnh", "fx_audusd", "fx_usdbrl",
        "fx_usdclp", "fx_usdcad", "fx_usdnok", "fx_usdzar",
        "ust_10y", "ust_5y", "ust_2y", "tips_10y", "tips_5y", "be_10y", "be_5y",
        "inflsw_5y", "inflsw_10y", "hy_oas", "ig_oas", "vix", "move", "ovx", "gvz",
        "spx", "mxwo", "mxef", "xle",
    ]},
    **{s: "before" for s in ["csi300", "shcomp", "hscei", "cny_10y", "bdiy", "bdti"]},
    **{s: "ambiguous" for s in ["gpr", "epu"]},
}
