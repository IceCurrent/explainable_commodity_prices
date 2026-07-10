"""Shared helpers for the rolling pipelines (overall panel + sector blocks).

Both :mod:`eqcp.pipelines.rolling_forecast` (whole 21-commodity panel) and
:mod:`eqcp.pipelines.rolling_sectors` (energy / agriculture / metals) run the
*same* walk-forward engine (:func:`eqcp.forecasting.rolling.rolling_forecasts`)
and then reduce the pooled out-of-sample forecasts to the same tables. Keeping
that reduction in one place guarantees the overall and sector arms are scored
identically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eqcp.config import RollingForecastConfig
from eqcp.forecasting.ar1 import (
    AR_KEY,
    MEAN_KEY,
    ZERO_KEY,
    clark_west,
    h_period_returns,
    r2_oos,
)
from eqcp.forecasting.pbsv import bootstrap_pbsv, efficiency_gap, pbsv
from eqcp.forecasting.rolling import RollingResult


def standardized_weights(y_h: np.ndarray) -> np.ndarray:
    """1/variance commodity weights (standardized-loss pooling)."""
    var = np.nanvar(y_h, axis=0)
    var = np.where(var == 0, np.nan, var)
    w = 1.0 / var
    return w / np.nanmean(w)


def grouping_boundary(roll: RollingResult) -> tuple[int, list[tuple[int, ...]]]:
    """Median spanned-block boundary across windows -> {spanned | weak} groups."""
    K = roll.n_dims
    median_boundary = int(np.median(roll.spanned_by_window))
    median_boundary = min(max(median_boundary, 1), K - 1)
    groups = [tuple(range(median_boundary)), tuple(range(median_boundary, K))]
    return median_boundary, groups


def summarize_forecasts(
    roll: RollingResult,
    returns: np.ndarray,
    cfg: RollingForecastConfig,
    stability: pd.DataFrame,
    groups: list[tuple[int, ...]],
    seed: int,
) -> dict[str, Any]:
    """Reduce pooled per-window OOS forecasts to accuracy / Shapley / grouped tables.

    Returns a dict with ``pooled`` (per-horizon accuracy + Clark-West + bootstrap
    v_full CI), ``shapley`` (rank-pooled PBSV phi with bootstrap bands, tagged by
    the rank's cross-window macro stability), ``grouped`` (spanned vs weak block
    attribution), and ``acceptance`` (Shapley-efficiency checks).
    """
    K = roll.n_dims
    full_key = tuple(range(K))
    pooled_rows: list[dict] = []
    shap_rows: list[pd.DataFrame] = []
    grouped_rows: list[dict] = []
    acceptance: list[tuple[str, bool, str]] = []

    for h, fr in roll.forecasts.items():
        y_h = h_period_returns(returns, h)
        weights = standardized_weights(y_h)
        cw = clark_west(fr, full_key, AR_KEY, weights=weights)
        nomask = fr.nonoverlap_mask()
        cw_no = clark_west(
            fr.select_origins(nomask), full_key, AR_KEY, hac_lags=1, weights=weights
        )

        res_p = pbsv(fr, weights=weights, groups=groups)
        boot = bootstrap_pbsv(
            fr,
            weights=weights,
            groups=groups,
            mean_block=max(cfg.mean_block, 2 * h),
            n_boot=cfg.n_boot,
            seed=seed,
        )
        v_full_draws = boot["phi_draws"].sum(axis=1)
        pooled_rows.append(
            {
                "horizon": h,
                "n_origins": len(fr.origins),
                "n_nonoverlap": int(nomask.sum()),
                "r2_vs_ar1": r2_oos(fr, full_key, AR_KEY, weights),
                "r2_vs_mean": r2_oos(fr, full_key, MEAN_KEY, weights),
                "r2_vs_zero": r2_oos(fr, full_key, ZERO_KEY, weights),
                "r2_ar1_vs_zero": r2_oos(fr, AR_KEY, ZERO_KEY, weights),
                "cw_stat": cw[0],
                "cw_p": cw[1],
                "cw_nonoverlap_p": cw_no[1],
                "v_full": res_p["v_full"],
                "v_full_boot_lo": float(np.percentile(v_full_draws, 2.5)),
                "v_full_boot_hi": float(np.percentile(v_full_draws, 97.5)),
                "beats_zero": bool(r2_oos(fr, full_key, ZERO_KEY, weights) > 0),
            }
        )
        shap_rows.append(
            pd.DataFrame(
                {
                    "horizon": h,
                    "rank": [f"V{k + 1}" for k in range(K)],
                    "macro_stable": [
                        bool(stability.iloc[k]["macro_stable"]) for k in range(K)
                    ],
                    "phi": res_p["phi"],
                    "boot_lo": boot["phi_lo"],
                    "boot_hi": boot["phi_hi"],
                }
            )
        )
        grouped_rows.append(
            {
                "horizon": h,
                "phi_spanned": float(res_p["phi_grouped"][0]),
                "phi_weak": float(res_p["phi_grouped"][1]),
                "boot_lo_spanned": float(boot["phi_grouped_lo"][0]),
                "boot_hi_spanned": float(boot["phi_grouped_hi"][0]),
                "boot_lo_weak": float(boot["phi_grouped_lo"][1]),
                "boot_hi_weak": float(boot["phi_grouped_hi"][1]),
                "v_full": res_p["v_full"],
            }
        )
        acceptance.append(
            (
                f"h={h}: Shapley efficiency |sum(phi)-v(full)| < 1e-10",
                efficiency_gap(res_p) < 1e-10,
                f"gap={efficiency_gap(res_p):.2e}",
            )
        )

    return {
        "pooled": pd.DataFrame(pooled_rows),
        "shapley": pd.concat(shap_rows, ignore_index=True),
        "grouped": pd.DataFrame(grouped_rows),
        "acceptance": acceptance,
    }


def fingerprint_frame(roll: RollingResult) -> pd.DataFrame:
    """Mean (sign-anchored) macro structure-correlation per rank across windows."""
    K = roll.n_dims
    mean_struct = roll.struct_macro.mean(axis=0)  # (J, K)
    return pd.DataFrame(
        mean_struct, index=roll.macro_cols, columns=[f"V{k + 1}" for k in range(K)]
    )


def top_drivers(fp: pd.DataFrame, rank: str, n: int = 4) -> str:
    """Human-readable top-|load| macro drivers for one rank's fingerprint."""
    col = fp[rank]
    order = col.abs().sort_values(ascending=False).head(n).index
    return "; ".join(f"{name}({col.loc[name]:+.2f})" for name in order)
