"""Markdown report for the rolling-window sector decomposition."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eqcp.config import RollingForecastConfig
from eqcp.factors.extract import FactorModelType
from eqcp.reporting.rolling_report import _md_table

_BLOCK_ORDER = ["overall", "energy", "agriculture", "metals"]


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_ord"] = df["block"].map({b: i for i, b in enumerate(_BLOCK_ORDER)}).fillna(99)
    return df.sort_values(["_ord", df.columns[1]]).drop(columns="_ord")


def write_rolling_sector_report(
    rep: Path,
    cfg: RollingForecastConfig,
    seed: int,
    model_type: FactorModelType,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    pooled: pd.DataFrame,
    fingerprint: pd.DataFrame,
) -> None:
    suffix = "" if model_type == "vanilla" else "_beta"
    path = rep / f"rolling_sectors{suffix}_report.md"

    stable_blocks = summary[summary["n_stable_ranks"] > 0]["block"].tolist()
    forecasting_blocks = summary[summary["any_forecast_beats_ar1"]]["block"].tolist()

    lines: list[str] = []
    a = lines.append
    a("# Rolling-Window Sector Decomposition Report")
    a("")
    a("## Verdict")
    a("")
    if stable_blocks:
        a(
            "**A stable macro axis under rolling is sector-heterogeneous.** Re-fitting the AE "
            "and the CCA basis on every window, the leading canonical rank clears the stability "
            f"bar in: **{', '.join(stable_blocks)}**. The remaining blocks keep only "
            "partial/weak macro identity across windows (their leading axis rotates with the "
            "AE re-fit)."
        )
    else:
        a(
            "**No block keeps a bar-clearing macro axis under per-window re-fitting.** Every "
            "block's leading canonical rank falls below the stability bar once the AE is "
            "retrained each window — a disclosed limitation of marrying rolling windows to "
            "sector-level explainability."
        )
    a("")
    if forecasting_blocks:
        a(
            f"Forecast value (factor-augmented AR beats AR(1), CW p<0.05) appears in: "
            f"**{', '.join(forecasting_blocks)}** at some horizon."
        )
    else:
        a(
            "**No block forecasts.** At no horizon does the factor-augmented AR beat AR(1) "
            "for any sector (all Clark–West p ≥ 0.05) — the EMH-coherent null holds "
            "sector-by-sector, not just in aggregate."
        )
    a("")
    a(
        f"Setup: walk-forward roll, **{cfg.train_window}d** train / **{cfg.test_block}d** test, "
        f"step {cfg.step}; model **{model_type}**, seed {seed}. Each sector runs the identical "
        "engine on its own returns (overall = full 21-commodity panel, K=5; each sector "
        f"K≤{3}). Nothing is fit on the full sample."
    )
    a("")

    a("## 1. Block summary")
    a("")
    a(_md_table(
        _ordered(summary)[[
            "block", "n_commodities", "n_factors", "n_windows", "n_stable_ranks",
            "n_ranks_above_null", "lead_rank_class", "lead_median_abs_cos",
            "lead_median_rho_oos", "best_forecast_cw_p",
        ]],
        floatfmt="{:.3f}",
    ))
    a("")
    a(
        "`lead_median_abs_cos` = the leading rank's cross-window median |cosine| of its macro "
        "structure-correlation vector (fixed macro space, so comparable across windows even as "
        "the AE latent axes rotate); `lead_median_rho_oos` = its frozen-in-window canonical "
        "correlation on the held-out test blocks; `best_forecast_cw_p` = the smallest "
        "Clark–West p across horizons for that block. See "
        f"`figures/rolling_sectors{suffix}/sector_leading_axis.png`."
    )
    a("")

    a("## 2. Per-rank cross-window stability & OOS macro correlation")
    a("")
    a(_md_table(
        _ordered(stability)[[
            "block", "rank", "median_abs_cos", "p_value", "median_rho_oos",
            "frac_windows_rho_oos_ge_0.3", "stability_class",
        ]],
        floatfmt="{:.3f}",
    ))
    a("")
    a(f"See `figures/rolling_sectors{suffix}/sector_oos_rho_by_rank.png`.")
    a("")

    a("## 3. Leading-rank macro fingerprint per block")
    a("")
    lead_fp = fingerprint[fingerprint["rank"] == "V1"][["block", "top_macro"]]
    a(_md_table(_ordered(lead_fp)))
    a("")

    a("## 4. Pooled forecast accuracy by sector and horizon")
    a("")
    a(_md_table(
        _ordered(pooled)[[
            "block", "horizon", "n_origins", "r2_vs_ar1", "r2_vs_zero", "cw_p",
            "cw_nonoverlap_p", "beats_zero",
        ]],
        floatfmt="{:+.5f}",
    ))
    a("")
    a(f"See `figures/rolling_sectors{suffix}/sector_forecast_r2.png`.")
    a("")

    path.write_text("\n".join(lines) + "\n")
