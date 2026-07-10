"""Rolling-window forecast + rolling-explainability pipeline.

Replaces the frozen-basis FAAR study with a walk-forward roll that re-fits the
AE and the CCA attribution basis on every window (see
:mod:`eqcp.forecasting.rolling`). Two questions are answered jointly:

1. *Forecasting* — pooled out-of-sample multi-horizon accuracy of the
   factor-augmented AR versus AR(1)/mean/zero benchmarks, over every rolling
   test block.
2. *Explainability under rolling* — does each canonical rank keep the same
   macro fingerprint across windows even as the AE latent coordinates rotate?
   Rank-pooled Shapley attribution is only licensed for ranks that clear the
   cross-window macro-cosine stability bar; the rest are reported as a
   disclosed limitation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from eqcp.config import (
    FactorModelConfig,
    RollingForecastConfig,
    load_factor_model_config,
    load_rolling_forecast_config,
)
from eqcp.factors.extract import FactorModelType
from eqcp.forecasting.rolling import RollingResult, rank_stability, rolling_forecasts
from eqcp.io.commodities import load_return_panel
from eqcp.io.macro import load_macro_stationary
from eqcp.pipelines._rolling_common import (
    fingerprint_frame,
    grouping_boundary,
    summarize_forecasts,
)
from eqcp.pipelines._runlog import RunLogger
from eqcp.reporting.rolling_figures import make_rolling_figures
from eqcp.reporting.rolling_report import write_rolling_report

logger = logging.getLogger(__name__)

ROLLING_SUBDIRS: dict[FactorModelType, str] = {
    "vanilla": "rolling_forecast",
    "beta_vae": "rolling_forecast_beta",
}


def run_rolling_forecast(args: argparse.Namespace) -> None:
    """Run the full rolling-window forecast + rolling-explainability pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg: RollingForecastConfig = load_rolling_forecast_config(args.config)
    factor_cfg: FactorModelConfig = load_factor_model_config(args.factor_config)
    seed = args.seed
    model_type: FactorModelType = getattr(args, "factor_model", "vanilla")

    out = Path(args.outdir)
    res = out / "results" / ROLLING_SUBDIRS[model_type]
    figs = out / "figures" / ROLLING_SUBDIRS[model_type]
    rep = out / "reports"
    for d in (res, figs, rep):
        d.mkdir(parents=True, exist_ok=True)
    log = RunLogger()
    acceptance: list[tuple[str, bool, str]] = []

    panel = load_return_panel()
    returns = panel.raw
    dates = panel.dates
    commodities = list(panel.commodities)
    T, N = returns.shape
    manifest = pd.read_csv(args.manifest)
    M_full = load_macro_stationary(args.macro, manifest)
    log(
        f"calendar: T={T} N={N} ({dates[0].date()} -> {dates[-1].date()}); "
        f"rolling {cfg.train_window}d train / {cfg.test_block}d test, step {cfg.step}; "
        f"model={model_type} seed={seed}"
    )

    roll: RollingResult = rolling_forecasts(
        returns, dates, M_full, cfg, factor_cfg, model_type, seed, log=log
    )
    roll.commodities = commodities
    K = roll.n_dims
    log(
        f"rolling roll complete: {roll.n_windows}/{roll.meta['n_windows_total']} windows usable; "
        f"K={K}; horizons={list(roll.forecasts)}"
    )

    roll.windows.to_csv(res / "rolling_windows.csv", index=False)

    # ----------------------------------------------- explainability-under-roll
    stab = rank_stability(
        roll, cfg.fingerprint_cos_stable, n_perm=cfg.n_perm_stability, seed=seed
    )
    stab.to_csv(res / "rank_stability.csv", index=False)
    n_stable = int(stab["macro_stable"].sum())
    n_above_null = int((stab["p_value"] < 0.05).sum())
    chance = float(stab["chance_abs_cos"].iloc[0])
    log(
        "rank stability (cross-window macro cosine vs shuffle null; chance |cos|~"
        f"{chance:.2f}): "
        + "; ".join(
            f"{r['rank']} med|cos|={r['median_abs_cos']:.2f} "
            f"(null p95={r['null_p95']:.2f}, p={r['p_value']:.3f}, "
            f"rho_oos={r['median_rho_oos']:.2f}) [{r['stability_class']}]"
            for _, r in stab.iterrows()
        )
    )
    log(
        f"  -> {n_stable}/{K} ranks macro-stable (bar={cfg.fingerprint_cos_stable}); "
        f"{n_above_null}/{K} clear the shuffle null (p<0.05)"
    )
    acceptance.append(
        (
            "leading rank is macro-stable AND clears the shuffle null",
            bool(stab.iloc[0]["macro_stable"]),
            f"V1 median|cos|={stab.iloc[0]['median_abs_cos']:.2f}, "
            f"p={stab.iloc[0]['p_value']:.3f}, rho_oos={stab.iloc[0]['median_rho_oos']:.2f}",
        )
    )

    # Average (sign-anchored) macro fingerprint per rank across windows.
    fp = fingerprint_frame(roll)
    fp.to_csv(res / "macro_fingerprint.csv")

    median_boundary, groups = grouping_boundary(roll)
    log(
        f"pooled grouping boundary (median spanned across windows) = V1..V{median_boundary} "
        f"| V{median_boundary + 1}..V{K}"
    )

    # ------------------------------------------------- forecasting + attribution
    summ = summarize_forecasts(roll, returns, cfg, stab, groups, seed)
    pooled_df = summ["pooled"]
    shap_df = summ["shapley"]
    grouped_df = summ["grouped"]
    acceptance.extend(summ["acceptance"])
    for _, row in pooled_df.iterrows():
        log(
            f"h={int(row['horizon'])}: r2_vs_ar1={row['r2_vs_ar1']:+.5f} "
            f"CW p={row['cw_p']:.4f} (nonoverlap {row['cw_nonoverlap_p']:.4f}); "
            f"v_full={row['v_full']:+.3e}"
        )

    pooled_df.to_csv(res / "forecast_accuracy_pooled.csv", index=False)
    shap_df.to_csv(res / "pbsv_shapley.csv", index=False)
    grouped_df.to_csv(res / "pbsv_grouped.csv", index=False)

    # ------------------------------------------------------------------ verdict
    h0 = min(roll.forecasts)
    head = pooled_df[pooled_df["horizon"] == h0].iloc[0]
    forecast_gate = bool(head["cw_p"] < 0.05 and head["v_full_boot_lo"] > 0 and head["beats_zero"])
    explain_survives = bool(stab.iloc[0]["macro_stable"])
    log(
        f"VERDICT: forecast gate (h={h0}) passed={forecast_gate}; "
        f"explainability-survives-rolling={explain_survives} "
        f"({n_stable}/{K} ranks macro-stable)"
    )

    make_rolling_figures(
        figs,
        windows=roll.windows,
        stability=stab,
        fingerprint=fp,
        pooled=pooled_df,
        shapley=shap_df,
        n_dims=K,
    )
    write_rolling_report(
        rep,
        cfg=cfg,
        seed=seed,
        model_type=model_type,
        n_windows=roll.n_windows,
        n_windows_total=roll.meta["n_windows_total"],
        span=(str(dates[0].date()), str(dates[-1].date())),
        stability=stab,
        fingerprint=fp,
        pooled=pooled_df,
        shapley=shap_df,
        grouped=grouped_df,
        median_boundary=median_boundary,
        forecast_gate=forecast_gate,
        explain_survives=explain_survives,
        acceptance=acceptance,
    )

    log.write(res / "run_log.txt")
    print("\n=== ACCEPTANCE ===")
    for name, ok, detail in acceptance:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"\nDone. Outputs under {res} , {figs} , {rep}")
