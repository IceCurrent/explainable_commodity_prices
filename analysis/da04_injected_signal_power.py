"""Deep-analysis 04: injected-signal power study (end-to-end pipeline positive control).

Question: if the factor state DID forecast next-day returns with predictive
R^2 = x, would the project's own engine (expanding FAAR + pooled Clark-West)
detect it? We plant a known signal on top of the real returns, keeping the
real state, the real engine, the real split and the real test:

    r*_{i,t+1} = r_{i,t+1} + c * g_i * x_t,   x_t = standardized combo of state dims

with c calibrated so the injected pooled predictive R^2 (in train-standardized
units) hits each target level. Detection = pooled CW p < 0.05.

This simultaneously (a) validates the entire wiring end-to-end (a broken
engine cannot recover planted signal), (b) traces the power curve / minimal
detectable effect at h=1, and (c) verifies the R^2_OOS accounting recovers the
planted magnitude.

Output: results/deep_analysis/da04_power_curve.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqcp.config import load_factor_model_config, load_forecast_pbsv_config
from eqcp.forecasting.ar1 import AR_KEY, clark_west, expanding_subset_forecasts, r2_oos
from eqcp.forecasting.basis import fit_frozen_basis
from eqcp.io.commodities import load_return_panel
from eqcp.io.macro import load_macro_stationary
from eqcp.pipelines.forecast_pbsv import _standardized_weights, train_leakfree_factors
from eqcp.pipelines.macro_mapping import RunLogger

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)

R2_LEVELS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02]
N_REP = 8


def main() -> None:
    cfg = load_forecast_pbsv_config(None)
    factor_cfg = load_factor_model_config(None)
    panel = load_return_panel()
    returns = panel.raw
    dates = panel.dates
    T, N = returns.shape
    t_split = int(cfg.train_frac * T)

    log = RunLogger()
    F = train_leakfree_factors(returns, dates, t_split, factor_cfg, seed=0, log=log)
    manifest = pd.read_csv(ROOT / "data/processed/transform_manifest.csv")
    M_full = load_macro_stationary(ROOT / "data/processed/macro_stationary.csv", manifest)
    basis = fit_frozen_basis(
        F.iloc[:t_split],
        M_full.loc[M_full.index < dates[t_split]],
        n_perm=50,
        seed=0,
    )
    state = basis.variates(F).to_numpy(float)
    K = state.shape[1]
    full_key = tuple(range(K))

    sd_train = returns[:t_split].std(axis=0, ddof=0)  # (N,)
    stale = np.zeros(N, dtype=bool)
    weights = _standardized_weights(returns[:t_split], ~stale)

    rows = []
    rng_master = np.random.default_rng(2024)
    for r2_target in R2_LEVELS:
        for rep in range(N_REP if r2_target > 0 else 2 * N_REP):
            rng = np.random.default_rng(rng_master.integers(1 << 31))
            # signal direction: random combo of the K state dims, standardized on train
            w = rng.standard_normal(K)
            x = state @ w
            x = (x - x[:t_split].mean()) / x[:t_split].std(ddof=0)
            g = 0.5 + rng.random(N)  # heterogeneous positive loadings
            g = g / np.sqrt((g**2).mean())
            # injected component in standardized units: c * g_i * x_t adds
            # variance c^2 g_i^2 to r_i/sd_i; pooled injected R^2 = c^2 mean(g^2) = c^2.
            c = np.sqrt(r2_target)
            r_star = returns.copy()
            r_star[1:] += c * (g[None, :] * x[:-1, None]) * sd_train[None, :]

            fr = expanding_subset_forecasts(
                r_star, state, t_split, horizon=1,
                subsets=[AR_KEY, full_key], min_train=cfg.min_train,
            )
            stat, p = clark_west(fr, full_key, AR_KEY, weights=weights)
            r2 = r2_oos(fr, full_key, AR_KEY, weights)
            rows.append({
                "r2_injected": r2_target, "rep": rep, "cw_stat": stat, "cw_p": p,
                "r2_oos_vs_ar1": r2, "detected": p < 0.05,
            })
            print(f"R2_inj={r2_target:.4f} rep={rep}: CW p={p:.4f} R2_OOS={r2:+.5f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "da04_power_curve.csv", index=False)
    summary = df.groupby("r2_injected").agg(
        detect_rate=("detected", "mean"),
        median_p=("cw_p", "median"),
        mean_r2_oos=("r2_oos_vs_ar1", "mean"),
    )
    print("\npower curve (pooled CW, h=1, n_oos=1148):")
    print(summary.to_string())
    summary.to_csv(OUT / "da04_power_summary.csv")


if __name__ == "__main__":
    main()
