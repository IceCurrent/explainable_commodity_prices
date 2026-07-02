"""Forecast-PBSV pipeline: factor-augmented AR(1) + forecast-based Shapley values.

Stages:
1. leak-free factors: retrain the vanilla AE on the TRAIN segment only
   (train-only z-scores), freeze the encoder, encode all days;
2. frozen attribution basis: CCA(F_train, M_train) with factor-side ridge
   pinned to 0 (invariance requirement), spanning labels from train-only
   purged-CV inference, data-driven spanned|weak boundary;
3. expanding-window direct h-step FAAR forecasts, refit for all 2^K factor
   subsets via one shared Gram per commodity;
4. PBSV: exact Shapley over canonical-variate players, grouped game over the
   {spanned, weakly-macro-correlated} blocks, bootstrap CIs (descriptive,
   conditional on the frozen basis), cardinality-matched placebo bands,
   boundary/weighting/burn-in/per-year robustness;
5. controls arm (EWMA vol + 12m momentum in BOTH benchmark and full model),
   raw-basis diagnostic + AE-seed subspace stability, macro-substitution arm
   on a common macro-available calendar with publication-lagged gpr/epu;
6. gate: share-of-gain language is licensed only if the pooled Clark-West
   rejects AND the v(full) bootstrap CI excludes 0 AND the full model beats
   the zero forecast; otherwise the headline is a well-identified null.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from eqcp.config import (
    FactorModelConfig,
    ForecastPBSVConfig,
    load_factor_model_config,
    load_forecast_pbsv_config,
)
from eqcp.factors.autoencoder import AETrainConfig, train_vanilla_autoencoder
from eqcp.forecasting.ar1 import (
    AR_KEY,
    MEAN_KEY,
    ZERO_KEY,
    SubsetForecasts,
    bh_fdr,
    clark_west,
    clark_west_per_commodity,
    default_hac_lags,
    ewma_vol,
    expanding_subset_forecasts,
    h_period_returns,
    r2_oos,
    r2_oos_per_commodity,
    staleness_stats,
    timing_utility_gain,
)
from eqcp.forecasting.basis import (
    basis_stability_bootstrap,
    fit_frozen_basis,
    frozen_oos_diagnostics,
    principal_angles,
)
from eqcp.forecasting.pbsv import (
    bootstrap_pbsv,
    boundary_sensitivity,
    efficiency_gap,
    pbsv,
    pbsv_on_mask,
    pbsv_per_commodity,
    placebo_pbsv,
)
from eqcp.io.commodities import load_return_panel
from eqcp.io.macro import load_macro_stationary
from eqcp.pipelines.macro_mapping import RunLogger
from eqcp.reporting.forecast_figures import make_forecast_figures
from eqcp.reporting.forecast_report import write_forecast_report

logger = logging.getLogger(__name__)


def train_leakfree_factors(
    returns_raw: np.ndarray,
    dates: pd.DatetimeIndex,
    t_split: int,
    factor_cfg: FactorModelConfig,
    seed: int,
    log: RunLogger,
) -> pd.DataFrame:
    """Train the vanilla AE on the train segment only; encode all days frozen."""
    mu = returns_raw[:t_split].mean(axis=0, keepdims=True)
    sd = returns_raw[:t_split].std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    xz = (returns_raw - mu) / sd
    cfg = AETrainConfig(
        n_factors=factor_cfg.n_factors,
        epochs=factor_cfg.epochs,
        batch_size=factor_cfg.batch_size,
        learning_rate=factor_cfg.learning_rate,
        patience=factor_cfg.patience,
        seed=seed,
        activation=factor_cfg.activation,
    )
    log(
        f"  AE(leak-free): K={cfg.n_factors} activation={cfg.activation} seed={seed} "
        f"trained on first {t_split} days, frozen encode of all {len(dates)}"
    )
    model, _ = train_vanilla_autoencoder(xz[:t_split], cfg)
    import torch

    model.eval()
    with torch.no_grad():
        factors = model.encode(torch.tensor(xz, dtype=torch.float32)).cpu().numpy()
    F = pd.DataFrame(
        factors.astype(np.float64),
        index=dates,
        columns=[f"f{i + 1}" for i in range(cfg.n_factors)],
    )
    F.index.name = "date"
    return F


def _standardized_weights(y_h_train: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """1/train-variance weights on the clean panel, zero on stale series."""
    var = np.nanvar(y_h_train, axis=0, ddof=0)
    var = np.where(var <= 0, np.nan, var)
    w = np.where(clean, 1.0 / var, 0.0)
    w = np.nan_to_num(w, nan=0.0)
    if w.sum() > 0:
        w = w * (clean.sum() / w.sum() * (len(w) / max(clean.sum(), 1)))
    return w


def _accuracy_rows(
    fr: SubsetForecasts,
    full_key: tuple[int, ...],
    weights: np.ndarray,
    commodities: list[str],
    stale: np.ndarray,
    zero_frac: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """Per-commodity accuracy table and pooled metrics for one horizon."""
    r2_ar = r2_oos_per_commodity(fr, full_key, AR_KEY)
    r2_mean = r2_oos_per_commodity(fr, full_key, MEAN_KEY)
    r2_zero = r2_oos_per_commodity(fr, full_key, ZERO_KEY)
    cw_stat, cw_p = clark_west_per_commodity(fr, full_key, AR_KEY)
    fdr = bh_fdr(cw_p, q=0.10)
    df = pd.DataFrame(
        {
            "commodity": commodities,
            "stale": stale,
            "zero_return_frac": zero_frac,
            "mse_full": fr.sqerr(full_key).mean(axis=0),
            "mse_ar1": fr.sqerr(AR_KEY).mean(axis=0),
            "mse_mean": fr.sqerr(MEAN_KEY).mean(axis=0),
            "mse_zero": fr.sqerr(ZERO_KEY).mean(axis=0),
            "r2_oos_vs_ar1": r2_ar,
            "r2_oos_vs_mean": r2_mean,
            "r2_oos_vs_zero": r2_zero,
            "cw_stat": cw_stat,
            "cw_p": cw_p,
            "cw_fdr10_reject": fdr,
        }
    )
    clean_idx = np.where(~stale)[0]
    fr_clean = fr.select(clean_idx)
    w_clean = weights[clean_idx]
    cw_pool = clark_west(fr_clean, full_key, AR_KEY, weights=w_clean)
    pooled = {
        "horizon": fr.horizon,
        "n_origins": len(fr.origins),
        "effective_n": len(fr.origins) // fr.horizon,
        "hac_lags": default_hac_lags(fr.horizon),
        "r2_pool_std_vs_ar1": r2_oos(fr_clean, full_key, AR_KEY, w_clean),
        "r2_pool_std_vs_mean": r2_oos(fr_clean, full_key, MEAN_KEY, w_clean),
        "r2_pool_std_vs_zero": r2_oos(fr_clean, full_key, ZERO_KEY, w_clean),
        "r2_pool_raw_vs_ar1": r2_oos(fr, full_key, AR_KEY),
        "r2_ar1_vs_zero": r2_oos(fr_clean, AR_KEY, ZERO_KEY, w_clean),
        "cw_pool_stat": cw_pool[0],
        "cw_pool_p": cw_pool[1],
        "n_cw_fdr10": int(fdr.sum()),
    }
    return df, pooled


def run_forecast_pbsv(args: argparse.Namespace) -> None:
    """Run the full forecast-PBSV pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg: ForecastPBSVConfig = load_forecast_pbsv_config(args.config)
    factor_cfg = load_factor_model_config(args.factor_config)
    seed = args.seed

    out = Path(args.outdir)
    res = out / "results" / "forecast_pbsv"
    figs = out / "figures" / "forecast_pbsv"
    rep = out / "reports"
    for d in (res, figs, rep):
        d.mkdir(parents=True, exist_ok=True)
    log = RunLogger()
    acceptance: list[tuple[str, bool, str]] = []

    # ---------------------------------------------------------------- data
    panel = load_return_panel()
    returns = panel.raw  # (T, N) raw log returns
    dates = panel.dates
    commodities = list(panel.commodities)
    T, N = returns.shape
    t_split = int(cfg.train_frac * T)
    oos_start = t_split
    log(
        f"calendar: T={T} N={N} ({dates[0].date()} -> {dates[-1].date()}); "
        f"train_frac={cfg.train_frac} -> split at {dates[t_split].date()} "
        f"(train {t_split}, OOS {T - t_split} days); COVID crash is IN TRAIN"
    )

    zero_frac, longest_run = staleness_stats(returns)
    stale = zero_frac > cfg.stale_zero_frac_max
    log(
        f"staleness: {int(stale.sum())} series above zero-frac {cfg.stale_zero_frac_max}: "
        f"{[c for c, s in zip(commodities, stale) if s]} (headline targets exclude them)"
    )

    manifest = pd.read_csv(args.manifest)
    M_full = load_macro_stationary(args.macro, manifest)

    # ------------------------------------------------- factors + frozen basis
    F = train_leakfree_factors(returns, dates, t_split, factor_cfg, seed, log)
    F_train = F.iloc[:t_split]
    M_train = M_full.loc[M_full.index < dates[t_split]]
    basis = fit_frozen_basis(
        F_train,
        M_train,
        ridge_grid=cfg.ridge_grid,
        n_folds=cfg.n_folds,
        embargo=cfg.embargo,
        n_perm=cfg.n_perm_basis,
        seed=seed,
        spanned_rho_min=cfg.spanned_rho_min,
        spanned_p_max=cfg.spanned_p_max,
    )
    K = basis.n_dims
    full_key = tuple(range(K))
    groups = basis.groups
    log(
        f"basis: ridge_m={basis.ridge_m} (ridge_f=0 pinned) cond(F)={basis.factor_cond:.1f} "
        f"rho_train={np.round(basis.rho_train, 3)}"
    )
    log(
        f"basis: rho_cv_train={np.round(basis.rho_cv_train, 3)} perm_p={basis.perm_p_train} "
        f"-> spanned block = cv1..cv{basis.n_spanned} "
        f"(weakly-macro-correlated: cv{basis.n_spanned + 1}..cv{K})"
    )

    V = basis.variates(F)  # (T, K) canonical-variate state on the returns calendar
    state = V.to_numpy(float)

    F_oos = F.iloc[t_split:]
    M_oos = M_full.loc[M_full.index >= dates[t_split]]
    basis_diag = frozen_oos_diagnostics(basis, F_oos, M_oos)
    basis_diag["top_loadings_train"] = [
        "; ".join(f"{n}({v:+.2f})" for n, v in basis.top_loadings(k)) for k in range(K)
    ]
    basis_diag.to_csv(res / "basis_summary.csv", index=False)
    log("basis OOS(frozen): " + basis_diag.to_string(index=False, max_colwidth=40))

    stab = basis_stability_bootstrap(
        F_train, M_train, basis, n_boot=cfg.n_boot_basis, mean_block=cfg.mean_block, seed=seed
    )
    stab.to_csv(res / "basis_stability.csv", index=False)
    ang_med = float(stab["max_principal_angle_deg"].median())
    bnd_match = float(stab["boundary_matches"].mean())
    log(
        f"basis stability (train bootstrap, n={cfg.n_boot_basis}): median max principal angle "
        f"of spanned subspace = {ang_med:.1f} deg; boundary reproduced in {bnd_match:.0%} of draws"
    )
    acceptance.append(
        (
            "spanned-subspace stable under train bootstrap (median angle < 30 deg)",
            ang_med < 30.0,
            f"median={ang_med:.1f} deg, boundary match {bnd_match:.0%}",
        )
    )

    # ------------------------------------------------------------ main runs
    accuracy_rows: list[pd.DataFrame] = []
    pooled_rows: list[dict] = []
    shapley_rows: list[pd.DataFrame] = []
    grouped_rows: list[dict] = []
    boundary_rows: list[dict] = []
    fr_headline: SubsetForecasts | None = None
    weights_by_h: dict[int, np.ndarray] = {}

    for h in cfg.horizons:
        y_h = h_period_returns(returns, h)
        weights = _standardized_weights(y_h[:t_split], ~stale)
        weights_by_h[h] = weights
        fr = expanding_subset_forecasts(
            returns, state, oos_start, horizon=h, min_train=cfg.min_train
        )
        acc, pooled = _accuracy_rows(fr, full_key, weights, commodities, stale, zero_frac)
        acc.insert(0, "horizon", h)
        accuracy_rows.append(acc)

        vol = ewma_vol(returns, cfg.ewma_lambda)
        ugain, sdiff = timing_utility_gain(
            fr.select(np.where(~stale)[0]),
            full_key,
            AR_KEY,
            vol[:, ~stale],
            gamma=cfg.utility_gamma,
        )
        pooled["utility_gain_ann"] = ugain
        pooled["sharpe_diff_ann"] = sdiff

        res_p = pbsv(fr, weights=weights, groups=groups)
        boot = bootstrap_pbsv(
            fr,
            weights=weights,
            groups=groups,
            mean_block=max(cfg.mean_block, 2 * h),
            n_boot=cfg.n_boot,
            seed=seed,
        )
        pooled["v_full_std"] = res_p["v_full"]
        pooled["v_full_boot_lo"] = float(
            np.percentile(boot["phi_draws"].sum(axis=1), 2.5)
        )
        pooled["v_full_boot_hi"] = float(
            np.percentile(boot["phi_draws"].sum(axis=1), 97.5)
        )
        pooled_rows.append(pooled)

        mse_ar_std = fr.mse(AR_KEY, weights)
        sh = pd.DataFrame(
            {
                "horizon": h,
                "dim": [f"cv{k + 1}" for k in range(K)],
                "spanned": [k < basis.n_spanned for k in range(K)],
                "phi_std": res_p["phi"],
                "phi_pct_of_ar1_mse": 100.0 * res_p["phi"] / mse_ar_std,
                "boot_lo": boot["phi_lo"],
                "boot_hi": boot["phi_hi"],
            }
        )
        shapley_rows.append(sh)

        gv = res_p["group_table"]
        grouped_rows.append(
            {
                "horizon": h,
                "phi_spanned": float(res_p["phi_grouped"][0]),
                "phi_weak": float(res_p["phi_grouped"][1]),
                "boot_lo_spanned": float(boot["phi_grouped_lo"][0]),
                "boot_hi_spanned": float(boot["phi_grouped_hi"][0]),
                "boot_lo_weak": float(boot["phi_grouped_lo"][1]),
                "boot_hi_weak": float(boot["phi_grouped_hi"][1]),
                "v_spanned": gv[(0,)],
                "v_weak": gv[(1,)],
                "v_full": gv[(0, 1)],
                "interaction": gv[(0, 1)] - gv[(0,)] - gv[(1,)],
            }
        )
        for row in boundary_sensitivity(res_p["values"], K):
            row["horizon"] = h
            boundary_rows.append(row)

        eff = efficiency_gap(res_p)
        acceptance.append(
            (f"h={h}: Shapley efficiency |sum(phi)-v(full)| < 1e-12", eff < 1e-12, f"gap={eff:.2e}")
        )
        log(
            f"h={h}: pooled std R2_OOS(full vs AR1)={pooled['r2_pool_std_vs_ar1']:+.5f} "
            f"CW p={pooled['cw_pool_p']:.4f}; v_full={res_p['v_full']:+.3e} "
            f"phi={np.round(res_p['phi'], 5)}"
        )

        if h == cfg.headline_horizon:
            fr_headline = fr

    assert fr_headline is not None
    h0 = cfg.headline_horizon
    w0 = weights_by_h[h0]

    pd.concat(accuracy_rows).to_csv(res / "forecast_accuracy.csv", index=False)
    pd.concat(shapley_rows).to_csv(res / "pbsv_shapley.csv", index=False)
    pd.DataFrame(grouped_rows).to_csv(res / "pbsv_grouped.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(res / "pbsv_boundary_sensitivity.csv", index=False)

    # --------------------------------------- signal concentration (LOCO CW)
    clean_idx = np.where(~stale)[0]
    loco_rows = []
    for drop_i in clean_idx:
        keep = np.array([i for i in clean_idx if i != drop_i])
        stat_i, p_i = clark_west(fr_headline.select(keep), full_key, AR_KEY, weights=w0[keep])
        loco_rows.append({"dropped": commodities[drop_i], "cw_stat": stat_i, "cw_p": p_i})
    loco = pd.DataFrame(loco_rows).sort_values("cw_p", ascending=False).reset_index(drop=True)
    loco.to_csv(res / "cw_leave_one_out.csv", index=False)
    loco_max_p = float(loco["cw_p"].iloc[0])
    loco_max_name = str(loco["dropped"].iloc[0])
    log(
        f"LOCO CW (h={h0}): weakest after dropping {loco_max_name}: p={loco_max_p:.4f} — "
        "pooled significance must survive removing its single most influential commodity"
    )
    w_loco = w0.copy()
    w_loco[commodities.index(loco_max_name)] = 0.0
    phi_loco = pbsv(fr_headline, weights=w_loco, groups=groups)["phi"]
    log(f"phi excluding {loco_max_name}: {np.round(phi_loco, 10)}")

    # ------------------------------------------------ robustness on headline h
    per_comm_phi = pbsv_per_commodity(fr_headline)
    pd.DataFrame(
        per_comm_phi, index=commodities, columns=[f"cv{k + 1}" for k in range(K)]
    ).to_csv(res / "pbsv_per_commodity.csv")

    target_years = np.asarray(pd.DatetimeIndex(dates[fr_headline.targets]).year)
    year_rows = []
    for yr in sorted(set(target_years)):
        mask = target_years == yr
        phi_y = pbsv_on_mask(fr_headline, mask, weights=w0)
        year_rows.append({"year": int(yr), "n_days": int(np.sum(mask)), **{
            f"phi_cv{k + 1}": float(phi_y[k]) for k in range(K)
        }})
    burn_mask = np.arange(len(fr_headline.origins)) >= cfg.burn_in
    phi_burn = pbsv_on_mask(fr_headline, burn_mask, weights=w0)
    year_rows.append({"year": -1, "n_days": int(burn_mask.sum()), **{
        f"phi_cv{k + 1}": float(phi_burn[k]) for k in range(K)
    }})
    pd.DataFrame(year_rows).to_csv(res / "pbsv_by_year.csv", index=False)

    res_raw_w = pbsv(fr_headline, weights=None, groups=groups)
    pd.DataFrame(
        {
            "dim": [f"cv{k + 1}" for k in range(K)],
            "phi_std_weighted": pbsv(fr_headline, weights=w0)["phi"],
            "phi_raw_pooled": res_raw_w["phi"],
            f"phi_excl_{loco_max_name}": phi_loco,
        }
    ).to_csv(res / "pbsv_weighting_check.csv", index=False)

    # controls arm: EWMA vol + momentum in benchmark AND full model
    vol = ewma_vol(returns, cfg.ewma_lambda)
    mom = h_period_returns(returns, cfg.momentum_lookback)
    controls = np.stack([vol, np.nan_to_num(mom, nan=0.0)], axis=2)
    fr_ctrl = expanding_subset_forecasts(
        returns, state, oos_start, horizon=h0, controls=controls, min_train=cfg.min_train
    )
    res_ctrl = pbsv(fr_ctrl, weights=w0, groups=groups)
    pd.DataFrame(
        {
            "dim": [f"cv{k + 1}" for k in range(K)],
            "phi_no_controls": pbsv(fr_headline, weights=w0)["phi"],
            "phi_vol_mom_controls": res_ctrl["phi"],
        }
    ).to_csv(res / "pbsv_controls.csv", index=False)
    log(
        f"controls arm: v_full {pbsv(fr_headline, weights=w0)['v_full']:+.3e} -> "
        f"{res_ctrl['v_full']:+.3e} with EWMA-vol+12m-momentum in both models"
    )

    # ------------------------------------------------------------- placebo
    def _placebo_cw(fr_b: SubsetForecasts) -> float:
        return clark_west(fr_b.select(clean_idx), full_key, AR_KEY, weights=w0[clean_idx])[0]

    plc = placebo_pbsv(
        returns,
        state,
        oos_start,
        horizon=h0,
        weights=w0,
        groups=groups,
        min_train=cfg.min_train,
        n_placebo=cfg.n_placebo,
        min_shift=cfg.placebo_min_shift,
        seed=seed,
        stat_fn=_placebo_cw,
    )
    res_head = pbsv(fr_headline, weights=w0, groups=groups)
    n_plc = plc["phi_draws"].shape[0]
    p_right = (1 + (plc["phi_draws"] >= res_head["phi"][None, :]).sum(0)) / (n_plc + 1)
    p_left = (1 + (plc["phi_draws"] <= res_head["phi"][None, :]).sum(0)) / (n_plc + 1)
    plc_df = pd.DataFrame(
        {
            "dim": [f"cv{k + 1}" for k in range(K)],
            "phi": res_head["phi"],
            "placebo_lo": plc["phi_lo"],
            "placebo_hi": plc["phi_hi"],
            "placebo_abs_p95": plc["abs_p95"],
            "placebo_p_right": p_right,
            "placebo_p_left": p_left,
            "outside_band": (res_head["phi"] < plc["phi_lo"]) | (res_head["phi"] > plc["phi_hi"]),
        }
    )
    head_cw_stat = float(
        next(p["cw_pool_stat"] for p in pooled_rows if p["horizon"] == h0)
    )
    cw_placebo_p = float(
        (1 + np.sum(plc["stat_draws"] >= head_cw_stat)) / (len(plc["stat_draws"]) + 1)
    )
    log(
        f"placebo-calibrated pooled CW: observed stat={head_cw_stat:.3f}, "
        f"placebo mean={plc['stat_draws'].mean():.3f} sd={plc['stat_draws'].std():.3f}, "
        f"empirical p={cw_placebo_p:.4f}"
    )
    plc_grouped = pd.DataFrame(
        {
            "group": ["spanned", "weak"],
            "phi": res_head["phi_grouped"],
            "placebo_lo": plc["phi_grouped_lo"],
            "placebo_hi": plc["phi_grouped_hi"],
            "outside_band": (res_head["phi_grouped"] < plc["phi_grouped_lo"])
            | (res_head["phi_grouped"] > plc["phi_grouped_hi"]),
        }
    )
    plc_df.to_csv(res / "pbsv_placebo.csv", index=False)
    plc_grouped.to_csv(res / "pbsv_placebo_grouped.csv", index=False)
    log("placebo bands (jointly shifted state, cardinality-matched):")
    log(plc_df.to_string(index=False))

    # ------------------------------------------- raw-basis diagnostic + seeds
    f_sd = F.iloc[:t_split].std(ddof=0).to_numpy()
    raw_state = (F.to_numpy(float) - basis.mu_f) / np.where(f_sd == 0, 1.0, f_sd)
    fr_raw = expanding_subset_forecasts(
        returns, raw_state, oos_start, horizon=h0, min_train=cfg.min_train
    )
    inv_gap = abs(fr_raw.mse(full_key, w0) - fr_headline.mse(full_key, w0))
    rel_gap = inv_gap / fr_headline.mse(full_key, w0)
    acceptance.append(
        (
            "basis invariance: full-model MSE identical in raw vs CV basis (<1e-8 rel)",
            rel_gap < 1e-8,
            f"rel gap={rel_gap:.2e}",
        )
    )

    seed_rows = []
    raw_phi_rows = []
    F_by_seed: dict[int, pd.DataFrame] = {}
    for s_i, sd_seed in enumerate(cfg.stability_seeds):
        try:
            F_s = (
                F
                if sd_seed == seed
                else train_leakfree_factors(returns, dates, t_split, factor_cfg, sd_seed, log)
            )
            basis_s = (
                basis
                if sd_seed == seed
                else fit_frozen_basis(
                    F_s.iloc[:t_split],
                    M_train,
                    ridge_grid=cfg.ridge_grid,
                    n_folds=cfg.n_folds,
                    embargo=cfg.embargo,
                    n_perm=max(cfg.n_perm_basis // 4, 50),
                    seed=sd_seed,
                    spanned_rho_min=cfg.spanned_rho_min,
                    spanned_p_max=cfg.spanned_p_max,
                )
            )
        except ValueError as exc:
            log(f"seed {sd_seed}: basis fit failed ({exc}); skipped")
            continue
        F_by_seed[sd_seed] = F_s
        state_s = basis_s.variates(F_s).to_numpy(float)
        fr_s = (
            fr_headline
            if sd_seed == seed
            else expanding_subset_forecasts(
                returns, state_s, oos_start, horizon=h0, min_train=cfg.min_train
            )
        )
        groups_s = basis_s.groups
        res_s = pbsv(fr_s, weights=w0, groups=groups_s)
        f0z = (F.iloc[:t_split] - F.iloc[:t_split].mean()) / F.iloc[:t_split].std(ddof=0)
        fsz = (F_s.iloc[:t_split] - F_s.iloc[:t_split].mean()) / F_s.iloc[:t_split].std(ddof=0)
        ang_space = np.degrees(np.max(principal_angles(f0z.to_numpy(), fsz.to_numpy())))
        v0 = basis.variates(F.iloc[:t_split]).to_numpy()[:, : basis.n_spanned]
        vs = basis_s.variates(F_s.iloc[:t_split]).to_numpy()[:, : basis_s.n_spanned]
        ang_span = np.degrees(np.max(principal_angles(v0, vs))) if basis_s.n_spanned else np.nan
        seed_rows.append(
            {
                "seed": sd_seed,
                "n_spanned": basis_s.n_spanned,
                "v_full_std": res_s["v_full"],
                "phi_spanned": float(res_s["phi_grouped"][0]),
                "phi_weak": float(res_s["phi_grouped"][1]),
                "max_angle_factor_space_deg": float(ang_space),
                "max_angle_spanned_plane_deg": float(ang_span),
            }
        )
        raw_state_s = F_s.to_numpy(float) - F_s.iloc[:t_split].mean().to_numpy()
        fr_raw_s = (
            fr_raw
            if sd_seed == seed
            else expanding_subset_forecasts(
                returns, raw_state_s, oos_start, horizon=h0, min_train=cfg.min_train
            )
        )
        phi_raw_s = pbsv(fr_raw_s, weights=w0)["phi"]
        raw_phi_rows.append(
            {"seed": sd_seed, **{f"phi_raw_f{k + 1}": float(phi_raw_s[k]) for k in range(K)}}
        )
    pd.DataFrame(seed_rows).to_csv(res / "seed_stability.csv", index=False)
    pd.DataFrame(raw_phi_rows).to_csv(res / "pbsv_raw_basis_by_seed.csv", index=False)

    # ------------------------------------------------------ substitution arm
    M_sub = M_full.copy()
    for col in cfg.lagged_macro_series:
        if col in M_sub.columns:
            M_sub[col] = M_sub[col].shift(1)
    M_sub = M_sub.dropna(how="any")
    U = basis.macro_variates(M_sub)
    U = U.reindex(dates)
    macro_avail = U.notna().all(axis=1).to_numpy()
    rho_proxy = basis.rho_train  # train-scale for the EIV proxy rho_k * u_k
    U_arr = np.nan_to_num(U.to_numpy(float), nan=0.0)

    def substituted_state(sub_dims: tuple[int, ...]) -> np.ndarray:
        s = state.copy()
        for k in sub_dims:
            s[:, k] = rho_proxy[k] * U_arr[:, k]
        return s

    arms: list[tuple[str, tuple[int, ...]]] = [("none", ())]
    arms += [(f"only_cv{k + 1}", (k,)) for k in range(K)]
    arms += [
        ("spanned_block", groups[0]),
        ("weak_block", groups[1]),
        ("all", full_key),
    ]
    sub_rows = []
    v_none = None
    for name, sub_dims in arms:
        fr_a = expanding_subset_forecasts(
            returns,
            substituted_state(sub_dims),
            oos_start,
            horizon=h0,
            subsets=[AR_KEY, full_key],
            available=macro_avail,
            min_train=cfg.min_train,
        )
        v_a = fr_a.mse(AR_KEY, w0) - fr_a.mse(full_key, w0)
        cw_a = clark_west(fr_a.select(np.where(~stale)[0]), full_key, AR_KEY, weights=w0[~stale])
        if name == "none":
            v_none = v_a
        retained = v_a / v_none if (v_none is not None and v_none != 0.0) else np.nan
        sub_rows.append(
            {
                "arm": name,
                "substituted_dims": ".".join(str(k + 1) for k in sub_dims) or "-",
                "v_full_std": v_a,
                "retained_share_vs_none": retained,
                "cw_p": cw_a[1],
                "n_origins": len(fr_a.origins),
            }
        )
        log(f"substitution[{name}]: v_full={v_a:+.3e} (n={len(fr_a.origins)})")
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(res / "macro_substitution.csv", index=False)
    n_dropped = int((~macro_avail[oos_start:]).sum())
    log(
        f"substitution calendar: macro-available OOS days only; {n_dropped} OOS days dropped; "
        f"gpr/epu lagged 1 day (publication lag); ALL arms share the calendar"
    )

    # ----------------------------------------------------- no-lookahead probe
    probe = oos_start + min(100, (T - oos_start) // 2)
    rng = np.random.default_rng(seed)
    returns_pert = returns.copy()
    state_pert = state.copy()
    returns_pert[probe + h0 + 1 :] = rng.standard_normal(returns[probe + h0 + 1 :].shape)
    state_pert[probe + h0 + 1 :] = rng.standard_normal(state[probe + h0 + 1 :].shape)
    fr_pert = expanding_subset_forecasts(
        returns_pert,
        state_pert,
        oos_start,
        horizon=h0,
        subsets=[AR_KEY, full_key],
        min_train=cfg.min_train,
    )
    common = fr_headline.origins <= probe
    common_p = fr_pert.origins <= probe
    same = np.allclose(
        fr_headline.preds[fr_headline.key_index(full_key)][common],
        fr_pert.preds[fr_pert.key_index(full_key)][common_p],
        atol=1e-12,
        equal_nan=True,
    )
    acceptance.append(
        (
            f"no-lookahead: forecasts at origins <= {probe} unchanged when later data scrambled",
            bool(same),
            f"probe origin index {probe} (h={h0})",
        )
    )

    # ------------------------------------------------------------------ gate
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df["cw_placebo_p"] = np.where(pooled_df["horizon"] == h0, cw_placebo_p, np.nan)
    pooled_df["cw_loco_max_p"] = np.where(pooled_df["horizon"] == h0, loco_max_p, np.nan)
    pooled_df.to_csv(res / "forecast_accuracy_pooled.csv", index=False)
    head = pooled_df[pooled_df["horizon"] == h0].iloc[0]
    gate = (
        (head["cw_pool_p"] < 0.05)
        and (loco_max_p < 0.05)
        and (cw_placebo_p < 0.05)
        and (head["v_full_boot_lo"] > 0)
        and (head["r2_pool_std_vs_zero"] > 0)
    )
    log(
        f"GATE (share-of-gain language): CW p={head['cw_pool_p']:.4f} (<0.05: "
        f"{head['cw_pool_p'] < 0.05}), LOCO max p={loco_max_p:.4f} (dropping {loco_max_name}), "
        f"placebo-calibrated CW p={cw_placebo_p:.4f}, "
        f"v_full CI=[{head['v_full_boot_lo']:.2e},{head['v_full_boot_hi']:.2e}] excludes 0: "
        f"{head['v_full_boot_lo'] > 0}, beats zero: {head['r2_pool_std_vs_zero'] > 0} "
        f"-> gate_passed={gate}"
    )

    make_forecast_figures(
        figs,
        shapley=pd.concat(shapley_rows),
        placebo=plc_df,
        grouped=pd.DataFrame(grouped_rows),
        by_year=pd.DataFrame(year_rows),
        per_commodity=pd.DataFrame(
            per_comm_phi, index=commodities, columns=[f"cv{k + 1}" for k in range(K)]
        ),
        substitution=sub_df,
        headline_horizon=h0,
    )

    write_forecast_report(
        rep,
        cfg=cfg,
        seed=seed,
        split_date=str(dates[t_split].date()),
        basis_diag=basis_diag,
        basis=basis,
        pooled=pooled_df,
        shapley=pd.concat(shapley_rows),
        grouped=pd.DataFrame(grouped_rows),
        boundary=pd.DataFrame(boundary_rows),
        placebo=plc_df,
        placebo_grouped=plc_grouped,
        controls=pd.read_csv(res / "pbsv_controls.csv"),
        substitution=sub_df,
        seed_stability=pd.DataFrame(seed_rows),
        raw_phi=pd.DataFrame(raw_phi_rows),
        accuracy=pd.concat(accuracy_rows),
        by_year=pd.DataFrame(year_rows),
        gate_passed=bool(gate),
        acceptance=acceptance,
        stale_names=[c for c, s in zip(commodities, stale) if s],
        n_sub_dropped=n_dropped,
        loco=loco,
        cw_placebo_p=cw_placebo_p,
        phi_loco=phi_loco,
        loco_max_name=loco_max_name,
    )

    log.write(res / "run_log.txt")
    print("\n=== ACCEPTANCE ===")
    for name, ok, detail in acceptance:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"\nDone. Outputs under {res} , {figs} , {rep}")
