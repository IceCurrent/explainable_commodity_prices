"""Markdown report for the forecast-PBSV pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqcp.config import ForecastPBSVConfig
from eqcp.forecasting.basis import FrozenBasis

LOADING_COS_MIN = 0.8  # train-era macro names are used only above this OOS drift cosine


def _md_table(df: pd.DataFrame, floatfmt: str = "{:+.4g}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v) if np.isfinite(v) else "nan"
        return str(v)

    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def write_forecast_report(
    rep: Path,
    cfg: ForecastPBSVConfig,
    seed: int,
    split_date: str,
    basis_diag: pd.DataFrame,
    basis: FrozenBasis,
    pooled: pd.DataFrame,
    shapley: pd.DataFrame,
    grouped: pd.DataFrame,
    boundary: pd.DataFrame,
    placebo: pd.DataFrame,
    placebo_grouped: pd.DataFrame,
    controls: pd.DataFrame,
    substitution: pd.DataFrame,
    transmission: pd.DataFrame,
    seed_stability: pd.DataFrame,
    raw_phi: pd.DataFrame,
    accuracy: pd.DataFrame,
    by_year: pd.DataFrame,
    gate_passed: bool,
    acceptance: list[tuple[str, bool, str]],
    stale_names: list[str],
    n_sub_dropped: int,
    loco: pd.DataFrame,
    cw_placebo_p: float,
    phi_loco: np.ndarray,
    loco_max_name: str,
    model_type: str = "vanilla",
) -> None:
    """Write reports/forecast_pbsv_report.md (or _beta_ for the beta-VAE arm)."""
    h0 = cfg.headline_horizon
    head = pooled[pooled["horizon"] == h0].iloc[0]
    sh0 = shapley[shapley["horizon"] == h0]
    g0 = grouped[grouped["horizon"] == h0].iloc[0]

    named = []
    for _, row in basis_diag.iterrows():
        ok = row["loading_cosine_oos"] >= LOADING_COS_MIN and row["spanned"]
        named.append(row["top_loadings_train"] if ok else "(train-era name not licensed OOS)")

    lines: list[str] = []
    a = lines.append
    a("# Forecast-PBSV Report — Factor-Augmented AR(1) + Forecast-Based Shapley Values")
    a("")
    a("## Verdict")
    a("")
    v_ci = f"[{head['v_full_boot_lo']:.2e}, {head['v_full_boot_hi']:.2e}]"
    loco_top = loco.iloc[0]
    if gate_passed:
        a(
            f"The factor state adds OOS forecasting value at h={h0}: pooled Clark–West "
            f"p={head['cw_pool_p']:.4f} (placebo-calibrated p={cw_placebo_p:.4f}, robust to "
            f"dropping any single commodity: max LOCO p={loco_top['cw_p']:.4f}), v(full) "
            f"bootstrap CI {v_ci} excludes 0, and the full model beats the zero forecast. "
            f"Share-of-gain attribution below is licensed."
        )
    else:
        a(
            f"**Well-identified null.** At h={h0} the factor-augmented AR(1) does NOT deliver "
            f"an economically meaningful OOS improvement over AR(1): "
            f"v(full)={head['v_full_std']:+.3e} with bootstrap CI {v_ci}; "
            f"R2_OOS vs zero {head['r2_pool_std_vs_zero']:+.5f}. Pooled Clark–West "
            f"p={head['cw_pool_p']:.4f}, but dropping the single most influential commodity "
            f"({loco_top['dropped']}) moves it to p={loco_top['cw_p']:.4f} — the rejection is "
            f"signal CONCENTRATION, not breadth (see leave-one-out table). Percent-of-gain "
            f"language is therefore NOT licensed; phi values below are absolute, with placebo "
            f"bands showing what pure estimation cost produces at matched cardinality."
        )
    a("")
    a(
        f"Setup: leak-free (train-only) AE, split at **{split_date}** "
        f"(train_frac={cfg.train_frac}; COVID crash in train), frozen canonical-variate basis "
        f"(factor-side ridge pinned to 0, macro-side ridge={basis.ridge_m}), "
        f"headline targets exclude stale series {stale_names} "
        f"(zero-return fraction > {cfg.stale_zero_frac_max}); standardized-loss pooling."
    )
    a("")
    gate_h = [int(r["horizon"]) for _, r in transmission.iterrows() if bool(r["gate_passed"])]
    horizons_txt = ", ".join(f"h={int(h)}" for h in transmission["horizon"])
    a(
        f"**Horizon ladder (the point of this study).** The same test was run at daily, weekly, "
        f"monthly and quarterly horizons ({horizons_txt}). The share-of-gain gate passes at: "
        f"**{('; '.join(f'h={h}' for h in gate_h)) if gate_h else 'NO horizon'}**. "
        + (
            "Macro moves commodities contemporaneously (the CCA spanning map), but that content "
            "is not converted into out-of-sample forecast power even at monthly/quarterly "
            "horizons — the EMH-coherent reading. See the horizon table below."
            if not gate_h
            else "See the horizon table below for where and how much."
        )
    )
    a("")

    a("## Forecast accuracy (clean panel, standardized pooling)")
    a("")
    cols = [
        "horizon",
        "n_origins",
        "effective_n",
        "r2_pool_std_vs_ar1",
        "r2_pool_std_vs_zero",
        "r2_ar1_vs_zero",
        "cw_pool_p",
        "cw_nonoverlap_p",
        "n_cw_fdr10",
        "utility_gain_ann",
        "sharpe_diff_ann",
    ]
    a(_md_table(pooled[cols]))
    a("")
    a(
        "R2_OOS columns compare the full factor model to each benchmark; `r2_ar1_vs_zero` shows "
        "whether the AR(1) baseline itself beats doing nothing. `cw_pool_p` uses all "
        "(overlapping) origins with a HAC bandwidth >= 2h; `cw_nonoverlap_p` re-runs Clark–West "
        "on targets spaced >= h apart (autocorrelation-free but lower power), the honest "
        "long-horizon check. `n_cw_fdr10` counts commodities whose per-commodity Clark–West "
        "rejects at BH-FDR 10%. `effective_n = n_origins / h` is the honest sample size."
    )
    a("")

    a("## Macro transmission across horizons")
    a("")
    a(
        "The central question — *do macros move commodities?* — is answered as a function of "
        "forecast horizon. For each horizon the full factor state (whose macro-spanned block is "
        "identified train-only) is scored against AR(1) out-of-sample, and the "
        "macro-substitution arm measures how much of any gain is macro-transmissible "
        "(errors-in-variables lower bound). `gate_passed` = the pre-registered share-of-gain "
        "gate (pooled CW<0.05, LOCO-robust, placebo-calibrated CW<0.05, v(full) bootstrap CI>0, "
        "beats zero)."
    )
    a("")
    tr_cols = [
        "horizon",
        "effective_n",
        "r2_oos_vs_ar1",
        "cw_p_overlap",
        "cw_p_nonoverlap",
        "cw_placebo_p",
        "loco_max_p",
        "v_full_std",
        "phi_spanned",
        "spanned_outside_band",
        "retained_share_spanned",
        "gate_passed",
    ]
    a(_md_table(transmission[tr_cols]))
    a("")
    a(
        "`retained_share_spanned` is interpretable only where `gate_passed` is true (otherwise it "
        "is a ratio of statistical zeros). `spanned_outside_band` flags whether the "
        "spanned-block PBSV clears its cardinality-matched zero-signal placebo band. See "
        "`figures/forecast_pbsv/transmission_by_horizon.png` and "
        "`results/forecast_pbsv/macro_transmission_by_horizon.csv`."
    )
    a("")

    a("## The attribution basis (train-frozen canonical variates)")
    a("")
    bd = basis_diag.copy()
    bd["macro_identity"] = named
    a(
        _md_table(
            bd[
                [
                    "dim",
                    "rho_cv_train",
                    "perm_p_train",
                    "rho_oos_frozen",
                    "loading_cosine_oos",
                    "spanned",
                    "macro_identity",
                ]
            ]
        )
    )
    a("")
    a(
        f"Spanned block = V1..V{basis.n_spanned} by the largest adjacent gap in the train "
        f"purged-CV rho spectrum. `rho_oos_frozen` is the strictly-forward correlation of frozen "
        f"factor- and macro-side variates over the OOS segment — the honest number for composite "
        f"claims. Macro names are shown only where the OOS loading cosine >= "
        f"{LOADING_COS_MIN} (loading drift check); the trailing block is 'weakly "
        f"macro-correlated', not 'orphaned' — all dims can reject the permutation null while "
        f"having small rho. Shared-variance: rho^2 of the spanned dims is "
        + ", ".join(
            f"{row['rho_oos_frozen'] ** 2:.0%}" for _, row in bd.iterrows() if row["spanned"]
        )
        + " — even 'spanned' directions are majority non-macro variance."
    )
    a("")

    a(f"## PBSV (h={h0}, exact 2^K Shapley, standardized pooling)")
    a("")
    plc_cols = ["dim", "placebo_lo", "placebo_hi", "outside_band"]
    if "placebo_p_right" in placebo.columns:
        plc_cols.insert(3, "placebo_p_right")
    m = sh0.merge(placebo[plc_cols], on="dim")
    a(
        _md_table(
            m[
                [
                    "dim",
                    "spanned",
                    "phi_std",
                    "boot_lo",
                    "boot_hi",
                    "placebo_lo",
                    "placebo_hi",
                ]
                + (["placebo_p_right"] if "placebo_p_right" in m.columns else [])
                + ["outside_band"]
            ]
        )
    )
    a("")
    a(
        "Bootstrap CIs are descriptive and conditional on the frozen basis and realized "
        "parameter path; significance belongs to the Clark–West tests. phi inside the placebo "
        "band is indistinguishable from pure estimation cost at matched subset cardinality and "
        "must not be interpreted. phi for the trailing (near-tied rho) dims is individually "
        "non-identified — only the block sum is meaningful."
    )
    a("")
    a("### Grouped game (rotation-invariant blocks)")
    a("")
    a(
        f"- v(spanned)={g0['v_spanned']:+.3e}, v(weak)={g0['v_weak']:+.3e}, "
        f"v(full)={g0['v_full']:+.3e}, interaction={g0['interaction']:+.3e}"
    )
    a(
        f"- phi_spanned={g0['phi_spanned']:+.3e} CI[{g0['boot_lo_spanned']:+.3e},"
        f"{g0['boot_hi_spanned']:+.3e}]  placebo[{placebo_grouped.iloc[0]['placebo_lo']:+.3e},"
        f"{placebo_grouped.iloc[0]['placebo_hi']:+.3e}]"
    )
    a(
        f"- phi_weak={g0['phi_weak']:+.3e} CI[{g0['boot_lo_weak']:+.3e},"
        f"{g0['boot_hi_weak']:+.3e}]  placebo[{placebo_grouped.iloc[1]['placebo_lo']:+.3e},"
        f"{placebo_grouped.iloc[1]['placebo_hi']:+.3e}]"
    )
    a(
        "- Grouped Shapley is NOT the within-block sum of per-direction phi (no group "
        "consistency); the full coalition table above shows the interaction explicitly."
    )
    a("")
    a("### Boundary sensitivity")
    a("")
    a(_md_table(boundary[boundary["horizon"] == h0].drop(columns=["horizon"])))
    a("")

    a("## Robustness")
    a("")
    a("### Signal concentration (leave-one-commodity-out Clark–West)")
    a("")
    a(_md_table(loco.head(5)))
    a(
        "\nPooled significance that dies when one series is dropped is a data artifact, not a "
        "cross-sectional phenomenon — the usual culprit is residual staleness (forward-filled "
        "assessment series make next-day returns partly deterministic). Series-level staleness "
        "stats are in `forecast_accuracy.csv`."
    )
    a("")
    a(
        f"phi recomputed with **{loco_max_name} excluded**: "
        + ", ".join(f"V{k + 1}={v:+.3e}" for k, v in enumerate(phi_loco))
        + ". Any placebo-band exceedance in the headline phi table that does not survive this "
        "exclusion is attributable to that single series, not to the factor direction."
    )
    a("")
    a("### Vol/momentum controls (both models)")
    a("")
    a(_md_table(controls))
    a(
        "\nIf spanned-block phi survives only without vol controls, the 'macro' signal is a "
        "volatility-timing effect wearing a macro label (ReLU latents are partly vol proxies)."
    )
    a("")
    a("### Weighting and sub-period stability")
    a("")
    yr_cols = ["year", "n_days"] + [c for c in by_year.columns if c.startswith("phi_")]
    a(_md_table(by_year[yr_cols]))
    a(f"\n(year -1 = burn-in robustness: first {cfg.burn_in} OOS days dropped.)")
    a("")

    a("## Macro substitution (the only arm licensed to say 'explained by macro')")
    a("")
    a(_md_table(substitution))
    a("")
    a(
        f"All arms (including 'none') share the macro-available calendar ({n_sub_dropped} OOS "
        f"days dropped); gpr/epu lagged one day for publication realism. Replacing v_k by "
        f"rho_k u_k is an errors-in-variables proxy discarding the (1-rho^2) variance share, so "
        f"retained shares are LOWER BOUNDS on macro-transmissibility"
        + (
            "."
            if gate_passed
            else " — and with the gate failed, retained shares are ratios of statistical "
            "zeros and are reported for completeness only."
        )
    )
    a("")
    a(
        "**Timing caveat.** US macro closes (FX/rates ~17:00 ET, equities 16:00-16:15 ET) "
        "postdate the commodity settlements (LME ~12:00, COMEX ~13:30, CME/ICE ~14:30 ET), so "
        "same-day macro variates contain up to ~5 hours of information from INSIDE the "
        "day-(t+1) settlement-to-settlement return window. A significant substitution-arm "
        "Clark-West is therefore NOT evidence of lagged macro transmission: it collapses when "
        "the macro variates are lagged one additional day, and its per-commodity strength "
        "lines up with each contract's after-settle overlap hours (LME metals first). See "
        "`reports/deep_analysis_report.md`."
    )
    a("")

    a("## Identification diagnostics")
    a("")
    a("### AE-seed subspace stability (illustration, n_seeds="
      f"{len(seed_stability)})")
    a("")
    if len(seed_stability):
        a(_md_table(seed_stability))
    a("")
    a("### Raw-coordinate PBSV by seed (label arbitrariness, do not interpret)")
    a("")
    if len(raw_phi):
        a(_md_table(raw_phi))
    a("")
    a(
        "Raw AE coordinates differ across seeds by optimizer multiplicity plus the ReLU gauge "
        "(permutation x positive scaling), so per-coordinate phi has no cross-seed "
        "correspondence — shown only to demonstrate that raw-coordinate attribution is an "
        "attribution to arbitrary labels. The invariance theorem itself is exercised by the "
        "unit test (random invertible transform leaves full-model forecasts and CV-basis PBSV "
        "unchanged); across-seed grouped-phi dispersion measures AE estimation variance, not "
        "basis-invariance failure."
    )
    a("")

    a("## Pre-registered outcome classes")
    a("")
    a(
        "- **Spanned-block phi ~ 0, weak-block phi ~ 0 (gate failed)**: daily factor state "
        "carries no exploitable mean signal — consistent with near-efficient daily commodity "
        "futures; the macro-composition question is then moot at this horizon."
    )
    a(
        "- **Weak-block phi > placebo, spanned ~ 0**: EMH-coherent — contemporaneously priced "
        "macro content is exactly the least forecastable part of the state; predictability "
        "lives in non-macro directions."
    )
    a(
        "- **Spanned-block phi > placebo surviving vol controls AND substitution retains it**: "
        "the surprising outcome that would license 'macro-content forecasts commodities'."
    )
    a("")

    a("## Acceptance checks")
    a("")
    for name, ok, detail in acceptance:
        a(f"- [{'x' if ok else ' '}] {name} — {detail}")
    a("")
    a(
        f"*Deterministic given --seed={seed} (single platform/BLAS). Design disclosure: K=5, "
        f"ReLU, and the two-block grouping schema were motivated by the full-sample mapping "
        f"study; spanning labels here are re-derived train-only, but the schema choice itself "
        f"post-dates that evidence.*"
    )
    a("")

    name = (
        "forecast_pbsv_report.md"
        if model_type == "vanilla"
        else "forecast_pbsv_beta_report.md"
    )
    (rep / name).write_text("\n".join(lines))
