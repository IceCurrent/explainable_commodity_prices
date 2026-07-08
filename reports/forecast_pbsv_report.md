# Forecast-PBSV Report — Factor-Augmented AR(1) + Forecast-Based Shapley Values

## Verdict

**Well-identified null.** At h=1 the factor-augmented AR(1) does NOT deliver an economically meaningful OOS improvement over AR(1): v(full)=-2.031e-06 with bootstrap CI [-4.09e-06, -4.26e-07]; R2_OOS vs zero -0.00449. Pooled Clark–West p=0.7546, but dropping the single most influential commodity (Zinc) moves it to p=0.8002 — the rejection is signal CONCENTRATION, not breadth (see leave-one-out table). Percent-of-gain language is therefore NOT licensed; phi values below are absolute, with placebo bands showing what pure estimation cost produces at matched cardinality.

Setup: leak-free (train-only) AE, split at **2021-04-30** (train_frac=0.6; COVID crash in train), frozen canonical-variate basis (factor-side ridge pinned to 0, macro-side ridge=0.0), headline targets exclude stale series [] (zero-return fraction > 0.15); standardized-loss pooling.

**Horizon ladder (the point of this study).** The same test was run at daily, weekly, monthly and quarterly horizons (h=1, h=5, h=21, h=63). The share-of-gain gate passes at: **NO horizon**. Macro moves commodities contemporaneously (the CCA spanning map), but that content is not converted into out-of-sample forecast power even at monthly/quarterly horizons — the EMH-coherent reading. See the horizon table below.

## Forecast accuracy (clean panel, standardized pooling)

| horizon | n_origins | effective_n | r2_pool_std_vs_ar1 | r2_pool_std_vs_zero | r2_ar1_vs_zero | cw_pool_p | cw_nonoverlap_p | n_cw_fdr10 | utility_gain_ann | sharpe_diff_ann |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1148 | 1148 | -0.005062 | -0.004493 | +0.0005667 | +0.7546 | +0.7296 | 0 | -0.1156 | -0.5396 |
| 5 | 1148 | 229 | -0.004761 | -0.009149 | -0.004368 | +0.7764 | +0.1908 | 0 | -0.0433 | -0.006596 |
| 21 | 1148 | 54 | -0.002623 | -0.01761 | -0.01495 | +0.08193 | +0.4559 | 0 | -0.008541 | +0.2599 |
| 63 | 1148 | 18 | -0.01481 | -0.06278 | -0.04727 | +0.5221 | +0.4208 | 0 | -0.005297 | +0.1684 |

R2_OOS columns compare the full factor model to each benchmark; `r2_ar1_vs_zero` shows whether the AR(1) baseline itself beats doing nothing. `cw_pool_p` uses all (overlapping) origins with a HAC bandwidth >= 2h; `cw_nonoverlap_p` re-runs Clark–West on targets spaced >= h apart (autocorrelation-free but lower power), the honest long-horizon check. `n_cw_fdr10` counts commodities whose per-commodity Clark–West rejects at BH-FDR 10%. `effective_n = n_origins / h` is the honest sample size.

## Macro transmission across horizons

The central question — *do macros move commodities?* — is answered as a function of forecast horizon. For each horizon the full factor state (whose macro-spanned block is identified train-only) is scored against AR(1) out-of-sample, and the macro-substitution arm measures how much of any gain is macro-transmissible (errors-in-variables lower bound). `gate_passed` = the pre-registered share-of-gain gate (pooled CW<0.05, LOCO-robust, placebo-calibrated CW<0.05, v(full) bootstrap CI>0, beats zero).

| horizon | effective_n | r2_oos_vs_ar1 | cw_p_overlap | cw_p_nonoverlap | cw_placebo_p | loco_max_p | v_full_std | phi_spanned | spanned_outside_band | retained_share_spanned | gate_passed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1148 | -0.005062 | +0.7546 | +0.7296 | +0.7525 | +0.8002 | -2.031e-06 | -1.416e-07 | False | +0.6432 | False |
| 5 | 229 | -0.004761 | +0.7764 | +0.1908 | +0.6634 | +0.8413 | -9.187e-06 | -2.157e-06 | False | +0.9612 | False |
| 21 | 54 | -0.002623 | +0.08193 | +0.4559 | +0.1089 | +0.1199 | -1.876e-05 | -1.032e-05 | False | +1.32 | False |
| 63 | 18 | -0.01481 | +0.5221 | +0.4208 | +0.4158 | +0.6211 | -0.0003199 | -1.381e-05 | False | +0.9815 | False |

`retained_share_spanned` is interpretable only where `gate_passed` is true (otherwise it is a ratio of statistical zeros). `spanned_outside_band` flags whether the spanned-block PBSV clears its cardinality-matched zero-signal placebo band. See `figures/forecast_pbsv/transmission_by_horizon.png` and `results/forecast_pbsv/macro_transmission_by_horizon.csv`.

## The attribution basis (train-frozen canonical variates)

| dim | rho_cv_train | perm_p_train | rho_oos_frozen | loading_cosine_oos | spanned | macro_identity |
|---|---|---|---|---|---|---|
| cv1 | +0.6361 | +0.004975 | +0.5815 | +0.9355 | True | xle(+0.78); fx_usdcad(-0.69); mxwo(+0.67); be_5y(+0.67) |
| cv2 | +0.05527 | +0.04975 | +0.04624 | +0.1611 | False | (train-era name not licensed OOS) |
| cv3 | +0.1571 | +0.004975 | +0.1859 | +0.8517 | False | (train-era name not licensed OOS) |
| cv4 | +0.01311 | +0.3582 | +0.009809 | -0.2224 | False | (train-era name not licensed OOS) |
| cv5 | +0.07943 | +0.02488 | +0.03987 | +0.289 | False | (train-era name not licensed OOS) |

Spanned block = cv1..cv1 by the largest adjacent gap in the train purged-CV rho spectrum. `rho_oos_frozen` is the strictly-forward correlation of frozen factor- and macro-side variates over the OOS segment — the honest number for composite claims. Macro names are shown only where the OOS loading cosine >= 0.8 (loading drift check); the trailing block is 'weakly macro-correlated', not 'orphaned' — all dims can reject the permutation null while having small rho. Shared-variance: rho^2 of the spanned dims is 34% — even 'spanned' directions are majority non-macro variance.

## PBSV (h=1, exact 2^K Shapley, standardized pooling)

| dim | spanned | phi_std | boot_lo | boot_hi | placebo_lo | placebo_hi | placebo_p_right | outside_band |
|---|---|---|---|---|---|---|---|---|
| cv1 | True | -1.413e-07 | -6.186e-07 | +3.168e-07 | -3.32e-07 | +1.537e-07 | +0.5842 | False |
| cv2 | False | -7.892e-07 | -2.055e-06 | +4.501e-09 | -3.646e-07 | +8.812e-08 | +1 | True |
| cv3 | False | -3.897e-07 | -9.051e-07 | +9.941e-08 | -2.866e-07 | +9.197e-08 | +1 | True |
| cv4 | False | -4.604e-07 | -9.622e-07 | -9.627e-08 | -3.364e-07 | +7.544e-08 | +1 | True |
| cv5 | False | -2.506e-07 | -7.93e-07 | +2.519e-07 | -3.857e-07 | +4.015e-08 | +0.8614 | False |

Bootstrap CIs are descriptive and conditional on the frozen basis and realized parameter path; significance belongs to the Clark–West tests. phi inside the placebo band is indistinguishable from pure estimation cost at matched subset cardinality and must not be interpreted. phi for the trailing (near-tied rho) dims is individually non-identified — only the block sum is meaningful.

### Grouped game (rotation-invariant blocks)

- v(spanned)=-1.479e-07, v(weak)=-1.896e-06, v(full)=-2.031e-06, interaction=+1.250e-08
- phi_spanned=-1.416e-07 CI[-6.208e-07,+3.164e-07]  placebo[-3.315e-07,+1.532e-07]
- phi_weak=-1.890e-06 CI[-3.816e-06,-5.383e-07]  placebo[-1.036e-06,-1.058e-07]
- Grouped Shapley is NOT the within-block sum of per-direction phi (no group consistency); the full coalition table above shows the interaction explicitly.

### Boundary sensitivity

| boundary | phi_lead_block | phi_tail_block | v_lead | v_tail | interaction |
|---|---|---|---|---|---|
| 1|2.3.4.5 | -1.416e-07 | -1.89e-06 | -1.479e-07 | -1.896e-06 | +1.25e-08 |
| 1.2|3.4.5 | -9.298e-07 | -1.101e-06 | -9.594e-07 | -1.131e-06 | +5.917e-08 |
| 1.2.3|4.5 | -1.321e-06 | -7.099e-07 | -1.352e-06 | -7.409e-07 | +6.194e-08 |
| 1.2.3.4|5 | -1.779e-06 | -2.518e-07 | -1.778e-06 | -2.506e-07 | -2.501e-09 |

## Robustness

### Signal concentration (leave-one-commodity-out Clark–West)

| dropped | cw_stat | cw_p |
|---|---|---|
| Zinc | -0.8424 | +0.8002 |
| NaturalGas | -0.8255 | +0.7955 |
| Soybeans | -0.7441 | +0.7716 |
| Copper | -0.7308 | +0.7675 |
| Platinum | -0.7298 | +0.7672 |

Pooled significance that dies when one series is dropped is a data artifact, not a cross-sectional phenomenon — the usual culprit is residual staleness (forward-filled assessment series make next-day returns partly deterministic). Series-level staleness stats are in `forecast_accuracy.csv`.

phi recomputed with **Zinc excluded**: cv1=-2.331e-07, cv2=-7.721e-07, cv3=-3.746e-07, cv4=-4.423e-07, cv5=-2.244e-07. Any placebo-band exceedance in the headline phi table that does not survive this exclusion is attributable to that single series, not to the factor direction.

### Vol/momentum controls (both models)

| dim | phi_no_controls | phi_vol_mom_controls |
|---|---|---|
| cv1 | -1.413e-07 | -1.547e-07 |
| cv2 | -7.892e-07 | -6.966e-07 |
| cv3 | -3.897e-07 | -3.444e-07 |
| cv4 | -4.604e-07 | -4.593e-07 |
| cv5 | -2.506e-07 | -2.104e-07 |

If spanned-block phi survives only without vol controls, the 'macro' signal is a volatility-timing effect wearing a macro label (ReLU latents are partly vol proxies).

### Weighting and sub-period stability

| year | n_days | phi_cv1 | phi_cv2 | phi_cv3 | phi_cv4 | phi_cv5 |
|---|---|---|---|---|---|---|
| 2021 | 155 | -2.55e-08 | +3.867e-07 | +3.702e-07 | -6.754e-08 | -2.089e-07 |
| 2022 | 222 | -9.767e-07 | -3.438e-06 | -8.394e-07 | -9.927e-07 | -1.306e-06 |
| 2023 | 226 | -7.963e-08 | -5.53e-08 | -1.204e-07 | -2.661e-07 | -2.433e-07 |
| 2024 | 227 | +1.169e-07 | -2.046e-07 | -1.371e-07 | -4.768e-07 | -1.845e-07 |
| 2025 | 227 | +2.406e-07 | -3.686e-07 | -6.02e-07 | -1.659e-07 | +3.783e-07 |
| 2026 | 91 | -5.016e-08 | -6.593e-07 | -1.357e-06 | -1.007e-06 | +5.018e-07 |
| -1 | 1085 | -2.349e-07 | -9.137e-07 | -4.49e-07 | -4.557e-07 | -3.007e-07 |

(year -1 = burn-in robustness: first 63 OOS days dropped.)

## Macro substitution (the only arm licensed to say 'explained by macro')

| arm | substituted_dims | v_full_std | retained_share_vs_none | cw_p | n_origins |
|---|---|---|---|---|---|
| none | - | -2.141e-06 | +1 | +0.7176 | 1081 |
| only_cv1 | 1 | -1.377e-06 | +0.6432 | +0.3558 | 1081 |
| only_cv2 | 2 | -1.873e-06 | +0.875 | +0.6171 | 1081 |
| only_cv3 | 3 | -2.002e-06 | +0.9349 | +0.4787 | 1081 |
| only_cv4 | 4 | -1.828e-06 | +0.8537 | +0.5314 | 1081 |
| only_cv5 | 5 | -1.988e-06 | +0.9287 | +0.6721 | 1081 |
| spanned_block | 1 | -1.377e-06 | +0.6432 | +0.3558 | 1081 |
| weak_block | 2.3.4.5 | -1.119e-06 | +0.5227 | +0.0836 | 1081 |
| all | 1.2.3.4.5 | -6.078e-07 | +0.2839 | +0.01608 | 1081 |

All arms (including 'none') share the macro-available calendar (67 OOS days dropped); gpr/epu lagged one day for publication realism. Replacing v_k by rho_k u_k is an errors-in-variables proxy discarding the (1-rho^2) variance share, so retained shares are LOWER BOUNDS on macro-transmissibility — and with the gate failed, retained shares are ratios of statistical zeros and are reported for completeness only.

## Identification diagnostics

### AE-seed subspace stability (illustration, n_seeds=3)

| seed | n_spanned | v_full_std | phi_spanned | phi_weak | max_angle_factor_space_deg | max_angle_spanned_plane_deg |
|---|---|---|---|---|---|---|
| 0 | 1 | -2.031e-06 | -1.416e-07 | -1.89e-06 | +3.415e-06 | +0 |
| 1 | 1 | -1.835e-06 | -2.472e-07 | -1.587e-06 | +84.73 | +17.49 |
| 2 | 1 | -1.303e-06 | +1.229e-07 | -1.426e-06 | +88.81 | +19.79 |

### Raw-coordinate PBSV by seed (label arbitrariness, do not interpret)

| seed | phi_raw_f1 | phi_raw_f2 | phi_raw_f3 | phi_raw_f4 | phi_raw_f5 |
|---|---|---|---|---|---|
| 0 | -3.123e-07 | -5.15e-07 | -2.403e-07 | -4.208e-07 | -5.427e-07 |
| 1 | -3.863e-07 | -3.287e-07 | -4.618e-07 | -4.172e-07 | -2.406e-07 |
| 2 | -4.08e-07 | -2.361e-07 | -1.337e-08 | -2.703e-07 | -3.755e-07 |

Raw AE coordinates differ across seeds by optimizer multiplicity plus the ReLU gauge (permutation x positive scaling), so per-coordinate phi has no cross-seed correspondence — shown only to demonstrate that raw-coordinate attribution is an attribution to arbitrary labels. The invariance theorem itself is exercised by the unit test (random invertible transform leaves full-model forecasts and CV-basis PBSV unchanged); across-seed grouped-phi dispersion measures AE estimation variance, not basis-invariance failure.

## Pre-registered outcome classes

- **Spanned-block phi ~ 0, weak-block phi ~ 0 (gate failed)**: daily factor state carries no exploitable mean signal — consistent with near-efficient daily commodity futures; the macro-composition question is then moot at this horizon.
- **Weak-block phi > placebo, spanned ~ 0**: EMH-coherent — contemporaneously priced macro content is exactly the least forecastable part of the state; predictability lives in non-macro directions.
- **Spanned-block phi > placebo surviving vol controls AND substitution retains it**: the surprising outcome that would license 'macro-content forecasts commodities'.

## Acceptance checks

- [x] spanned-subspace stable under train bootstrap (median angle < 30 deg) — median=10.4 deg, boundary match 100%
- [x] h=1: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=0.00e+00
- [x] h=5: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=0.00e+00
- [x] h=21: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=3.39e-21
- [x] h=63: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=5.42e-20
- [x] basis invariance: full-model MSE identical in raw vs CV basis (<1e-8 rel) — rel gap=0.00e+00
- [x] no-lookahead: forecasts at origins <= 1822 unchanged when later data scrambled — probe origin index 1822 (h=1)

*Deterministic given --seed=0 (single platform/BLAS). Design disclosure: K=5, ReLU, and the two-block grouping schema were motivated by the full-sample mapping study; spanning labels here are re-derived train-only, but the schema choice itself post-dates that evidence.*
