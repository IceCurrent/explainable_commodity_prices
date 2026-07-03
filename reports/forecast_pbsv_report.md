# Forecast-PBSV Report — Factor-Augmented AR(1) + Forecast-Based Shapley Values

## Verdict

**Well-identified null.** At h=1 the factor-augmented AR(1) does NOT deliver an economically meaningful OOS improvement over AR(1): v(full)=-1.244e-06 with bootstrap CI [-2.62e-06, 3.91e-08]; R2_OOS vs zero -0.00253. Pooled Clark–West p=0.3280, but dropping the single most influential commodity (Aluminium) moves it to p=0.4362 — the rejection is signal CONCENTRATION, not breadth (see leave-one-out table). Percent-of-gain language is therefore NOT licensed; phi values below are absolute, with placebo bands showing what pure estimation cost produces at matched cardinality.

Setup: leak-free (train-only) AE, split at **2021-04-30** (train_frac=0.6; COVID crash in train), frozen canonical-variate basis (factor-side ridge pinned to 0, macro-side ridge=0.0), headline targets exclude stale series [] (zero-return fraction > 0.15); standardized-loss pooling.

**Horizon ladder (the point of this study).** The same test was run at daily, weekly, monthly and quarterly horizons (h=1, h=5, h=21, h=63). The share-of-gain gate passes at: **NO horizon**. Macro moves commodities contemporaneously (the CCA spanning map), but that content is not converted into out-of-sample forecast power even at monthly/quarterly horizons — the EMH-coherent reading. See the horizon table below.

## Forecast accuracy (clean panel, standardized pooling)

| horizon | n_origins | effective_n | r2_pool_std_vs_ar1 | r2_pool_std_vs_zero | r2_ar1_vs_zero | cw_pool_p | cw_nonoverlap_p | n_cw_fdr10 | utility_gain_ann | sharpe_diff_ann |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1148 | 1148 | -0.003101 | -0.002533 | +0.0005667 | +0.328 | +0.3347 | 0 | -0.08507 | -0.1838 |
| 5 | 1148 | 229 | -0.004812 | -0.009201 | -0.004368 | +0.9965 | +0.7642 | 0 | -0.04719 | -0.2268 |
| 21 | 1148 | 54 | -0.003379 | -0.01838 | -0.01495 | +0.6903 | +0.377 | 0 | -0.009372 | +0.04177 |
| 63 | 1148 | 18 | -0.002095 | -0.04946 | -0.04727 | +0.0879 | +0.04835 | 0 | -0.0003149 | +0.0689 |

R2_OOS columns compare the full factor model to each benchmark; `r2_ar1_vs_zero` shows whether the AR(1) baseline itself beats doing nothing. `cw_pool_p` uses all (overlapping) origins with a HAC bandwidth >= 2h; `cw_nonoverlap_p` re-runs Clark–West on targets spaced >= h apart (autocorrelation-free but lower power), the honest long-horizon check. `n_cw_fdr10` counts commodities whose per-commodity Clark–West rejects at BH-FDR 10%. `effective_n = n_origins / h` is the honest sample size.

## Macro transmission across horizons

The central question — *do macros move commodities?* — is answered as a function of forecast horizon. For each horizon the full factor state (whose macro-spanned block is identified train-only) is scored against AR(1) out-of-sample, and the macro-substitution arm measures how much of any gain is macro-transmissible (errors-in-variables lower bound). `gate_passed` = the pre-registered share-of-gain gate (pooled CW<0.05, LOCO-robust, placebo-calibrated CW<0.05, v(full) bootstrap CI>0, beats zero).

| horizon | effective_n | r2_oos_vs_ar1 | cw_p_overlap | cw_p_nonoverlap | cw_placebo_p | loco_max_p | v_full_std | phi_spanned | spanned_outside_band | retained_share_spanned | gate_passed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1148 | -0.003101 | +0.328 | +0.3347 | +0.2673 | +0.4362 | -1.244e-06 | -2.219e-07 | False | +0.893 | False |
| 5 | 229 | -0.004812 | +0.9965 | +0.7642 | +1 | +0.9994 | -9.287e-06 | -1.846e-06 | False | +1.076 | False |
| 21 | 54 | -0.003379 | +0.6903 | +0.377 | +0.6337 | +0.7837 | -2.417e-05 | -8.554e-06 | False | +1.537 | False |
| 63 | 18 | -0.002095 | +0.0879 | +0.04835 | +0.198 | +0.1577 | -4.523e-05 | -1.153e-05 | False | +0.8844 | False |

`retained_share_spanned` is interpretable only where `gate_passed` is true (otherwise it is a ratio of statistical zeros). `spanned_outside_band` flags whether the spanned-block PBSV clears its cardinality-matched zero-signal placebo band. See `figures/forecast_pbsv/transmission_by_horizon.png` and `results/forecast_pbsv/macro_transmission_by_horizon.csv`.

## The attribution basis (train-frozen canonical variates)

| dim | rho_cv_train | perm_p_train | rho_oos_frozen | loading_cosine_oos | spanned | macro_identity |
|---|---|---|---|---|---|---|
| cv1 | +0.6907 | +0.004975 | +0.5867 | +0.9348 | True | xle(+0.82); inflsw_5y(+0.68); be_5y(+0.67); fx_usdcad(-0.67) |
| cv2 | +0.4377 | +0.004975 | +0.517 | +0.953 | False | (train-era name not licensed OOS) |
| cv3 | +0.2719 | +0.004975 | +0.1323 | +0.7964 | False | (train-era name not licensed OOS) |
| cv4 | +0.1392 | +0.004975 | +0.09 | +0.1083 | False | (train-era name not licensed OOS) |
| cv5 | -0.001384 | +0.5174 | +0.008514 | +0.4717 | False | (train-era name not licensed OOS) |

Spanned block = cv1..cv1 by the largest adjacent gap in the train purged-CV rho spectrum. `rho_oos_frozen` is the strictly-forward correlation of frozen factor- and macro-side variates over the OOS segment — the honest number for composite claims. Macro names are shown only where the OOS loading cosine >= 0.8 (loading drift check); the trailing block is 'weakly macro-correlated', not 'orphaned' — all dims can reject the permutation null while having small rho. Shared-variance: rho^2 of the spanned dims is 34% — even 'spanned' directions are majority non-macro variance.

## PBSV (h=1, exact 2^K Shapley, standardized pooling)

| dim | spanned | phi_std | boot_lo | boot_hi | placebo_lo | placebo_hi | placebo_p_right | outside_band |
|---|---|---|---|---|---|---|---|---|
| cv1 | True | -2.038e-07 | -7.203e-07 | +2.543e-07 | -3.947e-07 | +8.981e-08 | +0.7525 | False |
| cv2 | False | -4.789e-07 | -8.125e-07 | -2.172e-07 | -3.828e-07 | +1.065e-07 | +1 | True |
| cv3 | False | -1.695e-07 | -8.614e-07 | +5.356e-07 | -3.219e-07 | +8.183e-08 | +0.6238 | False |
| cv4 | False | +2.056e-07 | -4.836e-07 | +1.013e-06 | -3.429e-07 | +1.006e-07 | +0.009901 | True |
| cv5 | False | -5.977e-07 | -1.43e-06 | -4.856e-08 | -3.778e-07 | +1.426e-07 | +1 | True |

Bootstrap CIs are descriptive and conditional on the frozen basis and realized parameter path; significance belongs to the Clark–West tests. phi inside the placebo band is indistinguishable from pure estimation cost at matched subset cardinality and must not be interpreted. phi for the trailing (near-tied rho) dims is individually non-identified — only the block sum is meaningful.

### Grouped game (rotation-invariant blocks)

- v(spanned)=-1.204e-07, v(weak)=-9.210e-07, v(full)=-1.244e-06, interaction=-2.029e-07
- phi_spanned=-2.219e-07 CI[-7.439e-07,+2.419e-07]  placebo[-3.950e-07,+8.961e-08]
- phi_weak=-1.022e-06 CI[-2.357e-06,+3.152e-07]  placebo[-9.794e-07,-5.216e-08]
- Grouped Shapley is NOT the within-block sum of per-direction phi (no group consistency); the full coalition table above shows the interaction explicitly.

### Boundary sensitivity

| boundary | phi_lead_block | phi_tail_block | v_lead | v_tail | interaction |
|---|---|---|---|---|---|
| 1|2.3.4.5 | -2.219e-07 | -1.022e-06 | -1.204e-07 | -9.21e-07 | -2.029e-07 |
| 1.2|3.4.5 | -6.772e-07 | -5.671e-07 | -5.251e-07 | -4.149e-07 | -3.043e-07 |
| 1.2.3|4.5 | -8.364e-07 | -4.079e-07 | -8.052e-07 | -3.768e-07 | -6.237e-08 |
| 1.2.3.4|5 | -6.437e-07 | -6.006e-07 | -6.723e-07 | -6.292e-07 | +5.725e-08 |

## Robustness

### Signal concentration (leave-one-commodity-out Clark–West)

| dropped | cw_stat | cw_p |
|---|---|---|
| Aluminium | +0.1605 | +0.4362 |
| Zinc | +0.211 | +0.4164 |
| Brent | +0.2864 | +0.3873 |
| Sugar | +0.3286 | +0.3712 |
| WTI | +0.3368 | +0.3681 |

Pooled significance that dies when one series is dropped is a data artifact, not a cross-sectional phenomenon — the usual culprit is residual staleness (forward-filled assessment series make next-day returns partly deterministic). Series-level staleness stats are in `forecast_accuracy.csv`.

phi recomputed with **Aluminium excluded**: cv1=-2.067e-07, cv2=-4.446e-07, cv3=-2.353e-07, cv4=+1.861e-07, cv5=-5.868e-07. Any placebo-band exceedance in the headline phi table that does not survive this exclusion is attributable to that single series, not to the factor direction.

### Vol/momentum controls (both models)

| dim | phi_no_controls | phi_vol_mom_controls |
|---|---|---|
| cv1 | -2.038e-07 | -2.052e-07 |
| cv2 | -4.789e-07 | -5.042e-07 |
| cv3 | -1.695e-07 | -1.627e-07 |
| cv4 | +2.056e-07 | +2.112e-07 |
| cv5 | -5.977e-07 | -6.177e-07 |

If spanned-block phi survives only without vol controls, the 'macro' signal is a volatility-timing effect wearing a macro label (ReLU latents are partly vol proxies).

### Weighting and sub-period stability

| year | n_days | phi_cv1 | phi_cv2 | phi_cv3 | phi_cv4 | phi_cv5 |
|---|---|---|---|---|---|---|
| 2021 | 155 | -2.634e-07 | -3.668e-07 | +4.972e-07 | -4.05e-08 | -5.086e-07 |
| 2022 | 222 | -1.614e-06 | -1.226e-06 | -5.803e-07 | +9.174e-07 | -7.369e-07 |
| 2023 | 226 | +2.054e-08 | -5.34e-08 | -1.277e-07 | -1.886e-07 | -3.483e-08 |
| 2024 | 227 | +1.27e-07 | -8.947e-08 | -2.909e-08 | -7.409e-07 | +2.641e-07 |
| 2025 | 227 | +4.96e-07 | -1.725e-07 | -3.939e-07 | +4.339e-08 | -4.332e-07 |
| 2026 | 91 | +2.108e-07 | -1.641e-06 | -1.968e-07 | +2.634e-06 | -4.368e-06 |
| -1 | 1085 | -2.61e-07 | -4.895e-07 | -1.915e-07 | +2.371e-07 | -5.882e-07 |

(year -1 = burn-in robustness: first 63 OOS days dropped.)

## Macro substitution (the only arm licensed to say 'explained by macro')

| arm | substituted_dims | v_full_std | retained_share_vs_none | cw_p | n_origins |
|---|---|---|---|---|---|
| none | - | -1.416e-06 | +1 | +0.352 | 1081 |
| only_cv1 | 1 | -1.264e-06 | +0.893 | +0.1599 | 1081 |
| only_cv2 | 2 | -2.633e-07 | +0.186 | +0.007189 | 1081 |
| only_cv3 | 3 | -1.76e-06 | +1.243 | +0.4336 | 1081 |
| only_cv4 | 4 | -1.915e-06 | +1.353 | +0.755 | 1081 |
| only_cv5 | 5 | -8.085e-07 | +0.5711 | +0.07352 | 1081 |
| spanned_block | 1 | -1.264e-06 | +0.893 | +0.1599 | 1081 |
| weak_block | 2.3.4.5 | -5.93e-07 | +0.4189 | +0.007865 | 1081 |
| all | 1.2.3.4.5 | -6.902e-07 | +0.4875 | +0.007697 | 1081 |

All arms (including 'none') share the macro-available calendar (67 OOS days dropped); gpr/epu lagged one day for publication realism. Replacing v_k by rho_k u_k is an errors-in-variables proxy discarding the (1-rho^2) variance share, so retained shares are LOWER BOUNDS on macro-transmissibility — and with the gate failed, retained shares are ratios of statistical zeros and are reported for completeness only.

## Identification diagnostics

### AE-seed subspace stability (illustration, n_seeds=3)

| seed | n_spanned | v_full_std | phi_spanned | phi_weak | max_angle_factor_space_deg | max_angle_spanned_plane_deg |
|---|---|---|---|---|---|---|
| 0 | 1 | -1.244e-06 | -2.219e-07 | -1.022e-06 | +4.353e-06 | +0 |
| 1 | 1 | -7.229e-07 | -2.531e-07 | -4.698e-07 | +82.98 | +13.07 |
| 2 | 1 | -5.136e-07 | -2.321e-07 | -2.815e-07 | +59.19 | +9.909 |

### Raw-coordinate PBSV by seed (label arbitrariness, do not interpret)

| seed | phi_raw_f1 | phi_raw_f2 | phi_raw_f3 | phi_raw_f4 | phi_raw_f5 |
|---|---|---|---|---|---|
| 0 | -7.314e-07 | -2.23e-07 | -2.813e-07 | +3.346e-07 | -3.433e-07 |
| 1 | -1.697e-07 | -3.447e-07 | -1.896e-07 | +3.559e-07 | -3.748e-07 |
| 2 | -6.878e-08 | -3.419e-07 | +1.626e-07 | -1.869e-07 | -7.857e-08 |

Raw AE coordinates differ across seeds by optimizer multiplicity plus the ReLU gauge (permutation x positive scaling), so per-coordinate phi has no cross-seed correspondence — shown only to demonstrate that raw-coordinate attribution is an attribution to arbitrary labels. The invariance theorem itself is exercised by the unit test (random invertible transform leaves full-model forecasts and CV-basis PBSV unchanged); across-seed grouped-phi dispersion measures AE estimation variance, not basis-invariance failure.

## Pre-registered outcome classes

- **Spanned-block phi ~ 0, weak-block phi ~ 0 (gate failed)**: daily factor state carries no exploitable mean signal — consistent with near-efficient daily commodity futures; the macro-composition question is then moot at this horizon.
- **Weak-block phi > placebo, spanned ~ 0**: EMH-coherent — contemporaneously priced macro content is exactly the least forecastable part of the state; predictability lives in non-macro directions.
- **Spanned-block phi > placebo surviving vol controls AND substitution retains it**: the surprising outcome that would license 'macro-content forecasts commodities'.

## Acceptance checks

- [x] spanned-subspace stable under train bootstrap (median angle < 30 deg) — median=5.6 deg, boundary match 82%
- [x] h=1: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=2.12e-22
- [x] h=5: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=0.00e+00
- [x] h=21: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=0.00e+00
- [x] h=63: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=6.78e-21
- [x] basis invariance: full-model MSE identical in raw vs CV basis (<1e-8 rel) — rel gap=0.00e+00
- [x] no-lookahead: forecasts at origins <= 1822 unchanged when later data scrambled — probe origin index 1822 (h=1)

*Deterministic given --seed=0 (single platform/BLAS). Design disclosure: K=5, ReLU, and the two-block grouping schema were motivated by the full-sample mapping study; spanning labels here are re-derived train-only, but the schema choice itself post-dates that evidence.*
