# Forecast-PBSV Report — Factor-Augmented AR(1) + Forecast-Based Shapley Values

## Verdict

**Well-identified null.** At h=1 the factor-augmented AR(1) does NOT deliver an economically meaningful OOS improvement over AR(1): v(full)=-5.194e-07 with bootstrap CI [-1.77e-06, 6.79e-07]; R2_OOS vs zero -0.00221. Pooled Clark–West p=0.0020, but dropping the single most influential commodity (Diesel) moves it to p=0.3742 — the rejection is signal CONCENTRATION, not breadth (see leave-one-out table). Percent-of-gain language is therefore NOT licensed; phi values below are absolute, with placebo bands showing what pure estimation cost produces at matched cardinality.

Setup: leak-free (train-only) AE, split at **2021-04-30** (train_frac=0.6; COVID crash in train), frozen canonical-variate basis (factor-side ridge pinned to 0, macro-side ridge=0.0), headline targets exclude stale series ['Methanol', 'HRCSteel', 'SGXIronOre', 'Lithium'] (zero-return fraction > 0.15); standardized-loss pooling.

## Forecast accuracy (clean panel, standardized pooling)

| horizon | n_origins | effective_n | r2_pool_std_vs_ar1 | r2_pool_std_vs_mean | r2_pool_std_vs_zero | r2_ar1_vs_zero | cw_pool_stat | cw_pool_p | n_cw_fdr10 | utility_gain_ann | sharpe_diff_ann |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1148 | 1148 | -0.001158 | -0.001981 | -0.002209 | -0.001051 | +2.883 | +0.001969 | 3 | -0.04274 | +0.3994 |
| 5 | 1148 | 229 | -0.004655 | -0.01005 | -0.01104 | -0.006357 | -1.172 | +0.8794 | 1 | -0.05357 | -0.1954 |
| 21 | 1148 | 54 | -0.004465 | -0.01973 | -0.02199 | -0.01745 | -1.336 | +0.9092 | 0 | -0.01554 | -0.08494 |

R2_OOS columns compare the full factor model to each benchmark; `r2_ar1_vs_zero` shows whether the AR(1) baseline itself beats doing nothing. `n_cw_fdr10` counts commodities whose per-commodity Clark–West rejects at BH-FDR 10%. h>1 rows use overlapping targets: `effective_n` is the honest sample size and those rows are suggestive only.

## The attribution basis (train-frozen canonical variates)

| dim | rho_cv_train | perm_p_train | rho_oos_frozen | loading_cosine_oos | spanned | macro_identity |
|---|---|---|---|---|---|---|
| cv1 | +0.7086 | +0.004975 | +0.5593 | +0.8892 | True | xle(+0.83); inflsw_5y(+0.70); be_5y(+0.70); fx_usdcad(-0.64) |
| cv2 | +0.4315 | +0.004975 | +0.5289 | +0.9532 | False | (train-era name not licensed OOS) |
| cv3 | +0.2478 | +0.004975 | +0.1224 | +0.6129 | False | (train-era name not licensed OOS) |
| cv4 | +0.1499 | +0.004975 | +0.1281 | +0.5275 | False | (train-era name not licensed OOS) |
| cv5 | +0.002642 | +0.4378 | -0.04657 | -0.4886 | False | (train-era name not licensed OOS) |

Spanned block = cv1..cv1 by the largest adjacent gap in the train purged-CV rho spectrum. `rho_oos_frozen` is the strictly-forward correlation of frozen factor- and macro-side variates over the OOS segment — the honest number for composite claims. Macro names are shown only where the OOS loading cosine >= 0.8 (loading drift check); the trailing block is 'weakly macro-correlated', not 'orphaned' — all dims can reject the permutation null while having small rho. Shared-variance: rho^2 of the spanned dims is 31% — even 'spanned' directions are majority non-macro variance.

## PBSV (h=1, exact 2^K Shapley, standardized pooling)

| dim | spanned | phi_std | boot_lo | boot_hi | placebo_lo | placebo_hi | placebo_p_right | outside_band |
|---|---|---|---|---|---|---|---|---|
| cv1 | True | +2.386e-07 | -5.092e-07 | +9.409e-07 | -4.18e-07 | +1.068e-07 | +0.0198 | True |
| cv2 | False | -2.677e-07 | -6.281e-07 | +7.72e-08 | -3.963e-07 | +1.344e-07 | +0.8812 | False |
| cv3 | False | -1.197e-07 | -7.071e-07 | +4.32e-07 | -3.03e-07 | +2.481e-08 | +0.4059 | False |
| cv4 | False | -4.359e-08 | -5.19e-07 | +4.303e-07 | -3.638e-07 | +1.4e-07 | +0.2277 | False |
| cv5 | False | -3.27e-07 | -4.999e-07 | -1.595e-07 | -2.889e-07 | +8.22e-08 | +0.9901 | True |

Bootstrap CIs are descriptive and conditional on the frozen basis and realized parameter path; significance belongs to the Clark–West tests. phi inside the placebo band is indistinguishable from pure estimation cost at matched subset cardinality and must not be interpreted. phi for the trailing (near-tied rho) dims is individually non-identified — only the block sum is meaningful.

### Grouped game (rotation-invariant blocks)

- v(spanned)=+1.744e-07, v(weak)=-8.040e-07, v(full)=-5.194e-07, interaction=+1.102e-07
- phi_spanned=+2.295e-07 CI[-5.227e-07,+9.293e-07]  placebo[-4.183e-07,+1.072e-07]
- phi_weak=-7.489e-07 CI[-1.519e-06,-9.105e-09]  placebo[-9.740e-07,-1.170e-07]
- Grouped Shapley is NOT the within-block sum of per-direction phi (no group consistency); the full coalition table above shows the interaction explicitly.

### Boundary sensitivity

| boundary | phi_lead_block | phi_tail_block | v_lead | v_tail | interaction |
|---|---|---|---|---|---|
| 1|2.3.4.5 | +2.295e-07 | -7.489e-07 | +1.744e-07 | -8.04e-07 | +1.102e-07 |
| 1.2|3.4.5 | -2.452e-08 | -4.949e-07 | +5.006e-08 | -4.203e-07 | -1.492e-07 |
| 1.2.3|4.5 | -1.414e-07 | -3.78e-07 | -1.594e-07 | -3.961e-07 | +3.604e-08 |
| 1.2.3.4|5 | -1.944e-07 | -3.25e-07 | -1.974e-07 | -3.281e-07 | +6.042e-09 |

## Robustness

### Signal concentration (leave-one-commodity-out Clark–West)

| dropped | cw_stat | cw_p |
|---|---|---|
| Diesel | +0.3207 | +0.3742 |
| Aluminium | +2.642 | +0.004125 |
| Zinc | +2.77 | +0.002804 |
| Nickel | +2.778 | +0.002732 |
| Sugar | +2.828 | +0.002342 |

Pooled significance that dies when one series is dropped is a data artifact, not a cross-sectional phenomenon — the usual culprit is residual staleness (forward-filled assessment series make next-day returns partly deterministic). Series-level staleness stats are in `forecast_accuracy.csv`.

phi recomputed with **Diesel excluded**: cv1=-2.860e-07, cv2=-3.157e-07, cv3=-2.404e-07, cv4=-7.432e-08, cv5=-2.982e-07. Any placebo-band exceedance in the headline phi table that does not survive this exclusion is attributable to that single series, not to the factor direction.

### Vol/momentum controls (both models)

| dim | phi_no_controls | phi_vol_mom_controls |
|---|---|---|
| cv1 | +2.386e-07 | +2.502e-07 |
| cv2 | -2.677e-07 | -2.904e-07 |
| cv3 | -1.197e-07 | -1.191e-07 |
| cv4 | -4.359e-08 | -4.159e-08 |
| cv5 | -3.27e-07 | -3.264e-07 |

If spanned-block phi survives only without vol controls, the 'macro' signal is a volatility-timing effect wearing a macro label (ReLU latents are partly vol proxies).

### Weighting and sub-period stability

| year | n_days | phi_cv1 | phi_cv2 | phi_cv3 | phi_cv4 | phi_cv5 |
|---|---|---|---|---|---|---|
| 2021 | 155 | -1.41e-07 | +1.621e-08 | +6.527e-07 | +1.553e-07 | -4.44e-07 |
| 2022 | 222 | +7.706e-07 | -1.729e-07 | -5.182e-07 | +2.846e-07 | -2.312e-07 |
| 2023 | 226 | +8.517e-07 | -2.224e-07 | -1.756e-07 | -5.102e-07 | -4.175e-07 |
| 2024 | 227 | +5.475e-07 | -6.46e-08 | -1.186e-07 | -2.757e-07 | -8.484e-08 |
| 2025 | 227 | +8.396e-07 | +1.314e-07 | -5.984e-07 | +1.371e-07 | -2.434e-07 |
| 2026 | 91 | -4.205e-06 | -2.598e-06 | +8.676e-07 | +1.043e-07 | -9.495e-07 |
| -1 | 1085 | +1.849e-07 | -2.825e-07 | -1.215e-07 | -4.378e-08 | -3.093e-07 |

(year -1 = burn-in robustness: first 63 OOS days dropped.)

## Macro substitution (the only arm licensed to say 'explained by macro')

| arm | substituted_dims | v_full_std | retained_share_vs_none | cw_p | n_origins |
|---|---|---|---|---|---|
| none | - | -5.8e-07 | +1 | +0.00183 | 1081 |
| only_cv1 | 1 | -8.013e-07 | +1.382 | +0.02356 | 1081 |
| only_cv2 | 2 | +1.204e-07 | -0.2076 | +0.0001655 | 1081 |
| only_cv3 | 3 | -1.098e-06 | +1.893 | +0.00863 | 1081 |
| only_cv4 | 4 | -7.261e-07 | +1.252 | +0.001101 | 1081 |
| only_cv5 | 5 | +3.287e-07 | -0.5668 | +3.199e-06 | 1081 |
| spanned_block | 1 | -8.013e-07 | +1.382 | +0.02356 | 1081 |
| weak_block | 2.3.4.5 | +2.885e-07 | -0.4975 | +3.653e-06 | 1081 |
| all | 1.2.3.4.5 | +1.686e-07 | -0.2907 | +7.46e-05 | 1081 |

All arms (including 'none') share the macro-available calendar (67 OOS days dropped); gpr/epu lagged one day for publication realism. Replacing v_k by rho_k u_k is an errors-in-variables proxy discarding the (1-rho^2) variance share, so retained shares are LOWER BOUNDS on macro-transmissibility — and with the gate failed, retained shares are ratios of statistical zeros and are reported for completeness only.

## Identification diagnostics

### AE-seed subspace stability (illustration, n_seeds=3)

| seed | n_spanned | v_full_std | phi_spanned | phi_weak | max_angle_factor_space_deg | max_angle_spanned_plane_deg |
|---|---|---|---|---|---|---|
| 0 | 1 | -5.194e-07 | +2.295e-07 | -7.489e-07 | +0 | +2.091e-06 |
| 1 | 1 | +3.166e-07 | +4.338e-07 | -1.172e-07 | +89.32 | +10.01 |
| 2 | 1 | -4.522e-07 | +2.133e-07 | -6.655e-07 | +82.01 | +10.95 |

### Raw-coordinate PBSV by seed (label arbitrariness, do not interpret)

| seed | phi_raw_f1 | phi_raw_f2 | phi_raw_f3 | phi_raw_f4 | phi_raw_f5 |
|---|---|---|---|---|---|
| 0 | -2.237e-07 | -2.548e-07 | -1.492e-07 | -1.913e-07 | +2.997e-07 |
| 1 | -2.958e-07 | -1.768e-07 | -2.534e-07 | -7.929e-08 | +1.122e-06 |
| 2 | -1.08e-07 | +1.178e-07 | -1.763e-08 | -2.262e-07 | -2.181e-07 |

Raw AE coordinates differ across seeds by optimizer multiplicity plus the ReLU gauge (permutation x positive scaling), so per-coordinate phi has no cross-seed correspondence — shown only to demonstrate that raw-coordinate attribution is an attribution to arbitrary labels. The invariance theorem itself is exercised by the unit test (random invertible transform leaves full-model forecasts and CV-basis PBSV unchanged); across-seed grouped-phi dispersion measures AE estimation variance, not basis-invariance failure.

## Pre-registered outcome classes

- **Spanned-block phi ~ 0, weak-block phi ~ 0 (gate failed)**: daily factor state carries no exploitable mean signal — consistent with near-efficient daily commodity futures; the macro-composition question is then moot at this horizon.
- **Weak-block phi > placebo, spanned ~ 0**: EMH-coherent — contemporaneously priced macro content is exactly the least forecastable part of the state; predictability lives in non-macro directions.
- **Spanned-block phi > placebo surviving vol controls AND substitution retains it**: the surprising outcome that would license 'macro-content forecasts commodities'.

## Acceptance checks

- [x] spanned-subspace stable under train bootstrap (median angle < 30 deg) — median=4.6 deg, boundary match 86%
- [x] h=1: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=0.00e+00
- [x] h=5: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=0.00e+00
- [x] h=21: Shapley efficiency |sum(phi)-v(full)| < 1e-12 — gap=0.00e+00
- [x] basis invariance: full-model MSE identical in raw vs CV basis (<1e-8 rel) — rel gap=1.21e-16
- [x] no-lookahead: forecasts at origins <= 1822 unchanged when later data scrambled — probe origin index 1822 (h=1)

*Deterministic given --seed=0 (single platform/BLAS). Design disclosure: K=5, ReLU, and the two-block grouping schema were motivated by the full-sample mapping study; spanning labels here are re-derived train-only, but the schema choice itself post-dates that evidence.*
