# Rolling-Window Forecast + Rolling-Explainability Report

## Verdict

**Explainability survives rolling; forecast value does not.** Even as the AE latent axes rotate window to window, the leading canonical rank keeps the same macro fingerprint far above the label-shuffle null (see §1), and still tracks macro out-of-sample, so rank-level macro attribution remains meaningful under rolling. The pooled forecast gate still fails — the EMH-coherent null carries over from the frozen-basis study.

Setup: walk-forward roll, **252d** train / **21d** test, step 21; model **beta_vae**, seed 0; sample 2013-10-21 → 2026-06-01; **125/125** windows usable. Everything re-fits per window: the AE is retrained (budget 60 epochs), the CCA basis is refit, ranks are sorted by train canonical correlation and sign-anchored on their dominant macro loading so they are comparable across windows. Stability ladder: **4× weak, 1× stable**; **1/5** ranks clear the stability bar (0.7) and **5/5** beat the label-shuffle null (random |cos| ≈ 0.13).

## 1. Does explainability survive rolling?

The AE is retrained every window, so its latent coordinates rotate freely (gauge + optimizer multiplicity + regime drift). The macro side, however, lives in the same fixed macro-series space every window, so we judge rank identity by the cross-window cosine of each rank's macro structure-correlation vector. Two controls make this rigorous: a **label-shuffle null** (independently permuting each window's macro labels destroys cross-window identity; random |cos| ≈ 0.13) with a one-sided `p_value`, and an **out-of-sample check** (`median_rho_oos` = the frozen-in-window canonical correlation measured on the held-out test block). A rank earns 'stable' only if it clears the bar AND beats the shuffle null; 'partial'/'weak' are above-null but below the bar; 'rotating' is indistinguishable from the null.

| rank | median_abs_cos | null_p95 | chance_abs_cos | p_value | median_rho_oos | stability_class |
|---|---|---|---|---|---|---|
| V1 | 0.711 | 0.116 | 0.131 | 0.005 | 0.415 | stable |
| V2 | 0.343 | 0.121 | 0.131 | 0.005 | 0.182 | weak |
| V3 | 0.250 | 0.117 | 0.131 | 0.005 | 0.201 | weak |
| V4 | 0.204 | 0.118 | 0.131 | 0.005 | 0.150 | weak |
| V5 | 0.203 | 0.116 | 0.131 | 0.005 | 0.164 | weak |

See `figures/rolling_forecast_beta/rank_stability.png` and `rho_by_window.png`.

## 2. Average macro fingerprint per rank

Mean (sign-anchored) macro structure correlation across all windows; top drivers per rank:

| rank | top_macro (mean across windows) |
|---|---|
| V1 | ovx(+0.17); gvz(+0.10); vix(+0.09); spx(-0.09) |
| V2 | fx_usdzar(+0.13); fx_usdcnh(+0.12); fx_bbdxy(+0.12); fx_dxy(+0.12) |
| V3 | gpr(+0.08); shcomp(+0.06); csi300(+0.06); move(+0.05) |
| V4 | epu(+0.08); gpr(+0.07); ovx(+0.06); ust_5y(+0.06) |
| V5 | gpr(+0.07); ust_5y(+0.05); be_10y(+0.05); csi300(+0.05) |

## 3. Pooled forecast accuracy (across all rolling windows)

| horizon | n_origins | n_nonoverlap | r2_vs_ar1 | r2_vs_zero | cw_p | cw_nonoverlap_p | v_full | beats_zero |
|---|---|---|---|---|---|---|---|---|
| 1 | 2617 | 2617 | -0.03361 | -0.04227 | +0.83715 | +0.83126 | -0.00001 | False |
| 5 | 2613 | 523 | -0.02565 | -0.05808 | +0.30160 | +0.12125 | -0.00004 | False |
| 21 | 2597 | 124 | -0.02646 | -0.21134 | +0.83685 | +0.83820 | -0.00020 | False |
| 63 | 2555 | 41 | -0.02700 | -1.48817 | +0.94327 | +0.37713 | -0.00116 | False |

`r2_vs_ar1` is the pooled standardized OOS R² of the full factor-augmented AR vs AR(1); `cw_p` is the Clark–West one-sided p (nested), `cw_nonoverlap_p` restricts to non-overlapping targets. Forecast gate (h=1): **FAIL**.

## 4. Rank-pooled Shapley attribution

Players are canonical ranks V1..V5; grouping boundary is the median spanned-block size across windows (V1..V1 | V2..V5). Attribution is only interpretable for macro-stable ranks (see §1).

| horizon | rank | macro_stable | phi | boot_lo | boot_hi |
|---|---|---|---|---|---|
| 1 | V1 | True | -2.307e-06 | -3.283e-06 | -1.353e-06 |
| 1 | V2 | False | -2.851e-06 | -4.328e-06 | -1.539e-06 |
| 1 | V3 | False | -2.057e-06 | -2.814e-06 | -1.368e-06 |
| 1 | V4 | False | -2.194e-06 | -3.121e-06 | -1.423e-06 |
| 1 | V5 | False | -1.787e-06 | -2.511e-06 | -1.139e-06 |
| 5 | V1 | True | -1.063e-05 | -1.882e-05 | -4.035e-06 |
| 5 | V2 | False | -9.535e-06 | -1.519e-05 | -4.613e-06 |
| 5 | V3 | False | -5.306e-06 | -8.334e-06 | -2.707e-06 |
| 5 | V4 | False | -1.140e-05 | -1.652e-05 | -6.956e-06 |
| 5 | V5 | False | -6.137e-06 | -9.402e-06 | -3.446e-06 |
| 21 | V1 | True | -3.819e-05 | -8.307e-05 | -5.797e-06 |
| 21 | V2 | False | -5.304e-05 | -9.156e-05 | -2.614e-05 |
| 21 | V3 | False | -3.434e-05 | -5.213e-05 | -1.816e-05 |
| 21 | V4 | False | -5.000e-05 | -8.425e-05 | -2.376e-05 |
| 21 | V5 | False | -2.056e-05 | -3.244e-05 | -9.954e-06 |
| 63 | V1 | True | -1.313e-04 | -2.016e-04 | -7.107e-05 |
| 63 | V2 | False | -4.331e-04 | -1.203e-03 | +1.679e-05 |
| 63 | V3 | False | -3.064e-04 | -6.235e-04 | -1.023e-04 |
| 63 | V4 | False | -1.665e-04 | -2.851e-04 | -6.922e-05 |
| 63 | V5 | False | -1.234e-04 | -2.694e-04 | -1.476e-05 |

### Grouped (spanned vs weak)

| horizon | phi_spanned | phi_weak | v_full |
|---|---|---|---|
| 1 | -2.325e-06 | -8.872e-06 | -1.120e-05 |
| 5 | -1.067e-05 | -3.234e-05 | -4.301e-05 |
| 21 | -3.824e-05 | -1.579e-04 | -1.961e-04 |
| 63 | -1.307e-04 | -1.030e-03 | -1.161e-03 |

## Acceptance checks

- [x] leading rank is macro-stable AND clears the shuffle null — V1 median|cos|=0.71, p=0.005, rho_oos=0.42
- [x] h=1: Shapley efficiency |sum(phi)-v(full)| < 1e-10 — gap=0.00e+00
- [x] h=5: Shapley efficiency |sum(phi)-v(full)| < 1e-10 — gap=0.00e+00
- [x] h=21: Shapley efficiency |sum(phi)-v(full)| < 1e-10 — gap=2.71e-20
- [x] h=63: Shapley efficiency |sum(phi)-v(full)| < 1e-10 — gap=2.17e-19

