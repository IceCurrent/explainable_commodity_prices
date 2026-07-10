# Rolling-Window Forecast + Rolling-Explainability Report

## Verdict

**Explainability survives rolling; forecast value does not.** Even as the AE latent axes rotate window to window, the leading canonical rank keeps the same macro fingerprint far above the label-shuffle null (see §1), and still tracks macro out-of-sample, so rank-level macro attribution remains meaningful under rolling. The pooled forecast gate still fails — the EMH-coherent null carries over from the frozen-basis study.

Setup: walk-forward roll, **252d** train / **21d** test, step 21; model **vanilla**, seed 0; sample 2013-10-21 → 2026-06-01; **125/125** windows usable. Everything re-fits per window: the AE is retrained (budget 60 epochs), the CCA basis is refit, ranks are sorted by train canonical correlation and sign-anchored on their dominant macro loading so they are comparable across windows. Stability ladder: **3× weak, 1× stable, 1× partial**; **1/5** ranks clear the stability bar (0.7) and **5/5** beat the label-shuffle null (random |cos| ≈ 0.13).

## 1. Does explainability survive rolling?

The AE is retrained every window, so its latent coordinates rotate freely (gauge + optimizer multiplicity + regime drift). The macro side, however, lives in the same fixed macro-series space every window, so we judge rank identity by the cross-window cosine of each rank's macro structure-correlation vector. Two controls make this rigorous: a **label-shuffle null** (independently permuting each window's macro labels destroys cross-window identity; random |cos| ≈ 0.13) with a one-sided `p_value`, and an **out-of-sample check** (`median_rho_oos` = the frozen-in-window canonical correlation measured on the held-out test block). A rank earns 'stable' only if it clears the bar AND beats the shuffle null; 'partial'/'weak' are above-null but below the bar; 'rotating' is indistinguishable from the null.

| rank | median_abs_cos | null_p95 | chance_abs_cos | p_value | median_rho_oos | stability_class |
|---|---|---|---|---|---|---|
| V1 | 0.820 | 0.115 | 0.131 | 0.005 | 0.534 | stable |
| V2 | 0.605 | 0.117 | 0.131 | 0.005 | 0.317 | partial |
| V3 | 0.280 | 0.122 | 0.131 | 0.005 | 0.200 | weak |
| V4 | 0.214 | 0.119 | 0.131 | 0.005 | 0.185 | weak |
| V5 | 0.191 | 0.117 | 0.131 | 0.005 | 0.171 | weak |

See `figures/rolling_forecast/rank_stability.png` and `rho_by_window.png`.

## 2. Average macro fingerprint per rank

Mean (sign-anchored) macro structure correlation across all windows; top drivers per rank:

| rank | top_macro (mean across windows) |
|---|---|
| V1 | xle(+0.32); inflsw_5y(+0.23); be_5y(+0.23); fx_usdcad(-0.22) |
| V2 | fx_usdclp(+0.20); fx_usdzar(+0.15); xle(+0.14); be_5y(+0.14) |
| V3 | ust_10y(+0.08); ust_5y(+0.08); tips_10y(+0.07); hscei(+0.07) |
| V4 | fx_usdbrl(+0.10); fx_usdzar(+0.07); fx_bbdxy(+0.06); inflsw_10y(+0.06) |
| V5 | fx_usdbrl(+0.07); fx_usdcad(+0.06); fx_usdnok(+0.04); fx_usdcnh(+0.04) |

## 3. Pooled forecast accuracy (across all rolling windows)

| horizon | n_origins | n_nonoverlap | r2_vs_ar1 | r2_vs_zero | cw_p | cw_nonoverlap_p | v_full | beats_zero |
|---|---|---|---|---|---|---|---|---|
| 1 | 2617 | 2617 | -0.02653 | -0.03513 | +0.40376 | +0.40872 | -0.00001 | False |
| 5 | 2613 | 523 | -0.02606 | -0.05850 | +0.52910 | +0.47907 | -0.00004 | False |
| 21 | 2597 | 124 | -0.02452 | -0.20905 | +0.82427 | +0.76249 | -0.00018 | False |
| 63 | 2555 | 41 | -0.02351 | -1.47970 | +0.94289 | +0.25553 | -0.00101 | False |

`r2_vs_ar1` is the pooled standardized OOS R² of the full factor-augmented AR vs AR(1); `cw_p` is the Clark–West one-sided p (nested), `cw_nonoverlap_p` restricts to non-overlapping targets. Forecast gate (h=1): **FAIL**.

## 4. Rank-pooled Shapley attribution

Players are canonical ranks V1..V5; grouping boundary is the median spanned-block size across windows (V1..V1 | V2..V5). Attribution is only interpretable for macro-stable ranks (see §1).

| horizon | rank | macro_stable | phi | boot_lo | boot_hi |
|---|---|---|---|---|---|
| 1 | V1 | True | -2.188e-06 | -3.696e-06 | -9.903e-07 |
| 1 | V2 | False | -1.356e-06 | -1.959e-06 | -7.389e-07 |
| 1 | V3 | False | -1.882e-06 | -3.012e-06 | -9.502e-07 |
| 1 | V4 | False | -1.823e-06 | -2.593e-06 | -1.158e-06 |
| 1 | V5 | False | -1.588e-06 | -2.326e-06 | -8.336e-07 |
| 5 | V1 | True | -1.205e-05 | -2.318e-05 | -4.002e-06 |
| 5 | V2 | False | -5.987e-06 | -9.583e-06 | -2.651e-06 |
| 5 | V3 | False | -8.162e-06 | -1.221e-05 | -4.645e-06 |
| 5 | V4 | False | -9.446e-06 | -1.264e-05 | -6.486e-06 |
| 5 | V5 | False | -8.037e-06 | -1.243e-05 | -4.190e-06 |
| 21 | V1 | True | -4.857e-05 | -1.111e-04 | -1.056e-05 |
| 21 | V2 | False | -2.616e-05 | -4.269e-05 | -1.210e-05 |
| 21 | V3 | False | -3.468e-05 | -4.689e-05 | -2.134e-05 |
| 21 | V4 | False | -4.182e-05 | -6.230e-05 | -2.368e-05 |
| 21 | V5 | False | -3.049e-05 | -4.798e-05 | -1.380e-05 |
| 63 | V1 | True | -3.028e-04 | -6.813e-04 | -6.463e-05 |
| 63 | V2 | False | -1.550e-04 | -2.486e-04 | -7.118e-05 |
| 63 | V3 | False | -1.453e-04 | -2.475e-04 | -6.589e-05 |
| 63 | V4 | False | -2.426e-04 | -6.036e-04 | -1.557e-05 |
| 63 | V5 | False | -1.648e-04 | -3.243e-04 | -4.002e-05 |

### Grouped (spanned vs weak)

| horizon | phi_spanned | phi_weak | v_full |
|---|---|---|---|
| 1 | -2.203e-06 | -6.635e-06 | -8.838e-06 |
| 5 | -1.207e-05 | -3.162e-05 | -4.369e-05 |
| 21 | -4.857e-05 | -1.332e-04 | -1.817e-04 |
| 63 | -2.962e-04 | -7.143e-04 | -1.010e-03 |

## Acceptance checks

- [x] leading rank is macro-stable AND clears the shuffle null — V1 median|cos|=0.82, p=0.005, rho_oos=0.53
- [x] h=1: Shapley efficiency |sum(phi)-v(full)| < 1e-10 — gap=1.69e-21
- [x] h=5: Shapley efficiency |sum(phi)-v(full)| < 1e-10 — gap=6.78e-21
- [x] h=21: Shapley efficiency |sum(phi)-v(full)| < 1e-10 — gap=0.00e+00
- [x] h=63: Shapley efficiency |sum(phi)-v(full)| < 1e-10 — gap=0.00e+00

