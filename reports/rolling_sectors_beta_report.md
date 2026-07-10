# Rolling-Window Sector Decomposition Report

## Verdict

**A stable macro axis under rolling is sector-heterogeneous.** Re-fitting the AE and the CCA basis on every window, the leading canonical rank clears the stability bar in: **overall, energy, metals**. The remaining blocks keep only partial/weak macro identity across windows (their leading axis rotates with the AE re-fit).

**No block forecasts.** At no horizon does the factor-augmented AR beat AR(1) for any sector (all Clark–West p ≥ 0.05) — the EMH-coherent null holds sector-by-sector, not just in aggregate.

Setup: walk-forward roll, **252d** train / **21d** test, step 21; model **beta_vae**, seed 0. Each sector runs the identical engine on its own returns (overall = full 21-commodity panel, K=5; each sector K≤3). Nothing is fit on the full sample.

## 1. Block summary

| block | n_commodities | n_factors | n_windows | n_stable_ranks | n_ranks_above_null | lead_rank_class | lead_median_abs_cos | lead_median_rho_oos | best_forecast_cw_p |
|---|---|---|---|---|---|---|---|---|---|
| overall | 21 | 5 | 125 | 1 | 5 | stable | 0.711 | 0.415 | 0.302 |
| energy | 5 | 3 | 125 | 1 | 3 | stable | 0.833 | 0.543 | 0.651 |
| agriculture | 9 | 3 | 125 | 0 | 3 | weak | 0.252 | 0.186 | 0.108 |
| metals | 7 | 3 | 125 | 1 | 3 | stable | 0.753 | 0.367 | 0.124 |

`lead_median_abs_cos` = the leading rank's cross-window median |cosine| of its macro structure-correlation vector (fixed macro space, so comparable across windows even as the AE latent axes rotate); `lead_median_rho_oos` = its frozen-in-window canonical correlation on the held-out test blocks; `best_forecast_cw_p` = the smallest Clark–West p across horizons for that block. See `figures/rolling_sectors_beta/sector_leading_axis.png`.

## 2. Per-rank cross-window stability & OOS macro correlation

| block | rank | median_abs_cos | p_value | median_rho_oos | frac_windows_rho_oos_ge_0.3 | stability_class |
|---|---|---|---|---|---|---|
| overall | V1 | 0.711 | 0.005 | 0.415 | 0.632 | stable |
| overall | V2 | 0.343 | 0.005 | 0.182 | 0.272 | weak |
| overall | V3 | 0.250 | 0.005 | 0.201 | 0.232 | weak |
| overall | V4 | 0.204 | 0.005 | 0.150 | 0.208 | weak |
| overall | V5 | 0.203 | 0.005 | 0.164 | 0.224 | weak |
| energy | V1 | 0.833 | 0.005 | 0.543 | 0.792 | stable |
| energy | V2 | 0.197 | 0.005 | 0.178 | 0.248 | weak |
| energy | V3 | 0.202 | 0.005 | 0.176 | 0.208 | weak |
| agriculture | V1 | 0.252 | 0.005 | 0.186 | 0.216 | weak |
| agriculture | V2 | 0.226 | 0.005 | 0.196 | 0.288 | weak |
| agriculture | V3 | 0.234 | 0.005 | 0.189 | 0.192 | weak |
| metals | V1 | 0.753 | 0.005 | 0.367 | 0.648 | stable |
| metals | V2 | 0.446 | 0.005 | 0.269 | 0.424 | partial |
| metals | V3 | 0.218 | 0.005 | 0.156 | 0.208 | weak |

See `figures/rolling_sectors_beta/sector_oos_rho_by_rank.png`.

## 3. Leading-rank macro fingerprint per block

| block | top_macro |
|---|---|
| overall | ovx(+0.17); gvz(+0.10); vix(+0.09); spx(-0.09) |
| energy | xle(+0.49); be_5y(+0.37); inflsw_5y(+0.37); be_10y(+0.29) |
| agriculture | gpr(+0.11); epu(+0.10); hy_oas(+0.09); move(+0.08) |
| metals | fx_usdcnh(+0.13); fx_usdclp(+0.12); fx_bbdxy(+0.11); fx_usdzar(+0.11) |

## 4. Pooled forecast accuracy by sector and horizon

| block | horizon | n_origins | r2_vs_ar1 | r2_vs_zero | cw_p | cw_nonoverlap_p | beats_zero |
|---|---|---|---|---|---|---|---|
| overall | 1 | 2617 | -0.03361 | -0.04227 | +0.83715 | +0.83126 | False |
| overall | 5 | 2613 | -0.02565 | -0.05808 | +0.30160 | +0.12125 | False |
| overall | 21 | 2597 | -0.02646 | -0.21134 | +0.83685 | +0.83820 | False |
| overall | 63 | 2555 | -0.02700 | -1.48817 | +0.94327 | +0.37713 | False |
| energy | 1 | 2617 | -0.03059 | -0.04277 | +0.65058 | +0.64510 | False |
| energy | 5 | 2613 | -0.04446 | -0.08015 | +0.91558 | +0.91970 | False |
| energy | 21 | 2597 | -0.07846 | -0.33572 | +0.94601 | +0.85382 | False |
| energy | 63 | 2555 | -0.04899 | -2.29129 | +0.91040 | +0.30941 | False |
| agriculture | 1 | 2619 | -0.01836 | -0.02696 | +0.58813 | +0.59143 | False |
| agriculture | 5 | 2615 | -0.01470 | -0.05042 | +0.52592 | +0.40171 | False |
| agriculture | 21 | 2599 | -0.01471 | -0.19560 | +0.54696 | +0.10629 | False |
| agriculture | 63 | 2557 | -0.00403 | -1.35023 | +0.10797 | +0.01560 | False |
| metals | 1 | 2619 | -0.01219 | -0.01875 | +0.12390 | +0.12553 | False |
| metals | 5 | 2615 | -0.01101 | -0.03505 | +0.14555 | +0.12982 | False |
| metals | 21 | 2599 | -0.01191 | -0.15967 | +0.45395 | +0.20216 | False |
| metals | 63 | 2557 | -0.00613 | -1.06779 | +0.60021 | +0.17584 | False |

See `figures/rolling_sectors_beta/sector_forecast_r2.png`.

