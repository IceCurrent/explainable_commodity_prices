# Rolling-Window Sector Decomposition Report

## Verdict

**A stable macro axis under rolling is sector-heterogeneous.** Re-fitting the AE and the CCA basis on every window, the leading canonical rank clears the stability bar in: **overall, energy, metals**. The remaining blocks keep only partial/weak macro identity across windows (their leading axis rotates with the AE re-fit).

**No block forecasts.** At no horizon does the factor-augmented AR beat AR(1) for any sector (all Clark–West p ≥ 0.05) — the EMH-coherent null holds sector-by-sector, not just in aggregate.

Setup: walk-forward roll, **252d** train / **21d** test, step 21; model **vanilla**, seed 0. Each sector runs the identical engine on its own returns (overall = full 21-commodity panel, K=5; each sector K≤3). Nothing is fit on the full sample.

## 1. Block summary

| block | n_commodities | n_factors | n_windows | n_stable_ranks | n_ranks_above_null | lead_rank_class | lead_median_abs_cos | lead_median_rho_oos | best_forecast_cw_p |
|---|---|---|---|---|---|---|---|---|---|
| overall | 21 | 5 | 125 | 1 | 5 | stable | 0.820 | 0.534 | 0.404 |
| energy | 5 | 3 | 125 | 1 | 3 | stable | 0.778 | 0.441 | 0.130 |
| agriculture | 9 | 3 | 125 | 0 | 3 | weak | 0.381 | 0.178 | 0.381 |
| metals | 7 | 3 | 125 | 1 | 3 | stable | 0.845 | 0.441 | 0.109 |

`lead_median_abs_cos` = the leading rank's cross-window median |cosine| of its macro structure-correlation vector (fixed macro space, so comparable across windows even as the AE latent axes rotate); `lead_median_rho_oos` = its frozen-in-window canonical correlation on the held-out test blocks; `best_forecast_cw_p` = the smallest Clark–West p across horizons for that block. See `figures/rolling_sectors/sector_leading_axis.png`.

## 2. Per-rank cross-window stability & OOS macro correlation

| block | rank | median_abs_cos | p_value | median_rho_oos | frac_windows_rho_oos_ge_0.3 | stability_class |
|---|---|---|---|---|---|---|
| overall | V1 | 0.820 | 0.005 | 0.534 | 0.832 | stable |
| overall | V2 | 0.605 | 0.005 | 0.317 | 0.528 | partial |
| overall | V3 | 0.280 | 0.005 | 0.200 | 0.280 | weak |
| overall | V4 | 0.214 | 0.005 | 0.185 | 0.296 | weak |
| overall | V5 | 0.191 | 0.005 | 0.171 | 0.224 | weak |
| energy | V1 | 0.778 | 0.005 | 0.441 | 0.696 | stable |
| energy | V2 | 0.278 | 0.005 | 0.194 | 0.304 | weak |
| energy | V3 | 0.206 | 0.005 | 0.167 | 0.192 | weak |
| agriculture | V1 | 0.381 | 0.005 | 0.178 | 0.264 | weak |
| agriculture | V2 | 0.232 | 0.005 | 0.161 | 0.152 | weak |
| agriculture | V3 | 0.221 | 0.005 | 0.168 | 0.192 | weak |
| metals | V1 | 0.845 | 0.005 | 0.441 | 0.744 | stable |
| metals | V2 | 0.273 | 0.005 | 0.186 | 0.168 | weak |
| metals | V3 | 0.210 | 0.005 | 0.123 | 0.192 | weak |

See `figures/rolling_sectors/sector_oos_rho_by_rank.png`.

## 3. Leading-rank macro fingerprint per block

| block | top_macro |
|---|---|
| overall | xle(+0.32); inflsw_5y(+0.23); be_5y(+0.23); fx_usdcad(-0.22) |
| energy | xle(+0.17); inflsw_5y(+0.14); be_5y(+0.13); inflsw_10y(+0.13) |
| agriculture | gpr(+0.09); mxef(+0.08); hscei(+0.07); fx_audusd(+0.06) |
| metals | ovx(+0.10); gvz(+0.10); fx_usdclp(+0.09); cny_10y(+0.07) |

## 4. Pooled forecast accuracy by sector and horizon

| block | horizon | n_origins | r2_vs_ar1 | r2_vs_zero | cw_p | cw_nonoverlap_p | beats_zero |
|---|---|---|---|---|---|---|---|
| overall | 1 | 2617 | -0.02653 | -0.03513 | +0.40376 | +0.40872 | False |
| overall | 5 | 2613 | -0.02606 | -0.05850 | +0.52910 | +0.47907 | False |
| overall | 21 | 2597 | -0.02452 | -0.20905 | +0.82427 | +0.76249 | False |
| overall | 63 | 2555 | -0.02351 | -1.47970 | +0.94289 | +0.25553 | False |
| energy | 1 | 2617 | -0.01771 | -0.02973 | +0.13009 | +0.14032 | False |
| energy | 5 | 2613 | -0.02034 | -0.05520 | +0.71546 | +0.87483 | False |
| energy | 21 | 2597 | -0.03153 | -0.27759 | +0.88226 | +0.75616 | False |
| energy | 63 | 2555 | -0.01082 | -2.17151 | +0.80889 | +0.96294 | False |
| agriculture | 1 | 2619 | -0.01556 | -0.02413 | +0.48443 | +0.48404 | False |
| agriculture | 5 | 2615 | -0.01595 | -0.05172 | +0.54314 | +0.09493 | False |
| agriculture | 21 | 2599 | -0.01454 | -0.19540 | +0.50127 | +0.52815 | False |
| agriculture | 63 | 2557 | -0.00577 | -1.35430 | +0.38110 | +0.05069 | False |
| metals | 1 | 2619 | -0.01447 | -0.02105 | +0.28268 | +0.28364 | False |
| metals | 5 | 2615 | -0.01498 | -0.03911 | +0.15518 | +0.00608 | False |
| metals | 21 | 2599 | -0.00835 | -0.15559 | +0.10867 | +0.38156 | False |
| metals | 63 | 2557 | -0.00325 | -1.06186 | +0.32835 | +0.69078 | False |

See `figures/rolling_sectors/sector_forecast_r2.png`.

