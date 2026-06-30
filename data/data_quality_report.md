# Macro Panel Data Quality Report

- **Input:** `data/macro/NEW_MACRO_COMMODITY_PANEL.xlsx`
- **Window (inclusive):** 2013-10-18 -> 2026-06-01
- **Master trading-day grid length:** 3292
- **Stationary panel common start (NaN-free):** 2013-11-01
- **Rows dropped building stationary panel:** 218
  - Driven by residual NaNs (post-transform, pre-dropna) in: `gpr`=217. The 1 leading row is the differencing NaN; the rest are rows where `gpr` is missing — its ~10-day in-window head plus recurring ~23-calendar-day gaps (around Jan 1 / Nov 1 each year) that exceed the 10-trading-day bridge limit. These rows are dropped only from the convenience `macro_stationary` panel; the preferred downstream path (reindex `macro_levels_aligned` to factor dates, transform there) handles `gpr` NaNs independently and need not lose them.
- **macro_levels_raw rows:** 4610
- **macro_levels_aligned rows:** 3292
- **macro_stationary rows:** 3074
- **gpr/epu transform:** gpr=`level`, epu=`level` (default `level`; mean-reverting stationary uncertainty indices, editable to `diff`)

## Per-series QA

| series_id | sheet | ticker | field | transform | raw first | raw last | raw non-null | bridged days | longest bridge | min | max | mean | flat runs>7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fx_dxy | FX | DXY Curncy | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3282 | 10 | 1 | 79.09 | 114.1 | 96.79 | 0 |
| fx_bbdxy | FX | BBDXY Index | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3269 | 23 | 1 | 999.5 | 1353 | 1190 | 0 |
| fx_eurusd | FX | EURUSD Curncy | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3292 | 0 | 0 | 0.9594 | 1.393 | 1.142 | 0 |
| fx_usdcnh | FX | USDCNH Curncy | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3290 | 2 | 1 | 6.02 | 7.426 | 6.742 | 0 |
| fx_audusd | FX | AUDUSD Curncy | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3292 | 0 | 0 | 0.5743 | 0.9708 | 0.7286 | 0 |
| fx_usdbrl | FX | USDBRL Curncy | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3158 | 134 | 2 | 2.169 | 6.293 | 4.301 | 0 |
| fx_usdclp | FX | USDCLP Curncy | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3180 | 112 | 4 | 496.8 | 1048 | 755.6 | 0 |
| fx_usdcad | FX | USDCAD Curncy | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3292 | 0 | 0 | 1.029 | 1.458 | 1.302 | 0 |
| fx_usdnok | FX | USDNOK Curncy | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3291 | 1 | 1 | 5.892 | 11.71 | 8.919 | 0 |
| fx_usdzar | FX | USDZAR Curncy | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3289 | 3 | 1 | 9.737 | 19.8 | 15.1 | 0 |
| ust_10y | Rates_Infl | USGG10YR Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3289 | 3 | 1 | 0.5069 | 4.99 | 2.688 | 0 |
| ust_5y | Rates_Infl | USGG5YR Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3289 | 3 | 1 | 0.1902 | 4.956 | 2.337 | 0 |
| ust_2y | Rates_Infl | USGG2YR Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3289 | 3 | 1 | 0.1013 | 5.223 | 2.05 | 0 |
| tips_10y | Rates_Infl | GTII10 Govt | YLD_CNV_LAST | diff | 2013-10-18 | 2026-06-01 | 3288 | 4 | 1 | -1.204 | 2.519 | 0.6426 | 0 |
| tips_5y | Rates_Infl | GTII5 Govt | YLD_CNV_LAST | diff | 2013-10-18 | 2026-06-01 | 3288 | 4 | 1 | -1.981 | 2.609 | 0.3161 | 0 |
| be_10y | Rates_Infl | USGGBE10 Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3289 | 3 | 1 | 0.5521 | 3.037 | 2.046 | 0 |
| be_5y | Rates_Infl | USGGBE05 Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3289 | 3 | 1 | 0.1751 | 3.731 | 1.999 | 0 |
| inflsw_5y | Rates_Infl | USSWIT5 Curncy | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3237 | 55 | 1 | 0.2388 | 3.673 | 2.185 | 0 |
| inflsw_10y | Rates_Infl | USSWIT10 Curncy | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3258 | 34 | 1 | 0.813 | 3.23 | 2.279 | 0 |
| cny_10y | Rates_Infl | GCNY10YR Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 2873 | 419 | 8 | 1.597 | 4.7 | 3.012 | 7 |
| hy_oas | Credit_Vol | LF98OAS Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3181 | 111 | 1 | 2.5 | 11 | 4.003 | 0 |
| ig_oas | Credit_Vol | LUACOAS Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3181 | 111 | 1 | 0.71 | 3.73 | 1.165 | 10 |
| vix | Credit_Vol | VIX Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3202 | 90 | 1 | 9.14 | 82.69 | 17.94 | 0 |
| move | Credit_Vol | MOVE Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3204 | 88 | 1 | 36.62 | 198.7 | 78.2 | 0 |
| ovx | Credit_Vol | OVX Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3176 | 116 | 1 | 14.5 | 325.1 | 39.32 | 0 |
| gvz | Credit_Vol | GVZ Index | PX_LAST | diff | 2013-10-18 | 2026-06-01 | 3176 | 116 | 1 | 8.88 | 48.98 | 16.79 | 0 |
| spx | Equity_Growth | SPX Index | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3172 | 120 | 1 | 1742 | 7600 | 3563 | 0 |
| mxwo | Equity_Growth | MXWO Index | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3292 | 0 | 0 | 1469 | 4865 | 2528 | 0 |
| mxef | Equity_Growth | MXEF Index | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3292 | 0 | 0 | 688.5 | 1773 | 1066 | 0 |
| csi300 | Equity_Growth | SHSZ300 Index | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3068 | 224 | 6 | 2087 | 5808 | 3816 | 0 |
| shcomp | Equity_Growth | SHCOMP Index | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3068 | 224 | 6 | 1991 | 5166 | 3159 | 0 |
| hscei | Equity_Growth | HSCEI Index | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3103 | 189 | 3 | 4939 | 1.48e+04 | 9403 | 0 |
| xle | Equity_Growth | XLE US Equity | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3172 | 120 | 1 | 11.79 | 62.56 | 36.64 | 0 |
| bdiy | Freight | BDIY Index | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3147 | 145 | 6 | 290 | 5650 | 1457 | 0 |
| bdti | Freight | BIDY Index | PX_LAST | log_return | 2013-10-18 | 2026-06-01 | 3146 | 146 | 6 | 403 | 3737 | 951.3 | 2 |
| gpr | GPR | — | GPRD | level | 2013-11-01 | 2026-06-01 | 3773 | 380 | 10 | 9.49 | 540.8 | 123.6 | 38 |
| epu | EPR | — | daily_policy_index | level | 2013-10-18 | 2026-06-01 | 4610 | 0 | 0 | 3.32 | 1026 | 144.3 | 0 |

## Long bridged runs (> 7 trading days)

- `cny_10y`: longest bridged run = 8 trading days
- `gpr`: longest bridged run = 10 trading days

## Global flags

- **`cny_10y`**: Bloomberg pricing-source **seam at 2016-08-02** (pre-seam priced by PCS:BGNC). Treat R1/R2/early-R3 China-yield results with care.
- **`gpr`**: in-window coverage starts **2013-11-01**; ~10 leading trading days are NaN in levels and excluded from the stationary panel. It also has recurring **~23-calendar-day gaps around Jan 1 / Nov 1 each year** (≈16 trading days) that exceed the 10-trading-day bridge limit and therefore remain NaN on the master grid; those rows are dropped from the convenience `macro_stationary` panel (see dropped-row breakdown above). Shorter scattered gaps are bridged by ffill. Raw in-window count (3773) matches the reference exactly, so this is the source data, not a parse error.
- **`ig_oas`**: long flat runs observed (tight IG spreads at 2-dp precision); confirm not a stale-feed artifact (see flat-runs column).
- **`bdti`**: pulled under ticker **`BIDY Index`**, labelled 'Baltic Dirty Tanker.' Confirm `BIDY` vs `BDTI` if results look off.
- **`gpr`/`epu`**: calendar-daily sources reduced to the trading-day grid; defaulted to `level` transform.
- **FX direction is not uniform**: USD-per-FX for `fx_eurusd`, `fx_audusd`; FX-per-USD for the `fx_usd*` pairs — see manifest `direction` column for sign handling.

## Regime slices

| regime | name | start | end | rows |
|---|---|---|---|---|
| R1 | Late-QE calm | 2013-10-18 | 2014-06-19 | 154 |
| R2 | Oil crash / strong-USD bear | 2014-06-20 | 2016-02-11 | 397 |
| R3 | Reflation recovery | 2016-02-12 | 2018-09-30 | 654 |
| R4 | Trade war / late cycle | 2018-10-01 | 2020-02-19 | 329 |
| R5 | COVID shock & rebound | 2020-02-20 | 2020-12-31 | 210 |
| R6 | Reflation & Ukraine supercycle | 2021-01-01 | 2022-06-15 | 363 |
| R7 | Fed hiking / disinflation | 2022-06-16 | 2023-10-31 | 337 |
| R8 | Plateau & easing pivot | 2023-11-01 | 2024-12-31 | 278 |
| R9 | Recent / study-window events | 2025-01-01 | 2026-06-01 | 352 |
| **total** | | | | **3074** |

## Downstream interface (CCA)

- **Preferred path:** reindex `macro_levels_aligned` to the AE factor dates, then difference / log-return on that calendar, so innovations are computed across consecutive *factor* days (exactly correct).
- **Convenience path:** `macro_stationary` differences on the macro union grid; the two diverge only on days adjacent to calendar mismatches.
- CCA is scale-invariant (internal whitening); no standardization is required.

## Acceptance results

- [PASS] Exactly 37 value columns in manifest order — 37 cols
- [PASS] Every series raw date within window
- [PASS] Master grid length in 3290-3320 — 3292
- [PASS] macro_stationary has no NaN in value columns
- [PASS] regime non-null for all stationary rows
- [PASS] Regimes partition the stationary panel — sum=3074 rows=3074
- [PASS] No log_return on non-positive series
- [PASS] Output column order deterministic == manifest

**Overall: ALL PASS**
