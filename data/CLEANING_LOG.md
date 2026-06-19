# Commodity cleaning provenance log

Automated record of every parsing, alignment, guard, and transform step applied by `clean_commodities.py`.

## Run metadata

| Field | Value |
| --- | --- |
| Generated | 2026-06-18T10:55:03 |
| Source workbook | `Raw_Data_Final_Universe.xlsx` |
| Sheet | `Sheet1` |
| Outputs | `prices.csv`, `returns.csv` |

---


## Input and block layout

| Metric | Value |
| --- | --- |
| Sheet | Sheet1 |
| Raw shape (rows × cols) | 5537 × 86 |

Detected **29** `(date, price)` blocks from the header row (stride = `3` columns).

| # | Name | Ticker | Sector | Date col | Price col | Header |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Coffee | KC1 | Agri | 0 | 1 | Coffee (KC1) |
| 2 | SoybeanOil | BO1 | Agri | 3 | 4 | Soybean Oil (BO1) |
| 3 | Corn | C 1 | Agri | 6 | 7 | Corn (C 1) |
| 4 | Cotton | CT1 | Agri | 9 | 10 | Cotton (CT1) |
| 5 | HRWWheat | KC | Agri | 12 | 13 | HRW Wheat (KC) |
| 6 | LeanHogs | LH1 | Agri | 15 | 16 | Lean Hogs (LH1) |
| 7 | LiveCattle | LC1 | Agri | 18 | 19 | Live Cattle (LC1) |
| 8 | Soybeans | S1 | Agri | 21 | 22 | Soybeans (S1) |
| 9 | Sugar | SB1 | Agri | 24 | 25 | Sugar (SB1) |
| 10 | Brent | CO1 | Energy | 27 | 28 | Brent Crude Oil (CO1) |
| 11 | Diesel | QS1 | Energy | 30 | 31 | Diesel (QS1) |
| 12 | NaturalGas | NG1 | Energy | 33 | 34 | Natural Gas (NG1) |
| 13 | WTI | CL1 | Energy | 36 | 37 | WTI Crude Oil (CL1) |
| 14 | Gasoline | XB1 | Energy | 39 | 40 | Gasoline (XB1) |
| 15 | HeatingOil | HO1 | Energy | 42 | 43 | Heating Oil (HO1) |
| 16 | ThermalCoal | XW1 | Energy | 45 | 46 | Thermal Coal (XW1) |
| 17 | CokingCoal | CKCA | Energy | 48 | 49 | Coking Coal (CKCA) |
| 18 | Methanol | ZME1 | Energy | 51 | 52 | Methanol (ZME1) |
| 19 | Propane | BAPA | Energy | 54 | 55 | Propane (BAPA) |
| 20 | Gold | GC1 | Metals | 57 | 58 | Gold (GC1) |
| 21 | Silver | SI1 | Metals | 60 | 61 | Silver (SI1) |
| 22 | Copper | HG1 | Metals | 63 | 64 | Copper (HG1) |
| 23 | Aluminium | LA1 | Metals | 66 | 67 | Aluminum (LA1) |
| 24 | Nickel | LN1 | Metals | 69 | 70 | Nickel (LN1) |
| 25 | Zinc | LX1 | Metals | 72 | 73 | Zinc (LX1) |
| 26 | Platinum | PL1 | Metals | 75 | 76 | Platinum (PL1) |
| 27 | HRCSteel | — | Metals | 78 | 79 | HRC Steel |
| 28 | SGXIronOre | — | Metals | 81 | 82 | SGX Iron Ore |
| 29 | Lithium | — | Metals | 84 | 85 | Lithium |


### Per-commodity parse summary

| # | Name | Ticker | Sector | Obs | Start | End | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Coffee | KC1 | Agri | 5395 | 2005-01-03 | 2026-06-15 | — |
| 2 | SoybeanOil | BO1 | Agri | 5403 | 2005-01-03 | 2026-06-15 | — |
| 3 | Corn | C 1 | Agri | 5403 | 2005-01-03 | 2026-06-15 | — |
| 4 | Cotton | CT1 | Agri | 5201 | 2005-10-04 | 2026-06-01 | — |
| 5 | HRWWheat | KC | Agri | 5403 | 2005-01-03 | 2026-06-15 | — |
| 6 | LeanHogs | LH1 | Agri | 5404 | 2005-01-03 | 2026-06-15 | — |
| 7 | LiveCattle | LC1 | Agri | 5403 | 2005-01-03 | 2026-06-15 | — |
| 8 | Soybeans | S1 | Agri | 5403 | 2005-01-03 | 2026-06-15 | — |
| 9 | Sugar | SB1 | Agri | 5394 | 2005-01-03 | 2026-06-15 | — |
| 10 | Brent | CO1 | Energy | 5529 | 2005-01-04 | 2026-06-15 | — |
| 11 | Diesel | QS1 | Energy | 5532 | 2005-01-04 | 2026-06-15 | — |
| 12 | NaturalGas | NG1 | Energy | 5402 | 2005-01-03 | 2026-06-15 | — |
| 13 | WTI | CL1 | Energy | 5403 | 2005-01-03 | 2026-06-15 | — |
| 14 | Gasoline | XB1 | Energy | 5213 | 2005-10-03 | 2026-06-15 | — |
| 15 | HeatingOil | HO1 | Energy | 5402 | 2005-01-03 | 2026-06-15 | — |
| 16 | ThermalCoal | XW1 | Energy | 4518 | 2008-12-05 | 2026-06-15 | — |
| 17 | CokingCoal | CKCA | Energy | 423 | 2015-09-17 | 2026-06-15 | — |
| 18 | Methanol | ZME1 | Energy | 3552 | 2011-10-28 | 2026-06-15 | — |
| 19 | Propane | BAPA | Energy | 1092 | 2022-02-10 | 2026-06-15 | — |
| 20 | Gold | GC1 | Metals | 5403 | 2005-01-03 | 2026-06-15 | — |
| 21 | Silver | SI1 | Metals | 5403 | 2005-01-03 | 2026-06-15 | — |
| 22 | Copper | HG1 | Metals | 5403 | 2005-01-03 | 2026-06-15 | — |
| 23 | Aluminium | LA1 | Metals | 5403 | 2005-01-04 | 2026-06-12 | — |
| 24 | Nickel | LN1 | Metals | 5392 | 2005-01-04 | 2026-06-12 | — |
| 25 | Zinc | LX1 | Metals | 5394 | 2005-01-04 | 2026-06-12 | — |
| 26 | Platinum | PL1 | Metals | 5402 | 2005-01-03 | 2026-06-15 | — |
| 27 | HRCSteel | — | Metals | 4436 | 2008-10-20 | 2026-06-15 | — |
| 28 | SGXIronOre | — | Metals | 3073 | 2013-10-18 | 2026-06-15 | — |
| 29 | Lithium | — | Metals | 3682 | 2011-06-02 | 2026-06-15 | — |

Parsed **29** commodity series.


## Calendar alignment

> **Rule:** Outer-join all series on date, then keep only dates where every commodity has a non-missing price (complete-case intersection).

The following parsed series are **excluded from alignment** because their sparse or short calendars would collapse the shared panel: `CokingCoal`, `Propane`.

| Step | Result |
| --- | --- |
| After outer join | 5635 rows × 27 cols |
| Rows dropped (any missing price) | 2762 |
| Rows retained (intersection) | 2873 |
| Aligned matrix shape | 2873 × 27 |
| Date range | 2013-10-18 → 2026-06-01 |


## Non-positive price guard

> **Rule:** Any price ≤ 0 is replaced with `NaN`; each replacement is logged below.

Applied **before** log-returns so `ln(price)` is never taken on non-positive values.

| Commodity | Date | Original value | Replacement |
| --- | --- | --- | --- |
| WTI | 2020-04-20 | -37.63 | `NaN` |

**Total guard actions:** 1


## Log-returns

Formula per commodity: `r_t = ln(p_t / p_{t-1})`

| Date | Reason | Affected commodities | Action |
| --- | --- | --- | --- |
| 2013-10-18 | First calendar row | All commodities | No prior price for returns |
| 2020-04-20 | NaN log-return | WTI | Row removed from return matrix |
| 2020-04-21 | NaN log-return | WTI | Row removed from return matrix |

| Metric | Value |
| --- | --- |
| Rows before return drop | 2872 |
| Rows dropped (NaN returns) | 2 |
| Final return rows | 2870 |
| Shape (before z-score) | 2870 × 27 |

Return date range: **2013-10-21** → **2026-06-01**


## Per-commodity z-score standardization

> **Rule:** Each column is transformed as `(r - mean) / std` using the final return sample (population std, `ddof=0`).

Full-sample scaling is used here for **representation learning** only. For out-of-sample forecasting, refit the scaler inside each rolling window to avoid look-ahead bias.

| Commodity | Mean (raw returns) | Std (raw returns) |
| --- | --- | --- |
| Coffee | 0.000306 | 0.023208 |
| SoybeanOil | 0.000236 | 0.016657 |
| Corn | 0.000016 | 0.016331 |
| Cotton | -0.000032 | 0.016947 |
| HRWWheat | -0.000073 | 0.019656 |
| LeanHogs | -0.000027 | 0.027453 |
| LiveCattle | 0.000238 | 0.012399 |
| Soybeans | -0.000030 | 0.013790 |
| Sugar | -0.000083 | 0.019119 |
| Brent | 0.000079 | 0.025736 |
| Diesel | 0.000138 | 0.025933 |
| NaturalGas | -0.000072 | 0.043275 |
| WTI | 0.000178 | 0.029853 |
| Gasoline | 0.000165 | 0.028753 |
| HeatingOil | 0.000159 | 0.025181 |
| ThermalCoal | 0.000194 | 0.022811 |
| Methanol | 0.000015 | 0.020420 |
| Gold | 0.000429 | 0.010579 |
| Silver | 0.000440 | 0.021443 |
| Copper | 0.000257 | 0.015616 |
| Aluminium | 0.000259 | 0.013668 |
| Nickel | 0.000100 | 0.022494 |
| Zinc | 0.000226 | 0.016409 |
| Platinum | 0.000114 | 0.019409 |
| HRCSteel | 0.000193 | 0.018720 |
| SGXIronOre | -0.000072 | 0.028108 |
| Lithium | 0.000507 | 0.011521 |

Standardized return matrix shape: **2870 × 27**


## Final outputs

| File | Shape (rows × cols) | Path |
| --- | --- | --- |
| prices.csv | 2873 × 27 | `/Users/shreyanshsharma/Desktop/Resume Projects/Summer Project/explainable_commodity_prices/data/prices.csv` |
| returns.csv | 2870 × 27 | `/Users/shreyanshsharma/Desktop/Resume Projects/Summer Project/explainable_commodity_prices/data/returns.csv` |


### Aligned price panel (post-guard)

| Commodity | Non-null obs | Start | End | NaN count |
| --- | --- | --- | --- | --- |
| Coffee | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| SoybeanOil | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Corn | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Cotton | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| HRWWheat | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| LeanHogs | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| LiveCattle | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Soybeans | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Sugar | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Brent | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Diesel | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| NaturalGas | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| WTI | 2872 | 2013-10-18 | 2026-06-01 | 1 |
| Gasoline | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| HeatingOil | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| ThermalCoal | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Methanol | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Gold | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Silver | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Copper | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Aluminium | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Nickel | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Zinc | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Platinum | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| HRCSteel | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| SGXIronOre | 2873 | 2013-10-18 | 2026-06-01 | 0 |
| Lithium | 2873 | 2013-10-18 | 2026-06-01 | 0 |


### Return sample (post row-drop)

All **27** commodities share **2870** return observations from **2013-10-21** → **2026-06-01**.

