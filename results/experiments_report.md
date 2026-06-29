# AE Factor–Macro Mapping: Experiments Report

**Generated:** 2026-06-18  
**Directive:** `AE_factor_macro_mapping_directive.md`  
**Sample:** 27 commodities, daily log-returns, 2013-10-21 → 2026-06-01 (2,870 obs)  
**Rolling window:** **M = 252 trading days (~1 year)**  
**Macro panel:** Bloomberg workbook `data/macro_panel_data_raw.xlsx` → cleaned to `data/macro/` (28 regressors used; see §1)

This report summarizes every experiment run on the merged 2013+ dataset with a **1-year rolling estimation window**. Experiments E4 (regime overlay) and E5 (guided AE) were **not** run.

---

## 1. Data & prerequisites

### 1.1 Commodity panel
| Item | Value |
|---|---|
| Commodities | 27 (Energy × 8, Metals × 10, Agriculture × 9) |
| Date range | 2013-10-21 → 2026-06-01 |
| Source | `data/returns.csv` (from `Raw_Data_Final_Universe.xlsx`) |

### 1.2 Macro panel (cleaned)
Script: `python data/clean_macro_panel.py`  
Output folder: `data/macro/`

| Decision | Detail |
|---|---|
| **Included** | 28 series after dropping circular energy spot prices (WTI, Brent, NatGas) |
| **Excluded** | Global PMI, Global Manufacturing PMI (data starts ~2023) |
| **Groups** | FX (5), Rates (5), Risk (4), Inflation (5), Growth (6), Supply (3) |
| **Alignment** | Transforms at native frequency, forward-filled to daily calendar |
| **Mapping sample** | 2,618 dates (aligned factor as-of dates ∩ macro panel) |

### 1.3 AE factor extraction (input to all experiments)
Script: `python scripts/extract_factors.py --window-size 252`

| Parameter | Value |
|---|---|
| Model | Vanilla autoencoder (single-layer encoder/decoder, ReLU, K=5) |
| Rolling window M | **252 trading days (~1 year)** |
| OOS windows | **2,618** (first forecast ~2014-10) |
| Per-window output | In-window factor series (252×K), decoder weights (N×K), live factor f_t |
| Artifact | `results/factor_bundle.npz` |

Forecaster: AR(1) with p=1 and K=5 latent factors (paper's BIC choice).

---

## 2. E0 — Cross-window factor alignment & stability audit

**Script:** `python scripts/run_alignment.py`

### Method
- **Anchor 1:** Chained adjacent-window Hungarian matching on \|corr\|, sign fixed by matched correlation.
- **Anchor 2:** Decoder weight columns matched to reference by \|cosine\| similarity.
- **Reference window:** w₀ = 0.

### Results

| Metric | Value | Gate threshold |
|---|---|---|
| Median adjacent \|corr\| (anchor 1) | **0.999** | ≥ 0.8 → stable |
| Mean / min adjacent \|corr\| | 0.998 / 0.850 | |
| Median decoder \|cosine\| (anchor 2) | 0.929 | |
| Anchor agreement rate | **1.000** | |
| Permutation-event rate | 0.000 | |
| Sign-flip rate (all factors) | 0.000 | |

### Verdict: **STABLE**

Factors remain stable even with a short 1-year window. Cross-window pooled macro mapping is valid.

**Outputs:** `results/factors_aligned.csv`, `results/alignment_stability.csv`, `results/alignment_report.md`

---

## 3. Forecast-Shapley φ_k (factor importance for OOS accuracy)

**Script:** `python scripts/run_factor_shapley.py`

### Pooled results (all commodities, all 2,618 windows)

| Factor | φ_k (MSE reduction) | \|share\| | Rank |
|---|---|---|---|
| f4 | −3.17×10⁻⁶ | 25.0% | 1 |
| f5 | −2.73×10⁻⁶ | 21.5% | 2 |
| f1 | −2.65×10⁻⁶ | 20.9% | 3 |
| f3 | −2.17×10⁻⁶ | 17.1% | 4 |
| f2 | −1.97×10⁻⁶ | 15.5% | 5 |

| Aggregate | Value |
|---|---|
| Pooled MSE — AR(1) | 5.168×10⁻⁴ |
| Pooled MSE — full AE model | 5.295×10⁻⁴ |
| Sum φ_k (efficiency check) | −1.268×10⁻⁵ |

### Interpretation
All φ_k are **negative** — every factor increases forecast error on average. Magnitudes are ~10× larger than with the 5-year window but the sign is unchanged: AE factors **hurt** one-step forecasts. f4 carries the largest marginal harm.

**Outputs:** `results/forecast_shapley.csv`, `results/forecast_shapley_ranking.csv`

---

## 4. E1 — Linear spanning regression + incremental-content test

**Script:** `python scripts/run_macro_mapping.py`  
**Sample:** 2,618 aligned factor dates × 28 macro regressors

### 4.1 Per-factor OLS spanning (HAC standard errors)

| Factor | R² | Dominant macro drivers |
|---|---|---|
| f1 | 0.061 | credit_hy_oas, ust_10y, vix |
| f2 | 0.060 | credit_hy_oas, cpi_yoy, breakeven_10y |
| f3 | 0.087 | credit_ig_oas, cpi_yoy, credit_hy_oas |
| f4 | **0.176** | ust_10y, usd_twi, tips_10y |
| f5 | **0.200** | ust_10y, tips_10y, ppi_yoy |

**Pattern:** R² is **lower** than with the 5-year window (was 0.10–0.23; now 0.06–0.20). Credit/risk explain f1–f3; rates/TWI explain f4–f5. Macro explains even less of the short-window factor variation.

### 4.2 Bai–Ng spanning diagnostic

| Canonical correlation | Value |
|---|---|
| 1st | 0.497 |
| 2nd | 0.312 |
| 3rd | 0.236 |
| 4th | 0.154 |
| 5th | 0.112 |
| **Min** | **0.112** |
| Mean | 0.262 |

**Conclusion:** Factor space is **not spanned** by macro (max canonical corr 0.50, down from 0.58 with 5y window).

### 4.3 Incremental-content test

| Variant | RMSE (AR1) | RMSE (factor-augmented) | Δ vs AR1 |
|---|---|---|---|
| **Original f** | 0.022734 | 0.023011 | +0.000277 (worse) |
| **Macro-spanned f̂^M** | 0.022734 | 0.024412 | +0.001678 (much worse) |
| **Residual u** | 0.022734 | 0.022994 | +0.000261 (worse) |

All variants hurt vs AR(1). Macro-spanned reconstruction is especially harmful. Share of forecast value: **undefined** (original factors don't beat AR(1)).

**Outputs:** `results/e1_spanning_summary.csv`, `results/e1_bai_ng_spanning.json`, `results/e1_incremental_rmse.csv`

---

## 5. E2 — Nonlinear macro mapping + macro-SHAP

| Factor | R² (nonlinear) | R² (linear) | Nonlinearity premium | Top macro (SHAP) |
|---|---|---|---|---|
| f1 | −0.085 | −0.162 | +0.077 | usd_twi, usd_cny, usd_brl |
| f2 | −0.119 | −0.114 | −0.006 | tips_10y, usd_dxy, usd_cny |
| f3 | −0.154 | −0.052 | −0.102 | usd_twi, usd_aud, usd_brl |
| f4 | −0.007 | −0.027 | +0.020 | usd_twi, tips_10y, usd_cny |
| f5 | −0.075 | −0.048 | −0.028 | usd_twi, ust_10y, tips_10y |

**Interpretation:** Negative OOS R² for all factors. Nonlinearity premiums are small or negative. USD block dominates SHAP. **E1 linear mapping remains the defensible layer.**

**Output:** `results/e2_nonlinear_summary.csv`

---

## 6. E3 — Decoder-Jacobian commodity fingerprint

**Script:** `python scripts/run_jacobian.py`

### Sector intensity (mean |loading| per commodity in sector)

| Sector | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|
| PreciousMetals | **0.325** | **0.385** | 0.282 | 0.172 | 0.123 |
| Agriculture | 0.236 | 0.218 | **0.289** | 0.265 | **0.307** |
| BaseMetals | 0.189 | 0.174 | 0.172 | **0.289** | 0.205 |
| Energy | 0.224 | 0.193 | 0.192 | 0.194 | 0.194 |

### Per-factor identity

| Factor | Dominant sector | Top commodities | E1 macro match? |
|---|---|---|---|
| **f1** | PreciousMetals | HRWWheat, HeatingOil, ThermalCoal, Silver | Partial — credit/rates/VIX |
| **f2** | PreciousMetals | Gold, Methanol, Soybeans, Silver | Partial — credit/inflation |
| **f3** | Agriculture | SoybeanOil, Coffee, Silver, Gasoline | Partial — credit spreads |
| **f4** | BaseMetals | Copper, Corn, HRWWheat, Zinc | **Yes** — ust_10y, tips_10y, usd_twi |
| **f5** | Agriculture | SoybeanOil, Copper, Soybeans, Methanol | Partial — rates (ust_10y, tips_10y) |

Loadings remain diffuse (top-2 concentration 0.15–0.17). f4 still shows the best E1/E3 triangulation.

**Outputs:** `results/jacobian_loadings.csv`, `results/jacobian_fingerprint.md`

---

## 7. Cross-experiment synthesis

| Factor | (1) R² / concentration | (2) Triangulation | (3) Forecast φ_k | (4) E0 stability |
|---|---|---|---|---|
| f1 | Very low (0.06) | Weak | Harmful | Stable |
| f2 | Very low (0.06) | Weak | Harmful | Stable |
| f3 | Low (0.09) | Weak | Harmful | Stable |
| f4 | Moderate (0.18) | **Best** (base metals ↔ rates) | Most harmful | Stable |
| f5 | Moderate (0.20) | Mixed (agri Jacobian, rates macro) | Harmful | Stable |

### Headline findings (1-year window)

1. **E0 passes.** Factors are stable across 2,618 adjacent windows (median \|corr\| = 0.999).

2. **Macro mapping weaker than 5y.** R² drops to 0.06–0.20; max canonical correlation falls to 0.50. Short-window factors are even less macro-explainable.

3. **Forecast performance worse.** AE factors uniformly hurt forecasts — **0/27 commodities** beat AR(1) on MSE (vs 7/27 with 5y window). Pooled MSE +2.5% vs AR(1).

4. **Best partial identity still f4.** Base-metals Jacobian ↔ rate/TWI macro, but f4 also has the largest negative forecast-Shapley.

5. **Shorter window = noisier factors.** Less training data per window (252 vs 1,260 days) produces factors that are harder to map to macro and less useful for prediction.

---

## 8. Comparison: 1-year vs 5-year window

| Metric | M = 252 (1y) | M = 1,260 (5y) |
|---|---|---|
| OOS windows | 2,618 | 1,610 |
| First forecast | ~2014-10 | ~2018-10 |
| E0 median \|corr\| | 0.999 | 0.998 |
| Max R² (E1) | 0.200 (f5) | 0.231 (f4) |
| Max canonical corr | 0.497 | 0.581 |
| Pooled MSE Δ (AE−AR1) | **+2.5%** | +0.14% |
| Commodities AE wins | **0 / 27** | 7 / 27 |
| All φ_k sign | Negative | Negative |

---

## 9. Reproduction commands

```bash
python data/clean_macro_panel.py
python scripts/extract_factors.py --window-size 252
python scripts/run_alignment.py
python scripts/run_factor_shapley.py
python scripts/run_jacobian.py
python scripts/run_macro_mapping.py
python scripts/run_forecasts.py --window-size 252
```

---

## 10. Output file index

| File | Experiment |
|---|---|
| `results/factor_bundle.npz` | Factor extraction (252d) |
| `results/factors_aligned.csv` | E0 |
| `results/forecast_shapley_ranking.csv` | Forecast-Shapley |
| `results/e1_spanning_summary.csv` | E1 spanning |
| `results/e1_bai_ng_spanning.json` | E1 Bai–Ng |
| `results/e1_incremental_rmse.csv` | E1 incremental |
| `results/e2_nonlinear_summary.csv` | E2 |
| `results/jacobian_fingerprint.md` | E3 |
