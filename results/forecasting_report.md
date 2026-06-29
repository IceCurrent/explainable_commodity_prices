# Rolling Forecast Report

**Generated:** 2026-06-18  
**Reference:** Cerqueti et al. (2024) — vanilla autoencoder factor-augmented AR(1)  
**Script:** `python scripts/run_forecasts.py --window-size 252`

---

## Setup

| Item | Value |
|---|---|
| **Commodities** | 27 (merged 2013+ universe) |
| **Sample** | 2013-10-21 → 2026-06-01 (2,870 trading days) |
| **Estimation window M** | **252 days (~1 year)** |
| **OOS forecasts** | **2,618 windows** (first forecast ~2014-10) |
| **Models compared** | AR(1) baseline vs. vanilla AE factor-augmented AR(1) |
| **Latent factors K** | 5 |
| **Forecast lag p** | 1 |

### Per-window procedure
1. Take returns in window [d−252, …, d−1]; z-score inside the window (no look-ahead).
2. Train vanilla autoencoder; extract latent factors f_t.
3. Fit AR(1) and factor-augmented AR(1) per commodity on the window.
4. One-step-ahead forecast for date d; roll forward.

---

## Overall results

| Model | Pooled MSE | Pooled MAE | RMSE |
|---|---|---|---|
| **AR(1)** | **5.168×10⁻⁴** | 0.01401 | 0.02273 |
| Vanilla AE + AR(1) | 5.295×10⁻⁴ | 0.01429 | 0.02301 |
| **Δ (AE − AR1)** | **+2.5% worse** | +2.0% worse | +1.2% worse |

The AE-augmented model **underperforms** AR(1) on every commodity.

| Stat | Value |
|---|---|
| Commodities where AE wins (lower MSE) | **0 / 27** |
| Mean per-commodity MSE change | −1.6% (AE worse on average) |
| Total forecast observations | 70,686 (2,618 × 27) |
| Runtime | 2.6 min |

---

## Best & worst commodities (AE vs AR1)

With a 1-year window, **no commodity** shows AE improvement. Closest to parity:

| Commodity | MSE AR(1) | MSE AE | Change |
|---|---|---|---|
| Methanol | 4.31×10⁻⁴ | 4.31×10⁻⁴ | −0.06% |
| Diesel | 7.39×10⁻⁴ | 7.41×10⁻⁴ | −0.20% |
| Lithium | 1.01×10⁻⁴ | 1.01×10⁻⁴ | −0.58% |

**Largest AE deterioration:**

| Commodity | MSE AR(1) | MSE AE | Change |
|---|---|---|---|
| Nickel | 5.54×10⁻⁴ | 5.83×10⁻⁴ | −5.2% |
| Gasoline | 8.95×10⁻⁴ | 9.28×10⁻⁴ | −3.7% |
| Cotton | 2.94×10⁻⁴ | 3.03×10⁻⁴ | −3.1% |
| Gold | 1.16×10⁻⁴ | 1.19×10⁻⁴ | −3.0% |
| Brent | 7.26×10⁻⁴ | 7.47×10⁻⁴ | −2.9% |

**Highest volatility:** NaturalGas (MSE ~2.0×10⁻³), WTI, Gasoline, Diesel.  
**Lowest volatility:** Lithium, Gold, LiveCattle (MSE ~1.0–1.4×10⁻⁴).

---

## Comparison: 1-year vs 5-year window

| Metric | M = 252 (1y) | M = 1,260 (5y) |
|---|---|---|
| OOS windows | 2,618 | 1,610 |
| OOS period | ~2014–2026 | ~2018–2026 |
| Pooled MSE AR(1) | 5.168×10⁻⁴ | 6.002×10⁻⁴ |
| Pooled MSE AE | 5.295×10⁻⁴ | 6.010×10⁻⁴ |
| Δ (AE − AR1) | **+2.5%** | +0.14% |
| AE wins | **0 / 27** | 7 / 27 |

The 1-year window gives more OOS coverage (starts ~4 years earlier) but **worse** AE performance — likely because 252 days is too short to reliably estimate cross-sectional factor structure for 27 commodities.

---

## Interpretation

1. **AE uniformly hurts.** Every commodity shows equal or higher MSE with AE factors vs. plain AR(1).

2. **Short window is noisy.** With only ~1 year of training data, the AE cannot stably learn commodity co-movement patterns. Factors become estimation noise that degrades forecasts.

3. **Consistent with experiments.** Forecast-Shapley φ_k are all negative and ~10× larger in magnitude than the 5-year run (see `results/experiments_report.md`).

4. **AR(1) baseline is hard to beat** for one-step daily commodity return forecasting on this sample, regardless of window length.

---

## Outputs

| File | Contents |
|---|---|
| `results/forecasts.csv` | Date × commodity actuals and forecasts |
| `results/errors.csv` | Forecast errors (AR1, vanilla AE) |
| `results/summary.csv` | Per-commodity MSE/MAE summary |
| `results/summary.md` | Auto-generated table summary |

Full per-commodity table: `results/summary.csv` (27 rows + ALL aggregate).
