# Deep Analysis — Why the Explainable-AE Commodity Factor Models Do Not Forecast

**Branch:** `deep-analysis` · **Scripts:** `analysis/da01–da09` · **Artifacts:** `results/deep_analysis/`, `figures/deep_analysis/` · **Date:** 2026-07-08

---

## Verdict (TL;DR)

The forecasting stage fails because **there is no exploitable lagged signal to find, and the
pipeline is correct in saying so**. Three facts, each established independently:

1. **The machinery works.** In an end-to-end positive control that plants a known signal on the
   real data, the project's own engine detects any pooled predictive R² ≥ 0.25% with ~100%
   power and correct test size (0/16 false alarms at zero injection). Fed volatility targets
   instead of returns, the identical engine finds textbook vol clustering at OOS R² of
   +2.9%/+17%/+29% (h=1/5/21, CW p ≈ 10⁻⁸). A broken pipeline can do neither.

2. **The economics is genuinely null at short horizons.** A model-free scan of all 37 macro
   series × 21 commodities × horizons, calibrated against a dependence-preserving placebo,
   finds *no* lagged predictive association once a 24-hour information embargo is imposed
   (179 rejections → 51, placebo mean 40, p = 0.23). The macro-spanned factor state has
   AC(1) ≈ 0.01–0.08 — it is an innovation process, not a persistent state; there is nothing
   for tomorrow to inherit.

3. **Every apparent "positive" in the project is one artifact: asynchronous closing times.**
   US macro closes (FX/rates ~17:00 ET, equities 16:00) postdate commodity settlements
   (LME ~12:00, COMEX ~13:30, CME/ICE ~14:30 ET). Day-t macro therefore contains up to five
   hours of information from *inside* the day-(t+1) settlement-to-settlement commodity return.
   The strength of the fake "lag-1 predictability" scales almost perfectly with each contract's
   after-settle window (corr +0.78; Zinc/Aluminium worst, energy least), it vanishes under a
   one-day embargo, and it fully explains the substitution-arm Clark–West anomaly
   (p = 0.008 same-day → p = 0.77 embargoed).

The defensible conclusion is not "the model is bad" but sharper: **daily commodity futures
returns are unforecastable from lagged macro information at any effect size ≥ ~0.25% pooled R²
on this sample — consistent with near-efficient markets — while contemporaneous macro
co-movement (the mapping stage) and volatility dynamics are real and detectable.** The
monthly/quarterly rows of the horizon ladder are underpowered by construction (54 and 18
effective observations) and should be reported as "cannot be tested on this sample," not as
evidence of absence.

---

## 1. What was investigated

The project compresses 21 clean commodity return series into 5 AE latents, maps them to a
37-variable macro panel by CCA (finding ρ_OOS ≈ 0.64/0.49 on two dimensions — real,
interpretable contemporaneous co-movement), then asks the factor state to forecast h-period
returns in a factor-augmented AR(1) with exact-Shapley attribution. The forecast stage returns
negative OOS R² against both AR(1) and the zero forecast at every horizon, with the
pre-registered gate failing everywhere (`results/forecast_pbsv/macro_transmission_by_horizon.csv`).
The question posed: **why, exactly?**

Hypotheses considered from the start (none privileged):

| # | Hypothesis | Final assessment |
|---|---|---|
| H1 | Implementation bug destroys or hides signal | **Rejected** (§3) |
| H2 | Data corrupted / misaligned / stale | **Rejected for the retained panel**; predictor-side staleness exists but cannot rescue the null (§2) |
| H3 | AE compression loses forecastable content | **Rejected** (§6) |
| H4 | Statistical power too low to see plausible signal | **Rejected at h=1/5; TRUE at h=21/63** (§4) |
| H5 | The relationship is genuinely absent at daily lag (EMH) | **Supported — dominant explanation** (§5) |
| H6 | Predictability exists in second moments, not means | **Confirmed as a fact of the data** (§7) |
| H7 | Apparent positives are microstructure timing artifacts | **Confirmed; unifies every anomaly** (§5) |

---

## 2. Data forensics (da01, da02-A)

*Script:* `analysis/da01_data_forensics.py`, `analysis/da02_lag_scan_calibrated.py`

- The retained 21-series panel is clean: all zero-return fractions ≤ 1.7%, longest stale run
  ≤ 2 days, |AC(1)| < 0.05 everywhere.
- The data are economically real — every cross-asset sanity anchor holds with the expected
  sign: WTI–Brent +0.90, Gold–Silver +0.78, Gold–DXY −0.36, Copper–AUDUSD +0.38, WTI–XLE
  +0.53, Gold–10y TIPS yield −0.29.
- A **date-shift audit** (correlation profile of each suspicious macro series against SPX at
  offsets −3..+3) shows `ig_oas`, `hy_oas`, `vix`, `move` all peak at offset 0: nothing is
  recorded on the wrong date. Freight (`bdiy`, `bdti`) and `gpr`/`epu` are slow indices with
  no meaningful SPX correlation at any offset.
- Predictor-side warts that matter for interpretation, not for the null: `ig_oas` has 37%
  zero-change days (stale index marks), `bdiy`/`bdti` AC(1) ≈ 0.6, `gpr`/`epu` are persistent
  levels (AC(1) ≈ 0.7). The aligned macro∩commodity calendar is 2,691 of 2,870 days.

**Conclusion:** the null is not a data-quality mirage. (The six stale series the project
already removed — Diesel etc. — were the *previous* generation of artifacts; the curation was
correct.)

## 3. The pipeline is correct: injected-signal positive control (da04)

*Script:* `analysis/da04_injected_signal_power.py` · *Figure:* `figures/deep_analysis/power_curve.png`

We planted `r*_{i,t+1} = r_{i,t+1} + c·g_i·x_t` (x = random fixed combination of the real
canonical-variate state, g = heterogeneous loadings, c set to hit a target pooled R²) and ran
the **actual** engine and the **actual** pooled Clark–West test:

| injected R² | detection rate | median CW p | mean OOS R² vs AR(1) |
|---|---|---|---|
| 0 (null) | 0/16 | 0.33 | −0.31% |
| 0.10% | 37.5% | 0.11 | −0.26% |
| 0.25% | 100% | 0.008 | −0.07% |
| 0.50% | 75%¹ | 7×10⁻⁵ | +0.07% |
| 1.00% | 100% | 7×10⁻⁹ | +0.56% |
| 2.00% | 100% | 7×10⁻¹¹ | +1.57% |

¹ one unlucky draw of the random signal direction; median p far below 0.05.

Three implications. (i) *Size is correct*: no false alarms under the null. (ii) *The minimal
detectable effect at h=1 is ≈ 0.2–0.3% pooled OOS R²* — a tiny effect by any absolute
standard. (iii) *The accounting is honest*: recovered OOS R² ≈ injected R² − 0.3%, i.e. the
engine charges ~0.3% estimation cost for 5 extra regressors and returns the rest — exactly the
observed −0.31% when nothing is injected. **The observed "failure" (−0.3% OOS R²) is the
estimation cost of a correct model of nothing.**

The pipeline's own acceptance checks (Shapley efficiency ≤ 1e-12, basis invariance, scrambled
future no-lookahead probe) all pass independently.

## 4. Power: which horizons can this design even test? (da04, da07)

With 1,148 OOS days at h=1 the empirical MDE is ≈ 0.25% pooled R². Scaling by effective
sample size (√(1148/n_eff)), the approximate MDEs are:

| horizon | effective n | approximate MDE (pooled R², daily-equivalent) |
|---|---|---|
| 1 | 1,148 | ~0.25% |
| 5 | 229 | ~0.6% |
| 21 | 54 | ~1.2% |
| 63 | 18 | ~2%+ |

At h=1/5 these are demanding thresholds and the tests are informative. At h=21/63 the design
is fighting 54 and 18 effective observations on a 2021–2026 OOS window: plausible
macro-transmission effects at monthly/quarterly frequency in the literature are *smaller than
the detection floor*. The horizon ladder's monthly/quarterly rows are therefore
**uninformative, not negative** — the honest phrasing for the report is "cannot be answered on
a 13-year daily panel with a 60/40 split," and answering it would need decades of data at
monthly frequency (§9). (The stray marginal p-values that appear at h=63 in some arms, e.g.
`cw_p_nonoverlap` = 0.048 on 19 non-overlapping targets, are noise at these sample sizes; the
placebo-calibrated p there is 0.20.)

## 5. The central finding: the only "signal" in the data is a timing artifact (da02, da03, da05)

*Scripts:* `analysis/da02_lag_scan_calibrated.py`, `da03_timing_overlap.py`,
`da05_substitution_anomaly.py` · *Figures:* `embargo.png`, `settle_fingerprint.png`

### 5.1 A model-free scan does reject the global null — at first

All 37 × 21 macro→commodity predictive regressions with Newey–West t-stats, calibrated
against 200 **joint circular shifts** of the entire macro panel (preserving every internal
auto- and cross-correlation): 180 rejections at h=1 vs placebo mean 43 (p = 0.005). So a naive
researcher would conclude macro *does* forecast commodities daily. The top "predictors" are
FX (`fx_audusd` 13/21 commodities, `fx_usdcnh` 11/21), TIPS yields, VIX, XLE.

### 5.2 Three discriminating tests attribute all of it to asynchronous closes

1. **24h embargo** (predict t+1 from macro at t−1): 179 → 51 rejections, placebo mean 40,
   **p = 0.23**. Genuine multi-day information diffusion would decay smoothly; mechanical
   overlap dies instantly. It died instantly.
2. **Settle-time fingerprint**: each contract's mean |t| over after-US-close predictors
   correlates **+0.78** with its after-settle window length. LME metals (noon ET settle,
   5h window): Zinc 2.79, Aluminium 2.63. CME energy (14:30 settle, 2.5h): WTI 0.54,
   NatGas 0.62. No economic diffusion story predicts this gradient; clock arithmetic does.
3. **Predictors that close before US settlements** (Chinese equities, China 10y, Baltic
   freight): 3 rejections of 126 pairs — *below* placebo mean (p = 0.85). Where overlap is
   impossible, "predictability" is exactly zero.

Rejection rate by predictor group: after-close 27.9% → 6.7% under embargo; before-close 2.4%;
(gpr/epu 14.3% → 11.9%, small counts on persistent levels — see §8 residuals).

### 5.3 The substitution-arm anomaly is the same artifact inside the pipeline

The one nominally significant number in the entire forecast stage was the macro-substitution
'all' arm: pooled CW p = 0.0077 (placebo-calibrated 0.020) while its MSE improvement is
*negative*. Replacing the canonical variates (functions of commodity **settlements**) by their
macro proxies (functions of **17:00 ET closes**) injects the after-settle window into the
state. Under one extra day of lag the CW statistic collapses: **+2.42 (p = 0.008) → −0.73
(p = 0.77)**. Its per-commodity fingerprint ranks Zinc (p = 0.0013) and Aluminium (0.0013)
first, correlation of CW stat with after-settle hours +0.55. The beta-VAE arm replicates the
same artifact (sub_cw_p_all = 0.016). This is contemporaneous co-movement measured across
misaligned clocks — not macro forecasting commodities. A caveat has been added to the
generated forecast reports.

The same mechanism operates *within* the commodity panel: the factor state contains
14:30-settled components (energy, grains) that mechanically "lead" the noon-settled LME
contracts. Zinc is the only per-commodity CW rejection of the factor-state model at h=1
(p = 0.034), and the correlation of per-commodity CW stats with after-settle hours is +0.35.
The pipeline's pooled gate (LOCO + placebo calibration) correctly refused to promote this —
the earlier "Diesel staleness" episode and this one are the same lesson at different clocks.

### 5.4 Why the factor state cannot forecast even in principle here

The forecastability chain multiplies to ~zero (da07):

- 5 latents capture ~55% (train) / 50% (OOS) of daily return variance — fine;
- of the state, only cv1 is meaningfully macro-spanned (train purged-CV ρ² = 0.48, then
  0.19/0.07/0.02/0.00) — modest;
- the state's own persistence is **AC(1) = 0.011 (cv1), −0.007, −0.070, 0.024, 0.015**; the
  macro-side variates: 0.077, 0.050, −0.029, −0.086, 0.148. Daily macro *changes* — which is
  what stationarity transforms deliver — are news, and news does not repeat tomorrow.
- Consequently the full-look-ahead, in-sample pooled R² of next-day returns on the state is
  **0.37%**, of which ~0.17% is the mechanical K/T fitting floor (5/2870) and part of the rest
  is the LME overlap just described. The honest predictive content is ≪ the 0.25% MDE, and
  OOS the 0.3% estimation cost swamps it: negative OOS R² is then *arithmetic*.

**Economically:** contemporaneous spanning + white-noise state = EMH. Commodity prices load on
dollar/real-rate/risk news within the same session; whatever is priced by 17:00 ET is in the
next settlement print by construction of the artifact above, and nothing measurable is left
for horizon-1 forecasting. The project's mapping stage and its forecasting null are not in
tension — they are the same fact seen from two sides.

## 6. The AE is not the bottleneck (da08 + synthetic track)

Identical engine, three state constructions: AE canonical-variate state (CW p = 0.33), PCA-5
(p = 0.20), 3 sector means (p = 0.24) — the same null everywhere, and PCA-5 carries slightly
*more* train variance (57.7%) than the AE latents. Combined with the model-free scan (§5.1,
which involves no factor model at all and finds nothing genuine), H3 is dead: there was no
signal for the AE to destroy. This matches the earlier synthetic-recovery finding that
apparent AE explainability loss is coordinate splitting, not information loss.

## 7. The positive control that matters to an economist: vol (da06)

*Figure:* `figures/deep_analysis/vol_vs_mean.png`

Feeding |r| through the identical engine (AR-in-vol vs expanding-mean benchmark):

| horizon | OOS R², vol persistence | CW p | state adds on top? |
|---|---|---|---|
| 1 | +2.9% | 7×10⁻⁹ | no (−0.14%, p = 0.10) |
| 5 | +17.4% | 1×10⁻⁸ | no (−0.25%, p = 0.24) |
| 21 | +29.2% | 4×10⁻⁵ | no (−0.52%, p = 0.41) |

Same data, same code, same test — overwhelming detection of the predictability that actually
exists (volatility clustering), and a clean null for the one that does not (mean returns).
This is the economically sensible shape of results for liquid futures markets: **second
moments forecastable, first moments not.**

## 8. Residual uncertainties (honest list)

- **Settlement-time table**: institutional closing times are approximate (LME evaluations,
  Bloomberg PX_LAST snapshot conventions). The *evidence* is the monotone empirical gradient
  and the embargo collapse, which do not depend on any single entry being exact.
- **LeanHogs** ranks high in the substitution fingerprint (CW p = 0.006) despite a mid-pack
  window — thin, idiosyncratic market; consistent with noise among 21 draws, worth watching.
- **gpr/epu**: 6/42 same-day rejections barely attenuating under embargo (5/42). Persistent
  uncertainty levels could carry genuine slow risk-premium variation, but the counts are tiny
  and inside the joint placebo; classifying this as "suggestive at best" (any follow-up needs
  long samples and level-appropriate econometrics — see spurious-persistent-regressor risk).
- **h=63 marginalia**: isolated p ≈ 0.05 entries on 18–19 effective observations are expected
  under the null across this many tests; the placebo calibrations agree.
- The MDE scaling across horizons is √n arithmetic, not a full simulation per horizon.

## 9. Probability assessment

*P(hypothesis is the dominant explanation of the observed forecasting failure):*

| Hypothesis | Probability | Basis |
|---|---|---|
| H5+H7: genuine short-horizon null; positives = closing-time artifacts | **~0.90** | embargo collapse, settle-time gradient, before-closer null, power study, vol control |
| H4: real signal exists but below MDE at h=1/5 (sub-0.25% R² diffusion) | ~0.07 | cannot be excluded by construction; would be economically negligible anyway |
| H1/H2/H3: bug, data corruption, or AE information loss | ≤ 0.03 combined | injected-signal recovery, model-free scan bypassing AE, PCA parity, sanity anchors |

At h=21/63 the question is simply **untested** (power), and no probability statement about
economics should be made from those rows.

## 10. What was fixed in the repo during this investigation

1. **Report-file collision**: `make forecast-beta` silently overwrote
   `reports/forecast_pbsv_report.md` with beta-VAE numbers (the vanilla report was lost;
   discovered because the committed report's figures matched `results/forecast_pbsv_beta/`).
   Reports are now model-specific (`forecast_pbsv_report.md` / `forecast_pbsv_beta_report.md`),
   mirroring the sector-report convention.
2. **Timing caveat** added to the generated forecast report's substitution section so
   `sub_cw_p_all` can never again be read as macro transmission.
3. Pre-existing `ruff` E501 failure in `macro_mapping.py` fixed (lint gate was red).
4. Both pipelines regenerated so committed artifacts and reports correspond.

## 11. Recommendations (post-diagnosis)

1. **Reframe the project's claim** — it is a *contemporaneous pricing* study with a rigorous
   no-lagged-transmission result, not a forecasting study that failed. The mapping stage
   (which macro blocs price which factor directions, sector heterogeneity) is the deliverable.
2. **If forecasting must be pursued**, change the object, not the tuning: (a) volatility /
   covariance forecasting from the factor state (works, §7); (b) monthly returns over
   multi-decade panels (1960s– via index providers) where hedging-pressure/basis/momentum
   predictors have literature support; (c) intraday event studies of macro announcements,
   where transmission is measurable in minutes, not days.
3. **Respect the clock in any cross-asset daily design**: either sample all series at a
   common timestamp (e.g. synchronized 16:00 ET snapshots) or embargo predictors by one day
   and accept the loss; never mix LME noons with FX 17:00s in a "lag-1" regression.
4. **Do not spend further compute on h=21/63 with this sample** — the answer is
   power-bounded before the first regression runs.

## Appendix: experiment index

| Script | Question | Key artifact |
|---|---|---|
| `da01_data_forensics.py` | data integrity, sanity anchors | `da01_commodity_stats.csv`, `da01_macro_stats.csv` |
| `da02_lag_scan_calibrated.py` | any lagged macro signal? date shifts? | `da02_scan_calibration.csv`, `da02_dateshift_audit.csv` |
| `da03_timing_overlap.py` | embargo, settle fingerprint, before-closers | `da03_embargo.csv`, `da03_settle_fingerprint.csv`, `da03_group_counts.csv` |
| `da04_injected_signal_power.py` | engine validity + MDE | `da04_power_summary.csv` |
| `da05_substitution_anomaly.py` | substitution CW anomaly mechanism | `da05_substitution.csv`, `da05_substitution_fingerprint.csv` |
| `da06_vol_positive_control.py` | second-moment predictability | `da06_vol_control.csv` |
| `da07_ceiling_decomposition.py` | forecastability ceiling by link | `da07_ceiling.csv`, `da07_state_persistence.csv` |
| `da08_state_comparison.py` | AE vs PCA vs sector means | `da08_state_comparison.csv` |
| `da09_figures.py` | report figures | `figures/deep_analysis/*.png` |

All scripts are deterministic (fixed seeds), run offline against the committed data, and use
the project's own `eqcp` package for every engine call, so they exercise the code paths under
diagnosis rather than reimplementations.
