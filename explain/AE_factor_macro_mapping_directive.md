# Experiment Directive: Mapping Autoencoder Latent Factors to Macroeconomic Factors

**Goal.** Assign each AE latent factor $f_{k,t}$ an economic identity in terms of observable macro factors, and determine whether the AE captures anything beyond what observable macro already spans. Run the experiments below in the stated priority order; keep whichever yields the cleanest, most defensible mapping.

**Pipeline context.** $x_{i,t}$ are log-returns of the 22 commodities; the AE compresses them to $K$ latent factors $f_{k,t}$; a factor-augmented AR(1)/GARCH forecaster consumes lagged $f$; forecast-based Shapley $\phi_k$ ranks factor importance for out-of-sample (OOS) accuracy. The mapping experiments take the **aligned** factor series (E0 output) as input and relate them to an observable macro panel $M_t = (m_{1,t}, \dots, m_{J,t})$.

---

## E0 (PREREQUISITE, not optional): Cross-window factor alignment and stability audit

**Why first.** You re-estimate the AE on each rolling window, exactly as the paper does. AE latent coordinates are identified only up to permutation, sign, and nonlinear reparametrization. So "Factor 3" in window $w$ need not be "Factor 3" in window $w+1$. Any macro regression that pools across windows without alignment is averaging incoherent objects, and every downstream result is contaminated. This step gates the entire directive.

**Formulation.**
1. Pick a reference window $w_0$. For every other window $w$, build the $K \times K$ matrix of absolute correlations between its factor series and the reference's over their overlap. Match factors by maximizing total $|\text{corr}|$ (Hungarian / linear-assignment algorithm). Fix each matched factor's sign by the sign of its matched correlation.
2. Alternative / corroborating anchor: match by cosine similarity of decoder weight columns (or Jacobian columns, see E3) rather than by output correlation. Agreement between the two anchors is itself a stability signal.
3. Optional drift reduction: warm-start each window's AE from the previous window's weights.

**What to look for.**
- Distribution of adjacent-window matched $|\text{corr}|$. Median above ~0.8 means stable identity; proceed at full strength.
- Sign-flip rate and permutation frequency per factor.
- **Gate:** if median matched $|\text{corr}|$ is low (below ~0.5), the factors do not have a stable identity. Then either (a) restrict all mapping to within-window, or (b) report the instability itself as a finding (the AE coordinate system is not economically persistent), which is a legitimate and interesting result.

---

## E1 (PRIORITY 1): Linear spanning regression + incremental-content test

This is the inferential spine. It answers both core questions (what is $f_k$; does the AE add anything beyond macro) with standard, defensible econometrics, and it is fast.

**Formulation.**
1. Construct $M_t$ (see Data Appendix). Run at the native daily frequency of $f$; repeat monthly as robustness for slow macro.
2. For each $k$: OLS $f_{k,t} = a_k + \theta_k^\top M_t + u_{k,t}$ with Newey-West (HAC) standard errors. Report standardized $\hat\theta_k$, $R^2_k$, and the dominant regressors.
3. **Bai and Ng (2006) spanning test** ("Evaluating latent and observed factors in macroeconomics and finance"): formally test whether the latent factor space is spanned by $M$, using their canonical-correlation-based statistic, rather than eyeballing $R^2$.
4. **Incremental-content test.** Decompose $f_{k,t} = \hat f^{M}_{k,t} + \hat u_{k,t}$ with $\hat f^{M}_{k,t} = \hat\theta_k^\top M_t$. Re-run the factor-augmented forecaster three ways: using $f_k$, using only $\hat f^{M}_k$ (macro-spanned part), using only $\hat u_k$ (orthogonal residual). Compare OOS RMSE and re-compute the forecast-Shapley $\phi_k$ for each.

**What to look for.**
- **Clean map:** high $R^2_k$, one or two macro dominate, stable signs across windows and frequencies.
- **Spanned factor:** $\hat f^{M}_k$ reproduces most of the forecast value; the AE adds little beyond macro for that factor (so you can replace it with named macro).
- **Genuinely latent / nonlinear factor:** $\hat u_k$ retains forecast-Shapley importance; the AE is doing real work that macro misses. This is the most interesting outcome and motivates E2.

---

## E2 (PRIORITY 2): Nonlinear macro mapping + macro-Shapley attribution

The encoder is nonlinear, so a linear projection can understate the macro link. This recovers nonlinear identity and quantifies how much of it is nonlinear.

**Formulation.**
1. Fit $f_{k,t} = h_k(M_t) + \varepsilon_{k,t}$ with $h_k$ a gradient-boosted tree ensemble (or kernel ridge). Use purged/embargoed time-series cross-validation to avoid leakage.
2. Compute SHAP values of $M$ under $h_k$; rank macro drivers by mean $|\text{SHAP}|$. (Note: this is a **second, distinct** Shapley layer, on the mapping, not on the forecast. Keep it conceptually separate from $\phi_k$.)
3. Report the nonlinearity premium $R^2_{\text{nonlin}} - R^2_{\text{lin}}$ (E1 vs E2).

**What to look for.**
- SHAP concentrated on a few macro = interpretable nonlinear identity.
- Large nonlinearity premium = the factor is a nonlinear macro composite (e.g. a USD-by-volatility interaction). Small premium = E1 already suffices and wins on parsimony.
- Diffuse SHAP with low $R^2$ = the factor is not macro-explainable (genuinely idiosyncratic/latent).

---

## E3 (PRIORITY 3): Decoder-Jacobian commodity fingerprint to macro hypothesis

Corroboration layer and economic narrative. It is the rigorous version of the paper's sector association, and it triangulates E1/E2.

**Formulation.**
1. Compute the Jacobian $J_{ik} = \partial \hat x_i / \partial f_k$ at the sample mean (or averaged over OOS samples). The column $J_{\cdot k}$ is a loading vector over the 22 commodities.
2. Normalize, inspect top-loading commodities, group by sector (energy / metals / agriculture).
3. Map sector to a canonical macro hypothesis: energy to Kilian (2009) aggregate-demand and oil-specific shocks; base metals to global industrial production, USD, real rates; precious metals to real rates, USD, risk-off; agriculture to USD and supply/weather (typically a weak macro link).
4. **Cross-check:** does the dominant macro from E1/E2 match the sector the Jacobian points to?

**What to look for.**
- Sparse, sector-coherent loadings = clean factor; agreement with E1/E2 macro = strong, triangulated identity.
- Diffuse loadings across sectors = entangled factor; any macro mapping will be muddy, and you should say so.

---

## E4 (PRIORITY 4): Regime / event overlay + local projections

Cheap, visually persuasive robustness layer. Low standalone inferential rigor.

**Formulation.**
1. Mark regimes/events: NBER recessions, GFC 2008, 2013 taper, 2014-16 oil crash, COVID Feb-Apr 2020 (including the negative WTI close on 20 Apr 2020), 2022 Russia-Ukraine.
2. Conditional mean and volatility of $f_k$ by regime; local-projection impulse responses of $f_k$ to identified macro shocks (Kilian oil shocks, monetary-policy surprises).

**What to look for.**
- A factor that activates only in oil-supply events = oil-specific.
- A factor co-moving with risk-off across all crises = a global risk / USD factor.
- Consistency with the E1-E3 identity assignment.

---

## E5 (PRIORITY 5, CONDITIONAL, higher effort): Guided AE / conditional loadings

Pursue **only if** the post-hoc mappings (E1-E2) are weak. It changes the model and is expensive.

**Formulation.**
- **Option A (supervised penalty):** retrain the AE with loss = reconstruction $+\ \lambda \sum_k \big(1 - \text{corr}(f_k, \text{target macro}_k)^2\big)$ (or a partial-correlation penalty). Sweep $\lambda$ and plot the reconstruction-loss vs interpretability frontier.
- **Option B (conditional loadings, Gu-Kelly-Xiu 2021 / Kelly-Pruitt-Su IPCA):** make decoder loadings explicit functions of $M_t$, so factor identity is tied to observables by construction.

**What to look for.**
- Small reconstruction sacrifice for a large interpretability gain = worthwhile.
- Large sacrifice = the AE's nonlinear structure is genuinely non-macro; report that rather than forcing a mapping.

---

## How to pick the winner (cross-experiment rubric)

Score each factor's mapping on:

1. **Identity concentration:** does a small macro set explain $f_k$? (E1 $R^2$ / E2 SHAP concentration.)
2. **Triangulation:** do the E1/E2 dominant macro agree with the E3 Jacobian sector and the E4 regime behavior?
3. **Forecast relevance preserved:** does the macro-explained part $\hat f^M_k$ retain the forecast-Shapley importance? (E1 incremental test.)
4. **Stability:** holds across windows (E0) and across daily vs monthly frequency.

"Best result" = the approach maximizing (1) and (3), corroborated by (2), and stable under (4). Prior expectation: **E1 spine + E3 narrative** gives the cleanest, most publishable mapping; **E2** overtakes it only if the nonlinearity premium is large.

---

## Data Appendix

**Candidate macro panel** (all FRED-accessible unless noted):

- USD: broad dollar index (DTWEXBGS) or DXY, daily.
- Equity vol / risk: VIX, daily.
- Term structure: 10y-2y (T10Y2Y) or 10y-3m (T10Y3M), daily.
- Credit risk: BAA-AAA spread, or HY OAS (BAMLH0A0HYM2), daily.
- Funding stress: SOFR-OIS or legacy TED spread, daily.
- Real rate: 10y TIPS (DFII10), daily.
- Inflation expectations: 10y breakeven (T10YIE), daily.
- Real activity: industrial production (INDPRO, monthly), Kilian global real activity index (monthly), Baltic Dry Index (higher frequency proxy).
- Policy uncertainty: EPU index, daily.

**Circularity caution.** Do **not** use a broad commodity index (e.g. GSCI) as a regressor. The factors are built from commodities, so that link is mechanical and tells you nothing. Keep regressors to FX, rates, vol, credit, and real-activity series that are exogenous to the commodity panel.

**Frequency.** Run the primary mapping daily, using daily-available macro. Add a monthly robustness pass for INDPRO/Kilian by aggregating $f$ to monthly (mean or end-of-month).

**Stationarity.** Put everything in stationary form before regressing: spreads, vol, and breakevens are typically usable in levels (check), while DXY and IP enter as log-changes. Align all series to a common stationary convention.
