# Forecast-based Shapley values (PBSV) over AE factor directions — methods

This doc records the design of the `eqcp-forecast` stage (`src/eqcp/forecasting/`,
`eqcp/pipelines/forecast_pbsv.py`): a factor-model forecasting pipeline (AR(1) class),
a forecast-accuracy attribution to latent-factor directions (PBSV), and its composition
with the existing factor↔macro CCA map. Read alongside `CCA_methods.md` and
`AE_explainability_study.md`, whose identification doctrine this stage inherits.

---

## 1. Forecasting model

Direct h-step, per commodity i, expanding window, refit every origin:

```
y(h)_{i,t+h} = alpha_i + beta_i * y(h)_{i,t} + gamma_i' s_t + eps_{i,t+h}
```

- `y(h)` = trailing h-day cumulative log return (h ∈ {1, 5, 21, 63} = daily / weekly /
  monthly / quarterly; h=1 headline). Because macro fundamentals move at weekly-to-quarterly
  frequency, the full transmission analysis (§2, §5) is run at **every** horizon in
  `attribution_horizons`, not only h=1 — the horizon ladder is the object that answers "do
  macros move commodities?" as a function of horizon.
- `s_t` = the 5-dim latent state at the close of day t (basis: §3).
- **Exogenous information set = factor state only.** Macro is deliberately excluded from
  the forecasting information set so that macro content can only enter through the
  interpretation/substitution layers — otherwise the composition in §5 would be circular.
- Benchmarks: zero forecast (driftless RW), expanding mean, and plain AR(1) (`gamma=0`);
  the AR(1) is the PBSV baseline (nested, so Clark–West applies).
- Leakage contract: a forecast issued at origin t uses design rows `tau <= t` and training
  pairs whose target is fully realized (`tau + h <= t`). The AE and its z-score statistics
  are fit on the train segment only and frozen (leak-free factor construction); the CCA
  basis and all spanning labels are train-only.
- Evaluation: pooled MSE on per-commodity variance-standardized losses (raw pooling would
  make the game a NatGas/energy game), on the non-stale panel (series with >15% exact-zero
  daily returns — backfilled/illiquid assessments — are excluded from targets and reported
  separately).   Pooled Clark–West is computed on the time series of cross-sectional mean
  adjusted differentials (robust to arbitrary cross-sectional error correlation), Bartlett
  bandwidth ≥ 2h. Per-commodity CW carries BH-FDR flags. For h>1 the direct-h targets of
  adjacent origins overlap by h-1 daily returns, so the overlapping pooled test over-counts
  information; alongside it we report a **non-overlapping** Clark–West (`cw_nonoverlap_p`) run on
  a greedy subset of origins whose targets are spaced ≥ h apart (autocorrelation-free but
  lower-power: effective n ≈ T_oos/h). Agreement of the two is the honest long-horizon check.

## 1a. Horizon ladder (the transmission question)

For each horizon the pipeline emits `macro_transmission_by_horizon.csv`: OOS R² of the full
factor state vs AR(1), overlapping and non-overlapping CW p, the placebo-calibrated CW p, the
leave-one-commodity-out max p, v(full) with its bootstrap CI, the grouped spanned/weak PBSV with
its placebo band, the macro-substitution retained share, and the pre-registered `gate_passed`.
This ladder is the deliverable: it shows at which horizon (if any) the macro-linked commodity
factor directions translate contemporaneous spanning into out-of-sample predictability, and how
much of any gain is macro-transmissible. `retained_share` is interpretable only where the gate
passes; elsewhere it is a ratio of statistical zeros and is reported for completeness.

## 2. PBSV — the attribution game

Players = 5 directions of the state (basis in §3). For each coalition S:

```
v(S) = MSE_oos(AR1) − MSE_oos(AR1 + s_S),   v(∅) = 0
```

computed by refitting the expanding OLS for all 2^5 subsets (exact Shapley, shared
per-commodity Gram, subsets solved by masked batched solves). Efficiency is exact:
`Σ_k phi_k = v(full)`.

Three calibration/inference layers, each with a distinct job:

1. **Clark–West contrasts** — the only significance claims.
2. **Stationary-bootstrap CIs** over OOS origins (joint across subsets, block ≥ max(21, 2h))
   — descriptive dispersion, explicitly conditional on the frozen basis and the realized
   expanding-parameter path (nested-model degeneracy means these CIs cannot support
   `phi ≠ 0` claims on their own).
3. **Cardinality-matched placebo bands** — the identical 2^5 machinery re-run on jointly
   circular-shifted state (common tau per draw). Every regressor pays an O(sigma²·ln(T/T_tr)/T_oos)
   OOS estimation cost even under zero signal; Shapley spreads that mechanical cost across
   players, so small negative phi is *expected under the null*. phi inside the placebo band
   is not interpreted. (The circular shift destroys vol-synchronization as well as alignment,
   so the band is a calibration reference, not an exact test — significance stays with CW.)

## 3. The attribution basis (identification)

Two distinct ambiguities affect "attribute to factor k":

- the ReLU AE's own gauge is permutation × positive diagonal scaling; but different seeds
  reach genuinely different latent spaces (optimizer multiplicity), so raw coordinates have
  no stable identity across retrains;
- the downstream pipeline (OLS on `[1, y_lag, s_t]`; CCA) is invariant to **any invertible
  affine map** of the state block — the full-model forecast is basis-free, so any
  per-coordinate split finer than the span structure is an arbitrary-basis statement.

PBSV on raw AE coordinates is therefore an attribution to arbitrary labels. The pipeline
attributes in the **CCA canonical-variate basis fit on train and frozen**: `v_t = A'(f_t − mu_F)`.
Canonical variates are invariant to invertible linear maps of the factor block (up to sign,
and up to rotation inside blocks of tied canonical correlations) **provided the factor side
is exactly whitened** — with a factor-side ridge the whitener no longer transforms
covariantly and the invariance fails (verified numerically: O(1) variate changes at
ridge=0.01 under a generic invertible transform). Hence `ridge_f = 0` is pinned
structurally in `eqcp/forecasting/basis.py` (regularization is confined to the 37-dim
macro side, which does not disturb factor-side invariance), and a degeneracy guard refuses
to build a basis from dead/duplicated latents rather than regularizing them away.

Because the trailing canonical correlations are near-tied, individual trailing directions
rotate freely under resampling — they are **individually non-identified**; only their block
is meaningful. The robust attribution level is the grouped game over
`{spanned block, weakly-macro-correlated block}` (v of a union of whole groups depends only
on its span, so grouped values are invariant to within-block rotation). The boundary is set
by the largest adjacent gap in the train purged-CV rho spectrum (threshold fallback), with
a full boundary-sensitivity table. The trailing block is called *weakly macro-correlated*,
not "orphaned": all dims can reject the permutation null while having small rho.

Diagnostics that keep this honest: forward (strictly OOS) `corr(v_k, u_k)` under the frozen
basis; OOS loading-drift cosine (train-era macro names are used only above 0.8); train-window
bootstrap of the basis reporting principal-angle dispersion of the spanned subspace and
boundary reproducibility (basis-estimation noise is quantified separately, NOT folded into
phi CIs); an AE-seed arm reporting subspace principal angles and grouped-phi dispersion
(labelled illustration, n=3); and a raw-coordinate phi-by-seed table shown only to
demonstrate label arbitrariness. The invariance theorem itself is exercised by unit tests
(random invertible state transforms leave full-model forecasts and CV variates unchanged,
including with macro-side ridge > 0) and by a pipeline acceptance check (raw-basis vs
CV-basis full-model MSE equality).

## 4. Vol channel

ReLU latents are partly volatility proxies: for a symmetric portfolio return `w'x_t`,
`E[ReLU(w'x_t + b)]` increases with its vol, and GARCH persistence makes that component
autocorrelated without implying mean predictability. A controls arm reruns the whole game
with EWMA vol and 12-month momentum in BOTH the benchmark and the full model; spanned-block
phi that survives only without vol controls is read as vol timing wearing a macro label.

## 5. Composition with the macro map — what may be claimed

Stacking Shapley across stages (multiplying phi by squared CCA loadings to get per-macro-
variable attributions) is invalid: there is no single game with macro players, Shapley has
no chain rule, and the CCA map is correlational. The pipeline supports exactly two claim
forms:

- **(a) Juxtaposition**: "X of the OOS gain is attributable to the subspace that is
  macro-spanned (forward OOS rho and its rho² shared-variance share printed alongside);
  Y to the weakly-macro-correlated block." CCA loadings *name* directions in a separate
  identity table; they are never attached to phi as weights. Even "spanned" directions are
  majority non-macro variance (rho² < 50%), and the report says so.
- **(b) Substitution (the only arm licensed to say "explained by macro")**: replace
  `v_{k,t}` by its train-fitted macro proxy `rho_k u_{k,t}` for a set of directions and
  rerun the identical pipeline. All arms — including "none" — share the macro-available
  calendar, and gpr/epu are lagged one day (publication realism). Since the proxy discards
  the (1 − rho²) non-macro variance share (errors-in-variables), retained shares are
  **lower bounds** on macro-transmissibility.

**Gate (pre-registered)**: share-of-gain and retained-share language is licensed only if
the pooled Clark–West rejects at 5%, the v(full) bootstrap CI excludes 0, and the full
model beats the zero forecast. Otherwise the result is reported as a well-identified null
with absolute phi against placebo bands.

**Pre-registered outcome classes** (from the repo's own lag-0 lead/lag peak: the factor–macro
link is same-day news impoundment): (i) gate fails — no exploitable daily mean signal,
macro-composition moot at this horizon; (ii) weak-block phi > placebo with spanned ≈ 0 —
EMH-coherent: contemporaneously priced macro content is precisely the least forecastable
part of the state; (iii) spanned-block phi > placebo, surviving vol controls, with positive
substitution retention — the surprising outcome that would license "macro content forecasts
commodities."

## 6. Known limitations (disclosed)

- Single 60/40 split (split lands 2021; COVID in train); per-year phi tables address
  regime dependence descriptively.
- The circular-shift placebo preserves autocorrelation but not vol-synchronization with
  targets; CW is the significance instrument.
- Basis-estimation noise is quantified (principal-angle bootstrap) but not propagated into
  phi CIs (conditional-on-deployed-model convention, stated explicitly).
- K=5, ReLU, and the two-block grouping schema were motivated by the full-sample mapping
  study; labels are re-derived train-only but the schema choice post-dates that evidence.
- Carry/hedging-pressure predictors are out of scope: the null is "no predictability from
  prices + latent factor state," not "no predictability."
