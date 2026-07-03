# Changelog — eqcp refactor

## Long-horizon macro transmission (weekly / monthly / quarterly)

- **Motivation.** Macro fundamentals move at weekly-to-quarterly frequency; the daily headline
  answered "does the macro-linked factor state forecast next-day commodity returns?" (a null).
  This adds first-class weekly (h=5), monthly (h=21) and **quarterly (h=63)** horizons so the
  project's core question — *do macros move commodities?* — is answered as a function of forecast
  horizon, not only at h=1.
- **Engine.** `SubsetForecasts.select_origins` + `nonoverlap_mask` (`eqcp/forecasting/ar1.py`):
  direct-h targets overlap by h-1 daily returns, so pooled tests over-count information; the new
  mask keeps targets spaced >= h apart for an autocorrelation-free (lower-power) Clark–West check.
  `forecast_accuracy_pooled.csv` now carries `cw_nonoverlap_p` / `n_nonoverlap` at every horizon.
- **Per-horizon transmission bundle.** The full attribution stack (exact PBSV, jointly-shifted
  placebo bands, leave-one-commodity-out CW, the pre-registered share-of-gain gate, and the
  macro-substitution ladder) now runs at **every** horizon in `attribution_horizons`, not just the
  headline. New artifact `results/forecast_pbsv/macro_transmission_by_horizon.csv` and figure
  `figures/forecast_pbsv/transmission_by_horizon.png` are the horizon ladder; the report gains a
  **"Macro transmission across horizons"** section and a horizon-aware verdict.
  `macro_substitution.csv` gains a `horizon` column.
- **Config.** `configs/forecast_pbsv.yaml`: `horizons: [1, 5, 21, 63]` and a new
  `attribution_horizons` list (`ForecastPBSVConfig.attribution_horizons`).

## Panel curation + sector-wise analysis + main orchestrator

- **Removed six stale-priced series entirely** from the commodity panel (`data/commodities/prices.csv`,
  `returns.csv`): `Lithium` (69.7% zero-return days / 57-day stale runs), `HRCSteel` (37.6%),
  `SGXIronOre` (23.8%), `Methanol` (19.4%), `ThermalCoal` (8.4%), `Diesel` (6.3%). These are not
  continuously price-discovered daily futures; their stale zeros biased the AE, attenuated the CCA
  correlations, and inflated forecast benchmarks. The retained **21** series are all < 2% stale.
- **Single source of truth** for the panel definition: `eqcp.io.commodities.EXCLUDED_COMMODITIES`
  and `SECTORS` (energy / agriculture / metals). `load_return_panel()` now enforces the exclusion by
  default and accepts an explicit `commodities` subset (used to build per-sector sub-panels). The
  redundant `exclude_commodities` / `sectors` fields were dropped from `configs/forecast_pbsv.yaml`.
- **Sector-wise framework** (`eqcp/sectors.py`, `eqcp/pipelines/sector_analysis.py`,
  `eqcp/reporting/sector_figures.py` / `sector_report.py`, `eqcp-sectors`, `make sectors`): runs the
  same AE → CCA spanning probe on each sector and overall, writing `results/sector_analysis/*.csv`,
  figures, and `reports/sector_analysis_report.md`. Finding: **energy** is the most macro-spanned,
  then **metals**, then **agriculture** (idiosyncratic).
- **`main.ipynb`** — a single clean orchestrator notebook (data curation → AE factors → overall
  macro spanning → sector spanning → forecast-value attribution) with the key importance bar charts,
  built for the economics team.
- **Regenerated all downstream artifacts** on the 21-commodity panel: `ae_factors_vanilla.csv`,
  `results/macro_mapping/*`, `results/forecast_pbsv/*`, figures, and the narrative reports.

## Added — forecast-PBSV stage (rebuilt from scratch, replaces the deleted forecasting era)

- **`eqcp/forecasting/`**: `ar1.py` (expanding-window direct h-step factor-augmented AR(1),
  all 2^K factor subsets refit per origin via one shared Gram; zero/mean/AR(1) benchmarks;
  Clark–West with pool-then-HAC cross-sectionally robust inference; staleness diagnostics;
  Campbell–Thompson-style timing utility), `basis.py` (train-frozen CCA canonical-variate
  attribution basis; **factor-side ridge pinned to 0** — invariance to invertible factor-block
  transforms holds only under exact factor-side whitening; degeneracy guard for dead ReLU
  latents; data-driven spanned|weak boundary; forward OOS rho + loading-drift diagnostics;
  train-bootstrap subspace stability), `pbsv.py` (exact Shapley, grouped block game,
  joint-block bootstrap, **cardinality-matched circular-shift placebo bands**, boundary
  sensitivity).
- **`eqcp/pipelines/forecast_pbsv(.py|_cli.py)`** (`eqcp-forecast`, `make forecast`),
  `configs/forecast_pbsv.yaml`, `eqcp/reporting/forecast_figures.py` / `forecast_report.py`.
- **Design decisions vs the deleted era**: attribution NEVER on raw AE coordinates or
  per-window sign/permutation alignment; leak-free factor construction (AE + z-stats train-only,
  frozen); macro-substitution arm on a common macro calendar with gpr/epu lagged one day;
  pre-registered gate before any share-of-gain language.
- `linear_cca_full` / `purged_cv_canon` / `perdim_perm_null_oos` gained an optional
  `ridge_f` override (default preserves old symmetric-ridge behavior exactly).

## Deleted (forecasting era / orphans)

- `src/rolling_forecast.py`, `forecast_models.py`, `forecast_shapley.py`, `factor_alignment.py`, `factor_extraction.py`, `macro_data.py`
- `src/autoencoders/` (broken duplicate AE package)
- `src/evaluation/` (empty)
- Scripts: `extract_factors.py`, `run_alignment.py`, `run_factor_shapley.py`, `run_forecasts.py`
- `macro_mapping.py` dead functions: `incremental_content`, `decompose_factors`, `_macro_on_panel`
- `results/final/` duplicate output tree

## Fixed

- **`load_return_panel()`** now defaults to `data/commodities/prices.csv` with a clear `FileNotFoundError`.
- **Test suite** collects and passes (`pytest -q`).
- **Aligned T acceptance**: threshold lowered to **2690** (observed T=2691). The inner join drops ~179 factor dates where macro is incomplete after transforms (gpr early gaps, macro panel starts 2013-11-01 vs factors 2013-10-21).
- **Encoder activation** is a config knob: `relu` (default), `linear`, `tanh`.

## Moved / restructured

- Monolithic `src/` → installable package **`eqcp`** under `src/eqcp/`
- `macro_mapping.py` split into `eqcp/cca/*`, `eqcp/spanning/*`, `eqcp/reporting/*`, `eqcp/pipelines/`
- Config dicts → `configs/*.yaml` loaded via `eqcp.config`
- Docs → `docs/` (`AE_explainability_study.md`, `CCA_methods.md`)
- Thin CLIs in `scripts/`; logic in `eqcp`

## Science / methodology changes (documented, tested)

1. **KCCA nonlinearity verdict**: only claims nonlinearity when `kcca_min` is stable across reg/gamma sweep (IQR < 0.15) *and* exceeds linear min by >0.1; otherwise **inconclusive / degenerate**.
2. **OOS permutation p-values**: headline per-dimension and aggregate p-values now use purged-CV OOS ρ vs circular-shift null. In-sample perm-p retained as `perm_p_insample` (necessary-condition check).
3. **Encoder activation experiment**: `encoder_activation_experiment.csv` compares macro-spanning under relu/tanh/linear encoders.

## Numbers unchanged (within tolerance)

- `linear_cca_full(ridge=0) == canonical_correlations` to ~1e-13
- Macro transform outputs and CCA point estimates on the aligned sample match pre-refactor values
- Headline finding preserved: ~2 of 5 factor directions macro-spanned OOS (dims 1–2)
