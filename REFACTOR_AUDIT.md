# Refactor Audit — `explainable_commodity_prices`

Audit performed before Phase 1 deletions. **Disposition matches Appendix A** with no contradictions.

## Supported entry points

| Entry point | Script / module | Reachable modules |
|---|---|---|
| `run_macro_mapping` | `scripts/run_macro_mapping.py` | `macro_mapping`, `autoencoder`, `data` |
| `run_synthetic_recovery` | `scripts/run_synthetic_recovery.py` | `synthetic_recovery`, `autoencoder`, `macro_mapping` |
| `process_macro` | `src/macro_processing/process_macro.py` | standalone (openpyxl, pandas) |
| Test suite | `tests/` | broken imports to deleted era |

## Module inventory

| Path | Imports | Imported by | Reachable? | Disposition |
|---|---|---|---|---|
| `src/data.py` | numpy, pandas | macro_mapping, factor_extraction, rolling_forecast, forecast_shapley, extract_factors, tests | yes | **Keep → eqcp/io/commodities.py** (fix path bug) |
| `src/autoencoder.py` | torch | synthetic_recovery, factor_extraction, rolling_forecast, run_macro_mapping | yes | **Keep → eqcp/factors/autoencoder.py** |
| `src/macro_mapping.py` | data, factor_alignment, factor_extraction | run_macro_mapping, synthetic_recovery, tests | yes | **Split → eqcp/cca/*, eqcp/spanning/*** |
| `src/synthetic_recovery.py` | autoencoder, macro_mapping | run_synthetic_recovery | yes | **Keep → eqcp/synthetic/recovery.py** |
| `src/macro_processing/process_macro.py` | openpyxl, pandas | CLI only | yes | **Keep → eqcp/macro_processing/** |
| `src/rolling_forecast.py` | autoencoder, data, forecast_models | run_forecasts | no (orphan) | **Delete** |
| `src/forecast_models.py` | numpy | rolling_forecast | no | **Delete** |
| `src/forecast_shapley.py` | data, factor_alignment, factor_extraction | run_factor_shapley, macro_mapping.incremental_content | no | **Delete** |
| `src/factor_alignment.py` | factor_extraction | run_alignment, run_factor_shapley, macro_mapping | no | **Delete** |
| `src/factor_extraction.py` | autoencoder, data | extract_factors, alignment era | no | **Delete** |
| `src/macro_data.py` | numpy, pandas | none (stale docstring ref only) | no | **Delete** |
| `src/autoencoders/` | broken (`vae.py` missing) | test_synthetic only | no | **Delete** |
| `src/evaluation/` | empty | test_synthetic only | no | **Delete** |
| `scripts/run_macro_mapping.py` | macro_mapping | CLI | yes | **Thin CLI → eqcp** |
| `scripts/run_synthetic_recovery.py` | synthetic_recovery | CLI | yes | **Thin CLI** |
| `scripts/extract_factors.py` | factor_extraction | none useful | no | **Delete** |
| `scripts/run_alignment.py` | factor_alignment | none | no | **Delete** |
| `scripts/run_factor_shapley.py` | forecast_shapley | none | no | **Delete** |
| `scripts/run_forecasts.py` | rolling_forecast | none | no | **Delete** |
| `tests/test_mapping_pipeline.py` | data, factor_extraction, macro_mapping | pytest | partial (broken tests) | **Rewrite** |
| `tests/test_synthetic.py` | autoencoders, evaluation | pytest | broken | **Delete & rewrite** |
| `tests/conftest.py` | sys.path hack | pytest | yes | **Replace with editable install** |

## macro_mapping.py dead functions (forecasting era)

Delete: `incremental_content`, `IncrementalContentResult`, `_macro_on_panel`, `decompose_factors`, `FactorDecomposition`.

Keep: `spanning_regression`, `canonical_correlations`, `kernel_canonical_correlations`, `bai_ng_spanning_summary`, `nonlinear_macro_mapping`, `purged_time_series_folds`, `_shap_importance`, and all CCA_methods S6 functions.

## Confirmed bugs

1. **`load_return_panel()`** defaults to `data/prices.csv` (missing); real file is `data/commodities/prices.csv`.
2. **Test collection** fails on `test_synthetic.py` (missing `src.autoencoders`, `src.evaluation`, `src.jacobian`).
3. **Aligned T acceptance**: inner-join of factors (2870 days) with `macro_stationary` (3074 days) yields **T=2691** because ~179 factor dates lack complete macro after transforms (gpr gaps, macro start 2013-11-01 vs factors 2013-10-21). Preferred path gives same T. Threshold should be **≥2690**, documented.
4. **ReLU one-sided latents**: encoder activation not yet a first-class config knob with `tanh` option.

## Cruft

- `results/final/` duplicate of `results/` + `reports/`
- `__pycache__`, `.pytest_cache` (gitignored)
- `prompt.md` / `macro_mapping_prompt.md` absent from repo (already gitignored)

## Appendix A verification

All dispositions in Appendix A confirmed. No discrepancies flagged.
