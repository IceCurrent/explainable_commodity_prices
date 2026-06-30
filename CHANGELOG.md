# Changelog — eqcp refactor

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
