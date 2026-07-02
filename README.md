# Explainable Commodity Prices

Explaining daily commodity-futures returns with macro variables, in four stages:
**(1)** compress **21** clean commodity return series into 5 latent factors with an autoencoder,
**(2)** build a clean 37-variable daily macro panel,
**(3)** test — with canonical correlation analysis and proper out-of-sample/permutation
inference — whether and how those commodity factors are *spanned* by macro (overall **and
sector-by-sector**: energy / agriculture / metals), and
**(4)** attribute out-of-sample forecast accuracy of a factor-augmented AR(1) to the
factor directions via forecast-based Shapley values (PBSV), composed with the macro map.

Start with **`main.ipynb`** — the one-notebook orchestrator that runs the full engine and
surfaces the most valuable insights (built for the economics team).

## Panel curation

Six nominal series were **removed entirely** because they are not continuously price-discovered
daily futures — they carry stale settlement marks on a large fraction of days (zero-return days:
`Lithium` 69.7%, `HRCSteel` 37.6%, `SGXIronOre` 23.8%, `Methanol` 19.4%, `ThermalCoal` 8.4%,
`Diesel` 6.3%), which poison the AE, CCA, and forecast benchmarks. The retained 21 series are all
< 2% stale. The panel definition (exclusions + the 3-sector partition) lives in one place:
`src/eqcp/io/commodities.py` (`EXCLUDED_COMMODITIES`, `SECTORS`).

## Headline result

On the aligned daily sample, ~2 of the 5 commodity factor directions are linearly spanned
by macro out-of-sample: **dim1 ≈ dollar / inflation / energy-equity**, **dim2 ≈ dollar /
commodity-FX**. Dimensions 3–5 do not survive out-of-sample testing against the permutation
null. In-sample canonical correlations are reported only as a (inflated) descriptive ceiling.

## Why CCA, and why out-of-sample

Autoencoder latents are identified only up to rotation/sign/permutation, so per-factor R²
understates recovery; CCA scores the **factor space**, which is the invariant object. A 37-dim
collinear macro panel can align with any 5-dim target by chance in finite sample, so the
verdict rests on **purged-CV out-of-sample canonical correlations vs a circular-shift
permutation null**, with block-bootstrap CIs — not raw in-sample numbers. See `docs/CCA_methods.md`.

## Install

```bash
pip install -e ".[dev]"
```

## Pipelines

```bash
make macro-panel   # build raw/aligned/stationary macro panel from the workbook
make mapping       # AE factors <-> macro CCA: OOS rho, null, CIs, bloc map, per-regime, figures, report
make sectors       # sector-wise AE <-> macro CCA spanning (energy / agriculture / metals) + report
make synthetic     # synthetic ground-truth validation of the CCA probe
make forecast      # factor-augmented AR(1) forecasts + PBSV attribution + macro substitution
make all
```

Each is offline, deterministic (`--seed`), and reproduces identical outputs across runs.

## Repository layout

- `src/eqcp/io` — data loaders (commodities, macro); `commodities.py` holds the canonical panel/sector definition
- `src/eqcp/factors` — the vanilla autoencoder (encoder activation configurable: `relu` | `linear` | `tanh`)
- `src/eqcp/sectors.py` — sector-wise factor extraction + CCA spanning (energy / agriculture / metals)
- `src/eqcp/cca` — linear & kernel CCA, OOS/permutation/bootstrap inference, bloc reduction, lead/lag
- `src/eqcp/forecasting` — factor-augmented AR(1) engine, frozen canonical-variate basis, PBSV
- `src/eqcp/spanning` — spanning regressions + nonlinear (GBT/SHAP) mapping
- `src/eqcp/synthetic` — known-DGP recovery study
- `src/eqcp/macro_processing` — workbook → clean panel
- `src/eqcp/reporting` — figures and markdown reports
- `configs/` — typed YAML for each stage
- `tests/` — unit + integration, deterministic
- `data/`, `results/`, `figures/`, `reports/` — inputs and outputs
- `docs/` — methodology notes (`AE_explainability_study.md`, `CCA_methods.md`)

## Interpreting the outputs

- `results/macro_mapping/canonical_correlations.csv` — per-dimension in-sample ρ, **OOS ρ**, null band,
  **OOS permutation p** (headline), in-sample perm-p (necessary-condition), bootstrap CI, and KCCA columns.
  **Read the OOS column and the OOS p-value, not in-sample ρ.**
- `results/macro_mapping/canonical_loadings_macro.csv` — structure correlations: which macro variables
  load on each canonical direction (the interpretable map).
- `results/macro_mapping/bloc_cca_summary.csv` — the clean 9-bloc version of the map.
- `results/macro_mapping/per_regime_summary.csv` — stability across macro regimes R1..R9.
- `results/macro_mapping/encoder_activation_experiment.csv` — ReLU vs tanh vs linear encoder comparison.
- `reports/macro_mapping_report.md` — the narrative verdict.
- `results/sector_analysis/sector_cca_summary.csv` — per-sector OOS ρ, perm-p, spanned count, top drivers.
- `reports/sector_analysis_report.md` — the sector-by-sector narrative.
- `results/forecast_pbsv/` — forecast accuracy vs zero/mean/AR(1) benchmarks, PBSV Shapley
  tables with bootstrap CIs and cardinality-matched placebo bands, grouped (spanned vs
  weakly-macro-correlated block) attribution, macro-substitution ladder, seed/basis stability.
  **Attribution is in the train-frozen CCA canonical-variate basis, never raw AE coordinates**
  (AE latents are identified only up to invertible affine maps; raw-coordinate attribution is
  basis-dependent). `reports/forecast_pbsv_report.md` is the narrative verdict with a
  pre-registered gate on share-of-gain language.

## Caveats

Contemporaneous alignment; stationarity assumed; macro panel contains **no commodity prices**
(no mechanical leakage). Data seams documented in the QA report (`cny_10y` 2016 source seam,
`gpr` early gaps, `bdti` pulled as `BIDY`, non-uniform FX direction). KCCA is reported with a
stability sweep and is treated as inconclusive when degenerate.

## Reproducibility

Python ≥ 3.10. All randomness seeded from `--seed`. `make test` runs the suite; `make lint`
and `make typecheck` enforce style and types.
