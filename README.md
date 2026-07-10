# Explainable Commodity Prices — Rolling-Window Engine

Explaining daily commodity-futures returns with macro variables, **honestly** —
everything is re-fit on a rolling walk-forward window, nothing is trained on the
full sample.

The pipeline compresses **21** clean commodity return series into latent factors
with an autoencoder, builds a clean **37**-variable daily macro panel, and then, on
**every rolling window**, asks two questions jointly:

1. **Explainability under rolling** — retrain the AE and refit the canonical
   correlation (CCA) basis on each window; does the macro *explanation* of the
   factor space survive when the representation is allowed to drift regime to
   regime?
2. **Forecasting under rolling** — does a factor-augmented AR, with the state
   re-estimated every window, beat AR(1)/mean/zero out-of-sample at any horizon?

Start with **`main.ipynb`** — the one-notebook orchestrator and final deliverable.

## Why rolling, and why this is the honest design

An earlier version of this project trained one autoencoder and one CCA basis on
a long block of history (through ~2021) and tested afterwards. That leaks the
whole sample into the representation and reports a single, non-transportable
"explanation." **This engine deletes that approach entirely.** For each window it

1. z-scores returns on **train-window statistics only**, retrains the AE on the
   train block, and encodes train+test days through the just-trained encoder;
2. refits the linear CCA basis on the **train block only**, ordering canonical
   dimensions by train canonical correlation and sign-anchoring each on its
   dominant macro loading so ranks are comparable across windows;
3. projects onto the window's canonical-variate state and fits a direct h-step
   factor-augmented AR for every factor subset on the train block;
4. forecasts the held-out test block, then advances by one non-overlapping block.

The AE latent coordinates rotate window to window (gauge + optimizer multiplicity
+ genuine regime drift), so the *factor side* is not comparable across windows.
The **macro side is**: every window's canonical vectors live in the same fixed
37-series macro space. We therefore judge rank identity by the **cross-window
cosine** of each dim's 37-dim macro structure-correlation vector against a
label-shuffle null — if rank *k* keeps pointing at the same macro combination,
its macro identity is *regime-transportable* and rank-pooled attribution is
legitimate; if it shatters, that is the disclosed limitation of marrying rolling
windows to explainability.

## Panel curation

Six nominal series were **removed entirely** because they are not continuously
price-discovered daily futures — they carry stale settlement marks on a large
fraction of days (zero-return days: `Lithium` 69.7%, `HRCSteel` 37.6%,
`SGXIronOre` 23.8%, `Methanol` 19.4%, `ThermalCoal` 8.4%, `Diesel` 6.3%), which
poison the AE, the CCA, and the forecast benchmarks. The retained 21 series are
all < 2% stale. The panel definition (exclusions + the 3-sector partition) lives
in one place: `src/eqcp/io/commodities.py` (`EXCLUDED_COMMODITIES`, `SECTORS`).

## Headline results

- **The explanation survives rolling.** Re-fitting the AE and CCA basis on every
  252-day window, the leading canonical rank keeps the **same macro fingerprint**
  across all ~125 windows (cross-window median |cos| ≈ 0.82 vs a ≈ 0.13
  label-shuffle chance level) and still tracks macro out-of-sample. One dominant
  macro–commodity axis (≈ dollar / inflation / energy-equity) is
  **regime-transportable**; the remaining ranks rotate.
- **Sector heterogeneity is real, not an AE artifact.** Running the identical
  rolling engine per sector, **energy and metals keep a stable, out-of-sample
  macro axis** under full per-window re-fitting; **agriculture rotates** (the
  idiosyncratic sector). Reproduces under the β-VAE.
- **The forecast null survives rolling too.** Even with a fully regime-adaptive
  representation, the factor-augmented AR does **not** beat AR(1) at any horizon,
  overall or in any sector — consistent with near-efficient daily futures. What
  survives rolling is the *explanation*, not forecast value.

## Install

```bash
pip install -e ".[dev]"
```

## Pipelines

```bash
make macro-panel            # build raw/aligned/stationary macro panel from the workbook
make rolling                # overall 21-commodity rolling AE+CCA+forecast (vanilla AE)
make rolling-beta           # overall, beta-VAE robustness arm
make rolling-sectors        # per-sector rolling decomposition (energy/agriculture/metals) + overall
make rolling-sectors-beta   # per-sector, beta-VAE arm
make rolling-all            # all four (a few minutes)
```

Each is offline, deterministic (`--seed`), retrains the AE per window, and
reproduces identical outputs across runs.

## Repository layout

- `src/eqcp/io` — data loaders; `commodities.py` holds the canonical panel/sector definition
- `src/eqcp/factors` — the vanilla autoencoder and the β-VAE (encoder activation configurable)
- `src/eqcp/forecasting` — the factor-augmented AR engine (`ar1`), forecast-based Shapley
  values (`pbsv`), the frozen canonical-variate basis math (`basis`), and the **rolling
  walk-forward engine** (`rolling`) that re-fits everything per window
- `src/eqcp/cca` — linear & kernel CCA, OOS / permutation / bootstrap inference, bloc reduction
- `src/eqcp/pipelines` — `rolling_forecast` (overall panel) and `rolling_sectors` (per sector),
  sharing the reduction helpers in `_rolling_common`
- `src/eqcp/macro_processing` — workbook → clean stationary macro panel
- `src/eqcp/reporting` — rolling figures and markdown reports
- `configs/` — typed YAML (`factor_model`, `macro_processing`, `rolling_forecast`)
- `tests/` — unit + integration, deterministic
- `data/`, `results/`, `figures/`, `reports/` — inputs and rolling outputs
- `docs/CCA_methods.md` — CCA methodology note

## Interpreting the outputs

- `results/rolling_forecast[_beta]/rank_stability.csv` — per canonical rank: cross-window
  median |cosine| of the macro fingerprint, the label-shuffle null and one-sided `p_value`,
  the median frozen-in-window OOS canonical correlation, the fraction of windows clearing
  ρ=0.3, and the graded `stability_class` (`stable`/`partial`/`weak`/`rotating`).
  **This is the rolling replacement for "how many directions are macro-spanned."**
- `results/rolling_forecast[_beta]/macro_fingerprint.csv` — mean (sign-anchored) macro
  structure correlation per rank across windows (the interpretable, regime-averaged map).
- `results/rolling_forecast[_beta]/forecast_accuracy_pooled.csv` — pooled OOS R² vs
  AR(1)/mean/zero, overlapping and non-overlapping Clark–West p, per horizon.
- `results/rolling_sectors[_beta]/sector_summary.csv` — one row per block (overall / energy /
  agriculture / metals): leading-rank stability class, median cross-window cosine and OOS ρ,
  and the best forecast Clark–West p.
- `results/rolling_sectors[_beta]/sector_rank_stability.csv`,
  `sector_forecast_pooled.csv`, `sector_macro_fingerprint.csv` — the per-block detail.
- `reports/rolling_forecast[_beta]_report.md`, `reports/rolling_sectors[_beta]_report.md`
  — the narrative verdicts, with a pre-registered gate on share-of-gain language.

## Why CCA, and why out-of-sample

Autoencoder latents are identified only up to rotation/sign/permutation, so
per-factor R² understates recovery; CCA scores the **factor space**, which is the
invariant object. A 37-dim collinear macro panel can align with any small target
by chance in finite sample, so identity rests on **out-of-sample canonical
correlations vs a permutation/label-shuffle null**, not raw in-sample numbers.
See `docs/CCA_methods.md`.

## Caveats

Contemporaneous alignment; stationarity assumed; the macro panel contains **no
commodity prices** (no mechanical leakage). Publication-lagged series (`gpr`,
`epu`) are shifted one day. The h=21/63 forecast rows are power-bounded (few
effective non-overlapping origins) and are "untestable here", not evidence of
absence. On any daily cross-asset design, **respect the clock**: US macro closes
hours after commodity settlements, so naive same-day "lag-1" reads are
asynchronous-close artifacts — this engine embargoes/aligns accordingly.

## Reproducibility

Python ≥ 3.10. All randomness seeded from `--seed`. `make test` runs the suite;
`make lint` and `make typecheck` enforce style and types.
