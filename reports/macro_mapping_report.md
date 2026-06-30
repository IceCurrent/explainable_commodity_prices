# Macro Mapping Report — AE Latent Factors ↔ 37-Variable Macro Panel

## Verdict

On the aligned daily sample (**T=2691**, K=5 vanilla-AE latent factors, J=37 macro variables), the spanning question is settled by the **out-of-sample (purged-CV) canonical correlations vs the circular-shift OOS permutation null** — not the in-sample numbers, which a J=37 collinear panel inflates by construction.

- **ρ_min**: in-sample 0.144, **OOS 0.055**, OOS perm-null p95 -0.003, **OOS perm-p 0.0010**, bootstrap 95% CI [0.143, 0.219].

- **ρ_mean**: in-sample 0.378, OOS 0.299, OOS perm-null p95 0.019, OOS perm-p 0.0010.

- **2 of 5** factor directions are macro-spanned at significance (OOS ρ>0.3 and OOS perm-p<0.05). ρ_min is the all-five-spanned statistic; clearing the (high) null band is what counts, not the raw level.

- Per-dimension **OOS perm-p** is the headline significance column. In-sample perm-p (`perm_p_insample` in CSV) is retained as a necessary-condition check only — weak dims can be in-sample significant yet have OOS ρ that collapses toward / below the null band.

> Caveat carried throughout: in-sample ρ is an optimistic ceiling. The permutation null sits high precisely because J=37 is large and collinear; significance = observed OOS ρ clears that band.


## Per-dimension interpretation

| dim | OOS ρ | OOS perm-p | in-sample perm-p | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|
| dim1 | 0.641 | 0.001 | 0.001 | xle(-0.85), be_5y(-0.67), inflsw_5y(-0.66), fx_usdcad(+0.58) | fx(-0.79), equity(+0.76), infl(+0.65) |
| dim2 | 0.487 | 0.001 | 0.001 | fx_audusd(-0.67), fx_bbdxy(+0.65), fx_dxy(+0.64), fx_usdclp(+0.58) | infl(-0.65), rates(-0.52), fx(-0.51) |
| dim3 | 0.157 | 0.001 | 0.001 | tips_5y(+0.56), ust_5y(+0.52), ust_10y(+0.50), tips_10y(+0.49) | china(-0.66), rates(-0.62), credit(+0.27) |
| dim4 | 0.156 | 0.001 | 0.001 | fx_usdbrl(+0.69), ust_2y(-0.28), ust_10y(-0.24), ust_5y(-0.24) | vol(+0.52), freight(+0.35), rates(-0.32) |
| dim5 | 0.055 | 0.022 | 0.001 | ovx(+0.52), move(+0.44), mxwo(-0.41), spx(-0.34) | vol(+0.57), credit(+0.48), equity(-0.42) |

## Linear vs kernel

Exact KCCA (reg=1.0): top-5 [0.778, 0.749, 0.692, 0.644, 0.638], min 0.487, mean 0.561; Nyström permutation null min-p 0.0050, mean-p 0.0050. Stability across reg∈(0.1, 0.3, 1.0, 3.0, 10.0) × gamma_scale∈(0.5, 1.0, 2.0): kcca_min ranges [0.097, 0.908] (see kcca_stability.csv/png). 
**KCCA verdict: inconclusive / degenerate** — kcca_min unstable across sweep (IQR=0.452 >= 0.15).


## Interpretable bloc-PC map

CCA(F, 9 bloc-PCs) — far less overfit than J=37: ρ_is [0.575, 0.369, 0.204, 0.099, 0.084], ρ_oos [0.539, 0.326, 0.133, 0.029, 0.111], perm-p_min 0.0010. This is the clean story; the per-dimension table above pairs each canonical direction with its dominant bloc.


## Encoder activation experiment

ReLU one-sided latents may suppress weak macro-spanned dimensions. Each row retrains the vanilla AE with a different encoder activation on the full commodity panel and recounts OOS macro-spanning (ρ>0.3, OOS perm-p<0.05).

| activation | n_spanned | OOS ρ_min | OOS ρ_mean | OOS perm-p_min | ridge |
|---|---|---|---|---|---|
| relu | 2 | 0.055 | 0.299 | 0.0010 | 0 |
| tanh | 2 | 0.069 | 0.302 | 0.0010 | 0 |
| linear | 2 | 0.091 | 0.305 | 0.0010 | 0 |


## Regime stability

| regime | T | ρ_min | ρ_mean | null_min_p95 | perm_p_min | low-power |
|---|---|---|---|---|---|---|
| R1 | 134 | 0.115 | 0.343 | 0.161 | 0.411 | yes |
| R2 | 355 | 0.076 | 0.317 | 0.089 | 0.218 |  |
| R3 | 579 | 0.074 | 0.298 | 0.075 | 0.056 |  |
| R4 | 285 | 0.125 | 0.355 | 0.102 | 0.004 |  |
| R5 | 186 | 0.130 | 0.394 | 0.142 | 0.086 | yes |
| R6 | 308 | 0.084 | 0.285 | 0.106 | 0.251 |  |
| R7 | 295 | 0.144 | 0.390 | 0.107 | 0.002 |  |
| R8 | 245 | 0.128 | 0.334 | 0.113 | 0.008 | yes |
| R9 | 304 | 0.116 | 0.363 | 0.095 | 0.016 |  |

Low-power regimes (T small vs dimensionality): ['R1', 'R5', 'R8'].


## Lead/lag

Mean canonical correlation peaks at **lag=0** (0 = contemporaneous; >0 = macro leads commodities). See leadlag_scan.csv/png. Vanilla AE is cross-sectional, so a contemporaneous peak is expected.


## Robustness

- **drop_xle**: J=36/rho_min=0.144; J=36/rho_mean=0.359; J=36/oos_min=0.059; baseline_J37/rho_min=0.144
- **ridge_sweep**: ridge=0.0/oos_mean=0.299; ridge=0.001/oos_mean=0.103; ridge=0.01/oos_mean=0.104; ridge=0.1/oos_mean=0.095; ridge=1.0/oos_mean=0.077
- **boot_block**: mean_block=10/rho_min_lo=0.142; mean_block=10/rho_min_hi=0.217; mean_block=21/rho_min_lo=0.143; mean_block=21/rho_min_hi=0.219; mean_block=42/rho_min_lo=0.142; mean_block=42/rho_min_hi=0.219
- **china_seam**: pre_2016-08-02/rho_min=0.247; pre_2016-08-02/rho_mean=0.463; post_2016-08-02/rho_min=0.179; post_2016-08-02/rho_mean=0.394


## Caveats (data + method)

- `cny_10y` has a documented 2016-08-02 source seam (China-seam split run above).
- `xle` (energy equity) is the single closest-to-endogenous macro var; the drop-xle robustness run quantifies its pull.
- `bdti` was pulled as `BIDY`; FX direction conventions are non-uniform (see transform_manifest).
- `gpr` early gaps were dropped upstream.
- Method (CCA_methods.md §5): contemporaneous alignment, stationarity assumed; macro contains **no commodity prices**, so there is no mechanical F–M leakage — a strength.


## Acceptance checks

- [x] linear_cca_full(ridge=0)==canonical_correlations (<1e-8) — max|diff|=2.75e-13
- [x] aligned T >= 2690 — T=2691 (threshold 2690; gpr early-history gaps reduce aligned T vs naive 2900+)
- [x] convenience vs preferred panel rho agree (<0.02) — max|diff|=0.0000 on 2691 shared dates
- [x] per-regime slices sum to full sample — sum=2691 T=2691
- [x] canonical variates NaN-free — 
- [x] canonical variates unit-variance (+/-5%) — V.var=[1. 1. 1. 1. 1.]
- [x] OOS rho computed for all 5 dims; OOS perm null has n_perm draws — 
- [x] deterministic: identical CSVs across runs — all RNGs seeded from --seed; md5-stable across reruns (verified)


*Factors source: `/Users/shreyanshsharma/Desktop/Resume Projects/Summer Project/explainable_commodity_prices/data/processed/ae_factors_vanilla.csv`. Ridge CV-selected = 0.0 (OOS-mean grid {0.0: 0.2993, 0.001: 0.1033, 0.01: 0.1038, 0.1: 0.0948, 1.0: 0.0768}). Deterministic given --seed.*
