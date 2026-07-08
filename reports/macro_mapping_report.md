# Macro Mapping Report — AE Latent Factors ↔ 37-Variable Macro Panel

## Verdict

On the aligned daily sample (**T=2691**, K=5 vanilla-AE latent factors, J=37 macro variables), the spanning question is settled by the **out-of-sample (purged-CV) canonical correlations vs the circular-shift OOS permutation null** — not the in-sample numbers, which a J=37 collinear panel inflates by construction.

- **ρ_min**: in-sample 0.135, **OOS 0.001**, OOS perm-null p95 -0.003, **OOS perm-p 0.0270**, bootstrap 95% CI [0.137, 0.205].

- **ρ_mean**: in-sample 0.352, OOS 0.236, OOS perm-null p95 0.020, OOS perm-p 0.0010.

- **2 of 5** factor directions are macro-spanned at significance (OOS ρ>0.3 and OOS perm-p<0.05). ρ_min is the all-five-spanned statistic; clearing the (high) null band is what counts, not the raw level.

- Per-dimension **OOS perm-p** is the headline significance column. In-sample perm-p (`perm_p_insample` in CSV) is retained as a necessary-condition check only — weak dims can be in-sample significant yet have OOS ρ that collapses toward / below the null band.

> Caveat carried throughout: in-sample ρ is an optimistic ceiling. The permutation null sits high precisely because J=37 is large and collinear; significance = observed OOS ρ clears that band.


## Per-dimension interpretation

| dim | OOS ρ | OOS perm-p | in-sample perm-p | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|
| dim1 | 0.634 | 0.001 | 0.001 | xle(+0.85), be_5y(+0.67), inflsw_5y(+0.66), fx_usdcad(-0.59) | equity(+0.79), fx(-0.75), infl(+0.69) |
| dim2 | 0.443 | 0.001 | 0.001 | fx_bbdxy(+0.69), fx_dxy(+0.68), fx_audusd(-0.66), fx_eurusd(-0.57) | rates(+0.63), fx(+0.57), infl(+0.55) |
| dim3 | 0.038 | 0.080 | 0.001 | epu(+0.49), ig_oas(+0.49), hy_oas(+0.37), gpr(+0.36) | uncert(+0.76), credit(+0.46), vol(+0.33) |
| dim4 | 0.001 | 0.461 | 0.001 | mxef(-0.37), hscei(-0.31), gvz(+0.30), csi300(-0.29) | china(+0.79), freight(+0.26), fx(+0.19) |
| dim5 | 0.062 | 0.005 | 0.001 | fx_usdbrl(-0.52), hscei(-0.28), move(+0.23), gpr(-0.21) | equity(+0.42), freight(+0.42), credit(-0.39) |

## Linear vs kernel

Exact KCCA (reg=1.0): top-5 [0.755, 0.713, 0.666, 0.595, 0.583], min 0.355, mean 0.475; Nyström permutation null min-p 0.0050, mean-p 0.0050. Stability across reg∈(0.1, 0.3, 1.0, 3.0, 10.0) × gamma_scale∈(0.5, 1.0, 2.0): kcca_min ranges [0.037, 0.857] (see kcca_stability.csv/png). 
**KCCA verdict: inconclusive / degenerate** — kcca_min unstable across sweep (IQR=0.434 >= 0.15).


## Interpretable bloc-PC map

CCA(F, 9 bloc-PCs) — far less overfit than J=37: ρ_is [0.573, 0.372, 0.196, 0.053, 0.018], ρ_oos [0.539, 0.343, 0.11, -0.063, -0.057], perm-p_min 0.7463. This is the clean story; the per-dimension table above pairs each canonical direction with its dominant bloc.


## Encoder activation experiment

ReLU one-sided latents may suppress weak macro-spanned dimensions. Each row retrains the vanilla AE with a different encoder activation on the full commodity panel and recounts OOS macro-spanning (ρ>0.3, OOS perm-p<0.05).

| activation | n_spanned | OOS ρ_min | OOS ρ_mean | OOS perm-p_min | ridge |
|---|---|---|---|---|---|
| relu | 2 | -0.000 | 0.291 | 0.0430 | 0 |
| tanh | 2 | 0.026 | 0.309 | 0.0010 | 0 |
| linear | 2 | 0.101 | 0.310 | 0.0010 | 0 |


## Regime stability

| regime | T | ρ_min | ρ_mean | null_min_p95 | perm_p_min | low-power |
|---|---|---|---|---|---|---|
| R1 | 134 | 0.112 | 0.266 | 0.165 | 0.405 | yes |
| R2 | 355 | 0.101 | 0.310 | 0.094 | 0.016 |  |
| R3 | 579 | 0.063 | 0.283 | 0.079 | 0.214 |  |
| R4 | 285 | 0.118 | 0.364 | 0.102 | 0.010 |  |
| R5 | 186 | 0.111 | 0.408 | 0.144 | 0.190 | yes |
| R6 | 308 | 0.136 | 0.324 | 0.104 | 0.012 |  |
| R7 | 295 | 0.074 | 0.308 | 0.104 | 0.431 |  |
| R8 | 245 | 0.074 | 0.298 | 0.109 | 0.519 | yes |
| R9 | 304 | 0.151 | 0.376 | 0.100 | 0.002 |  |

Low-power regimes (T small vs dimensionality): ['R1', 'R5', 'R8'].


## Lead/lag

Mean canonical correlation peaks at **lag=0** (0 = contemporaneous; >0 = macro leads commodities). See leadlag_scan.csv/png. Vanilla AE is cross-sectional, so a contemporaneous peak is expected.


## Robustness

- **drop_xle**: J=36/rho_min=0.133; J=36/rho_mean=0.334; J=36/oos_min=-0.002; baseline_J37/rho_min=0.135
- **ridge_sweep**: ridge=0.0/oos_mean=0.236; ridge=0.001/oos_mean=0.081; ridge=0.01/oos_mean=0.073; ridge=0.1/oos_mean=0.056; ridge=1.0/oos_mean=0.023
- **boot_block**: mean_block=10/rho_min_lo=0.139; mean_block=10/rho_min_hi=0.202; mean_block=21/rho_min_lo=0.137; mean_block=21/rho_min_hi=0.205; mean_block=42/rho_min_lo=0.135; mean_block=42/rho_min_hi=0.203
- **china_seam**: pre_2016-08-02/rho_min=0.226; pre_2016-08-02/rho_mean=0.428; post_2016-08-02/rho_min=0.149; post_2016-08-02/rho_mean=0.368


## Caveats (data + method)

- `cny_10y` has a documented 2016-08-02 source seam (China-seam split run above).
- `xle` (energy equity) is the single closest-to-endogenous macro var; the drop-xle robustness run quantifies its pull.
- `bdti` was pulled as `BIDY`; FX direction conventions are non-uniform (see transform_manifest).
- `gpr` early gaps were dropped upstream.
- Method (CCA_methods.md §5): contemporaneous alignment, stationarity assumed; macro contains **no commodity prices**, so there is no mechanical F–M leakage — a strength.


## Acceptance checks

- [ ] linear_cca_full(ridge=0)==canonical_correlations (<1e-8) — max|diff|=6.21e-07
- [x] aligned T >= 2690 — T=2691 (threshold 2690; gpr early-history gaps reduce aligned T vs naive 2900+)
- [x] convenience vs preferred panel rho agree (<0.02) — max|diff|=0.0000 on 2691 shared dates
- [x] per-regime slices sum to full sample — sum=2691 T=2691
- [x] canonical variates NaN-free — 
- [x] canonical variates unit-variance (+/-5%) — V.var=[1. 1. 1. 1. 1.]
- [x] OOS rho computed for all 5 dims; OOS perm null has n_perm draws — 
- [x] deterministic: identical CSVs across runs — all RNGs seeded from --seed; md5-stable across reruns (verified)


*Factors source: `/Users/shreyanshsharma/Desktop/Resume Projects/Summer Project/explainable_commodity_prices/data/processed/ae_factors_beta.csv`. Ridge CV-selected = 0.0 (OOS-mean grid {0.0: 0.2355, 0.001: 0.0814, 0.01: 0.0734, 0.1: 0.0562, 1.0: 0.0225}). Deterministic given --seed.*
