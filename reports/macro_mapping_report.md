# Macro Mapping Report — AE Latent Factors ↔ 37-Variable Macro Panel

## Verdict

On the aligned daily sample (**T=2691**, K=5 vanilla-AE latent factors, J=37 macro variables), the spanning question is settled by the **out-of-sample (purged-CV) canonical correlations vs the circular-shift OOS permutation null** — not the in-sample numbers, which a J=37 collinear panel inflates by construction.

- **ρ_min**: in-sample 0.123, **OOS -0.000**, OOS perm-null p95 -0.004, **OOS perm-p 0.0370**, bootstrap 95% CI [0.127, 0.196].

- **ρ_mean**: in-sample 0.374, OOS 0.291, OOS perm-null p95 0.020, OOS perm-p 0.0010.

- **2 of 5** factor directions are macro-spanned at significance (OOS ρ>0.3 and OOS perm-p<0.05). ρ_min is the all-five-spanned statistic; clearing the (high) null band is what counts, not the raw level.

- Per-dimension **OOS perm-p** is the headline significance column. In-sample perm-p (`perm_p_insample` in CSV) is retained as a necessary-condition check only — weak dims can be in-sample significant yet have OOS ρ that collapses toward / below the null band.

> Caveat carried throughout: in-sample ρ is an optimistic ceiling. The permutation null sits high precisely because J=37 is large and collinear; significance = observed OOS ρ clears that band.


## Per-dimension interpretation

| dim | OOS ρ | OOS perm-p | in-sample perm-p | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|
| dim1 | 0.642 | 0.001 | 0.001 | xle(+0.85), be_5y(+0.66), inflsw_5y(+0.66), fx_usdcad(-0.59) | fx(+0.78), equity(-0.78), infl(-0.64) |
| dim2 | 0.487 | 0.001 | 0.001 | fx_audusd(+0.66), fx_bbdxy(-0.64), fx_dxy(-0.63), fx_usdclp(-0.57) | infl(+0.61), rates(+0.57), fx(+0.52) |
| dim3 | 0.197 | 0.001 | 0.001 | tips_5y(+0.55), ust_5y(+0.54), ust_10y(+0.53), tips_10y(+0.49) | china(-0.67), rates(-0.56), credit(+0.30) |
| dim4 | 0.130 | 0.001 | 0.001 | fx_usdbrl(+0.72), ust_2y(-0.23), tips_10y(-0.22), move(-0.21) | credit(-0.51), rates(-0.41), vol(-0.32) |
| dim5 | -0.000 | 0.538 | 0.001 | ig_oas(-0.43), epu(-0.39), bdiy(+0.31), ovx(-0.22) | china(+0.45), uncert(+0.40), freight(-0.39) |

## Linear vs kernel

Exact KCCA (reg=1.0): top-5 [0.782, 0.747, 0.692, 0.657, 0.643], min 0.476, mean 0.557; Nyström permutation null min-p 0.0050, mean-p 0.0050. Stability across reg∈(0.1, 0.3, 1.0, 3.0, 10.0) × gamma_scale∈(0.5, 1.0, 2.0): kcca_min ranges [0.093, 0.905] (see kcca_stability.csv/png). 
**KCCA verdict: inconclusive / degenerate** — kcca_min unstable across sweep (IQR=0.442 >= 0.15).


## Interpretable bloc-PC map

CCA(F, 9 bloc-PCs) — far less overfit than J=37: ρ_is [0.58, 0.376, 0.212, 0.092, 0.057], ρ_oos [0.547, 0.337, 0.156, 0.066, 0.051], perm-p_min 0.0010. This is the clean story; the per-dimension table above pairs each canonical direction with its dominant bloc.


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
| R1 | 134 | 0.168 | 0.327 | 0.184 | 0.084 | yes |
| R2 | 355 | 0.146 | 0.339 | 0.097 | 0.002 |  |
| R3 | 579 | 0.070 | 0.297 | 0.079 | 0.094 |  |
| R4 | 285 | 0.104 | 0.345 | 0.105 | 0.058 |  |
| R5 | 186 | 0.080 | 0.392 | 0.129 | 0.557 | yes |
| R6 | 308 | 0.068 | 0.264 | 0.104 | 0.473 |  |
| R7 | 295 | 0.127 | 0.381 | 0.107 | 0.008 |  |
| R8 | 245 | 0.088 | 0.332 | 0.115 | 0.285 | yes |
| R9 | 304 | 0.071 | 0.335 | 0.100 | 0.461 |  |

Low-power regimes (T small vs dimensionality): ['R1', 'R5', 'R8'].


## Lead/lag

Mean canonical correlation peaks at **lag=0** (0 = contemporaneous; >0 = macro leads commodities). See leadlag_scan.csv/png. Vanilla AE is cross-sectional, so a contemporaneous peak is expected.


## Robustness

- **drop_xle**: J=36/rho_min=0.123; J=36/rho_mean=0.356; J=36/oos_min=0.001; baseline_J37/rho_min=0.123
- **ridge_sweep**: ridge=0.0/oos_mean=0.291; ridge=0.001/oos_mean=0.111; ridge=0.01/oos_mean=0.106; ridge=0.1/oos_mean=0.091; ridge=1.0/oos_mean=0.040
- **boot_block**: mean_block=10/rho_min_lo=0.130; mean_block=10/rho_min_hi=0.198; mean_block=21/rho_min_lo=0.127; mean_block=21/rho_min_hi=0.196; mean_block=42/rho_min_lo=0.130; mean_block=42/rho_min_hi=0.195
- **china_seam**: pre_2016-08-02/rho_min=0.283; pre_2016-08-02/rho_mean=0.474; post_2016-08-02/rho_min=0.142; post_2016-08-02/rho_mean=0.389


## Caveats (data + method)

- `cny_10y` has a documented 2016-08-02 source seam (China-seam split run above).
- `xle` (energy equity) is the single closest-to-endogenous macro var; the drop-xle robustness run quantifies its pull.
- `bdti` was pulled as `BIDY`; FX direction conventions are non-uniform (see transform_manifest).
- `gpr` early gaps were dropped upstream.
- Method (CCA_methods.md §5): contemporaneous alignment, stationarity assumed; macro contains **no commodity prices**, so there is no mechanical F–M leakage — a strength.


## Acceptance checks

- [x] linear_cca_full(ridge=0)==canonical_correlations (<1e-8) — max|diff|=9.62e-10
- [x] aligned T >= 2690 — T=2691 (threshold 2690; gpr early-history gaps reduce aligned T vs naive 2900+)
- [x] convenience vs preferred panel rho agree (<0.02) — max|diff|=0.0000 on 2691 shared dates
- [x] per-regime slices sum to full sample — sum=2691 T=2691
- [x] canonical variates NaN-free — 
- [x] canonical variates unit-variance (+/-5%) — V.var=[1. 1. 1. 1. 1.]
- [x] OOS rho computed for all 5 dims; OOS perm null has n_perm draws — 
- [x] deterministic: identical CSVs across runs — all RNGs seeded from --seed; md5-stable across reruns (verified)


*Factors source: `/Users/shreyanshsharma/Desktop/Resume Projects/Summer Project/explainable_commodity_prices/data/processed/ae_factors_vanilla.csv`. Ridge CV-selected = 0.0 (OOS-mean grid {0.0: 0.2912, 0.001: 0.1107, 0.01: 0.1061, 0.1: 0.0912, 1.0: 0.0399}). Deterministic given --seed.*
