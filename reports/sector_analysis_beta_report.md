# Sector-Wise Macro Spanning Report

Each sector's commodity return panel is compressed with the same β-VAE and tested for macro spanning with purged-CV out-of-sample canonical correlations vs a circular-shift permutation null (n_perm=500, seed=0). A direction counts as macro-spanned when its OOS ρ>0.3 and OOS perm-p<0.05.

## Headline by block

| block | commodities | factors | macro-spanned | OOS ρ (per dim) |
|---|---|---|---|---|
| overall | 21 | 5 | 2/5 | 0.634, 0.443, 0.038, 0.001, 0.062 |
| agriculture | 9 | 3 | 1/3 | 0.329, 0.057, 0.052 |
| energy | 5 | 3 | 1/3 | 0.422, 0.047, 0.044 |
| metals | 7 | 3 | 1/3 | 0.543, 0.136, 0.051 |


## Overall

21 commodities → 5 AE factors vs 37 macro variables; CV-selected ridge=0. **2 of 5** directions macro-spanned OOS.

| dim | in-sample ρ | OOS ρ | OOS perm-p | spanned | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|---|
| dim1 | 0.687 | 0.634 | 0.002 | yes | xle(+0.85); be_5y(+0.67); inflsw_5y(+0.66); fx_usdcad(-0.59) | equity(+0.79); fx(-0.75); infl(+0.69) |
| dim2 | 0.513 | 0.443 | 0.002 | yes | fx_bbdxy(+0.69); fx_dxy(+0.68); fx_audusd(-0.66); fx_eurusd(-0.57) | rates(+0.63); fx(+0.57); infl(+0.55) |
| dim3 | 0.260 | 0.038 | 0.068 |  | epu(+0.49); ig_oas(+0.49); hy_oas(+0.37); gpr(+0.36) | uncert(+0.76); credit(+0.46); vol(+0.33) |
| dim4 | 0.163 | 0.001 | 0.437 |  | mxef(-0.37); hscei(-0.31); gvz(+0.30); csi300(-0.29) | china(+0.79); freight(+0.26); fx(+0.19) |
| dim5 | 0.135 | 0.062 | 0.004 |  | fx_usdbrl(-0.52); hscei(-0.28); move(+0.23); gpr(-0.21) | equity(+0.42); freight(+0.42); credit(-0.39) |


## Agriculture

9 commodities → 3 AE factors vs 37 macro variables; CV-selected ridge=0. **1 of 3** directions macro-spanned OOS.

| dim | in-sample ρ | OOS ρ | OOS perm-p | spanned | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|---|
| dim1 | 0.385 | 0.329 | 0.002 | yes | fx_usdbrl(-0.72); fx_usdcad(-0.63); xle(+0.61); fx_audusd(+0.58) | fx(-0.80); equity(+0.70); infl(+0.63) |
| dim2 | 0.170 | 0.057 | 0.014 |  | xle(+0.53); spx(+0.53); mxwo(+0.51); hy_oas(-0.42) | equity(+0.58); credit(-0.52); vol(-0.46) |
| dim3 | 0.147 | 0.052 | 0.022 |  | ig_oas(-0.43); xle(+0.38); fx_usdcnh(-0.34); fx_usdnok(-0.30) | credit(-0.59); freight(+0.53); equity(+0.25) |


## Energy

5 commodities → 3 AE factors vs 37 macro variables; CV-selected ridge=0. **1 of 3** directions macro-spanned OOS.

| dim | in-sample ρ | OOS ρ | OOS perm-p | spanned | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|---|
| dim1 | 0.547 | 0.422 | 0.002 | yes | xle(+0.87); inflsw_5y(+0.64); be_5y(+0.63); inflsw_10y(+0.54) | equity(+0.83); infl(+0.78); credit(-0.72) |
| dim2 | 0.247 | 0.047 | 0.026 |  | hy_oas(-0.48); ig_oas(-0.47); ovx(-0.43); mxwo(+0.40) | uncert(+0.88); vol(+0.30); rates(-0.15) |
| dim3 | 0.190 | 0.044 | 0.038 |  | epu(+0.63); gpr(+0.62); ig_oas(-0.18); hy_oas(-0.16) | vol(+0.53); credit(+0.51); uncert(-0.34) |


## Metals

7 commodities → 3 AE factors vs 37 macro variables; CV-selected ridge=0. **1 of 3** directions macro-spanned OOS.

| dim | in-sample ρ | OOS ρ | OOS perm-p | spanned | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|---|
| dim1 | 0.592 | 0.543 | 0.002 | yes | fx_audusd(-0.80); fx_bbdxy(+0.68); fx_usdnok(+0.68); fx_usdclp(+0.65) | fx(+0.91); equity(-0.63); china(-0.52) |
| dim2 | 0.280 | 0.136 | 0.002 |  | epu(+0.66); gvz(+0.48); gpr(+0.32); fx_dxy(+0.25) | uncert(+0.82); rates(+0.32); vol(+0.29) |
| dim3 | 0.177 | 0.051 | 0.026 |  | gvz(-0.41); ust_10y(+0.36); tips_10y(+0.34); ust_5y(+0.33) | rates(+0.67); vol(-0.56); equity(+0.19) |


## Reading this

- The **overall** row is the whole-panel benchmark (same probe as the macro-mapping report); sector rows decompose it.
- OOS ρ near the permutation null band means the direction is *not* linearly recoverable from macro out-of-sample — it is sector-idiosyncratic.
- `top macro` / `bloc reading` name the macro variables and 9-bloc PCs that load on each direction (structure correlations), i.e. the interpretable driver.


*Deterministic given --seed=0. Autoencoder latents are identified only up to rotation/sign/permutation, so CCA scores the factor space (the invariant object), not individual coordinates.*
