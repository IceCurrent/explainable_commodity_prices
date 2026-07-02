# Sector-Wise Macro Spanning Report

Each sector's commodity return panel is compressed with the same vanilla autoencoder and tested for macro spanning with purged-CV out-of-sample canonical correlations vs a circular-shift permutation null (n_perm=500, seed=0). A direction counts as macro-spanned when its OOS ρ>0.3 and OOS perm-p<0.05.

## Headline by block

| block | commodities | factors | macro-spanned | OOS ρ (per dim) |
|---|---|---|---|---|
| overall | 21 | 5 | 2/5 | 0.642, 0.487, 0.197, 0.130, -0.000 |
| agriculture | 9 | 3 | 1/3 | 0.358, 0.046, 0.055 |
| energy | 5 | 3 | 1/3 | 0.645, 0.067, 0.018 |
| metals | 7 | 3 | 1/3 | 0.524, 0.235, -0.030 |


## Overall

21 commodities → 5 AE factors vs 37 macro variables; CV-selected ridge=0. **2 of 5** directions macro-spanned OOS.

| dim | in-sample ρ | OOS ρ | OOS perm-p | spanned | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|---|
| dim1 | 0.696 | 0.642 | 0.002 | yes | xle(+0.85); be_5y(+0.66); inflsw_5y(+0.66); fx_usdcad(-0.59) | fx(+0.78); equity(-0.78); infl(-0.64) |
| dim2 | 0.554 | 0.487 | 0.002 | yes | fx_audusd(+0.66); fx_bbdxy(-0.64); fx_dxy(-0.63); fx_usdclp(-0.57) | infl(+0.61); rates(+0.57); fx(+0.52) |
| dim3 | 0.295 | 0.197 | 0.002 |  | tips_5y(+0.55); ust_5y(+0.54); ust_10y(+0.53); tips_10y(+0.49) | china(-0.67); rates(-0.56); credit(+0.30) |
| dim4 | 0.204 | 0.130 | 0.002 |  | fx_usdbrl(+0.72); ust_2y(-0.23); tips_10y(-0.22); move(-0.21) | credit(-0.51); rates(-0.41); vol(-0.32) |
| dim5 | 0.123 | -0.000 | 0.539 |  | ig_oas(-0.43); epu(-0.39); bdiy(+0.31); ovx(-0.22) | china(+0.45); uncert(+0.40); freight(-0.39) |


## Agriculture

9 commodities → 3 AE factors vs 37 macro variables; CV-selected ridge=0. **1 of 3** directions macro-spanned OOS.

| dim | in-sample ρ | OOS ρ | OOS perm-p | spanned | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|---|
| dim1 | 0.415 | 0.358 | 0.002 | yes | fx_usdbrl(-0.70); fx_usdcad(-0.65); xle(+0.64); fx_audusd(+0.63) | fx(-0.78); equity(+0.77); infl(+0.62) |
| dim2 | 0.179 | 0.046 | 0.036 |  | move(-0.57); mxwo(+0.56); spx(+0.47); vix(-0.41) | vol(-0.72); equity(+0.46); credit(-0.41) |
| dim3 | 0.140 | 0.055 | 0.022 |  | xle(-0.35); fx_usdclp(-0.29); epu(-0.28); inflsw_5y(-0.26) | infl(-0.43); uncert(-0.42); fx(-0.41) |


## Energy

5 commodities → 3 AE factors vs 37 macro variables; CV-selected ridge=0. **1 of 3** directions macro-spanned OOS.

| dim | in-sample ρ | OOS ρ | OOS perm-p | spanned | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|---|
| dim1 | 0.690 | 0.645 | 0.002 | yes | xle(+0.84); be_5y(+0.67); inflsw_5y(+0.67); inflsw_10y(+0.55) | infl(+0.90); equity(+0.76); credit(-0.54) |
| dim2 | 0.230 | 0.067 | 0.006 |  | ovx(-0.63); ig_oas(-0.51); hy_oas(-0.39); mxwo(+0.38) | credit(+0.71); vol(+0.54); equity(-0.24) |
| dim3 | 0.125 | 0.018 | 0.269 |  | bdiy(+0.33); ig_oas(-0.29); ovx(+0.26); fx_dxy(-0.24) | freight(-0.77); fx(+0.41); vol(-0.17) |


## Metals

7 commodities → 3 AE factors vs 37 macro variables; CV-selected ridge=0. **1 of 3** directions macro-spanned OOS.

| dim | in-sample ρ | OOS ρ | OOS perm-p | spanned | top macro (struct corr) | bloc reading |
|---|---|---|---|---|---|---|
| dim1 | 0.583 | 0.524 | 0.002 | yes | fx_audusd(+0.79); fx_usdnok(-0.69); fx_bbdxy(-0.68); fx_usdclp(-0.65) | fx(-0.91); equity(+0.63); china(+0.53) |
| dim2 | 0.320 | 0.235 | 0.002 |  | ust_10y(-0.67); ust_5y(-0.67); ust_2y(-0.58); tips_10y(-0.53) | rates(-0.78); credit(+0.49); china(-0.46) |
| dim3 | 0.119 | -0.030 | 0.898 |  | fx_eurusd(-0.33); gvz(+0.32); ig_oas(+0.26); fx_usdnok(+0.25) | vol(+0.50); rates(-0.29); fx(+0.29) |


## Reading this

- The **overall** row is the whole-panel benchmark (same probe as the macro-mapping report); sector rows decompose it.
- OOS ρ near the permutation null band means the direction is *not* linearly recoverable from macro out-of-sample — it is sector-idiosyncratic.
- `top macro` / `bloc reading` name the macro variables and 9-bloc PCs that load on each direction (structure correlations), i.e. the interpretable driver.


*Deterministic given --seed=0. Autoencoder latents are identified only up to rotation/sign/permutation, so CCA scores the factor space (the invariant object), not individual coordinates.*
