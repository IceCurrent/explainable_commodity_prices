# CCA between AE latent factors and a target panel — implementation notes

This doc explains the two canonical-correlation routines used to score how well the
autoencoder's latent factors `f` line up with a known target space `g` (synthetic
ground-truth factors today; the real macro panel `M` tomorrow). Both live in
[src/macro_mapping.py](src/macro_mapping.py) and are consumed by the synthetic
recovery harness in [src/synthetic_recovery.py](src/synthetic_recovery.py).

- **Linear CCA** — [`canonical_correlations`](src/macro_mapping.py#L98) → metrics `canon_min`, `canon_mean`.
- **Kernel CCA** — [`kernel_canonical_correlations`](src/macro_mapping.py#L139) → metrics `kcca_min`, `kcca_mean`.

> Read this alongside [AE_explainability_study.md](AE_explainability_study.md), which
> explains *why* these space-level metrics exist: AE latents are identified only up to
> permutation/sign/rotation, so a per-factor R² looks lossy even when no information is
> lost. CCA scores the **factor space**, which is the invariant object.

---

## 1. Why CCA at all

The encoder maps `x → f`. We want to know whether `f` carries the same information as
a target `g`, but we cannot compare them coordinate-by-coordinate: the AE is free to
mix, permute, sign-flip, and split the true factors across its latents. CCA answers the
**rotation-invariant** question:

> Is there *some* linear (linear CCA) or *some* smooth nonlinear (kernel CCA)
> recombination of `f` and of `g` that makes them line up?

It does this by finding pairs of canonical directions `(a, b)` maximizing
`corr(f a, g b)`, then the next pair orthogonal to the first, and so on. The resulting
**canonical correlations** `ρ₁ ≥ ρ₂ ≥ … ∈ [0, 1]` are the diagnostic:

- all `ρ` ≈ 1 → the spaces are equivalent (the target spans the factors);
- `min ρ` low → at least one direction in `f` has no counterpart in `g` (a factor not
  spanned by the target).

This is the canonical-correlation core of the **Bai & Ng (2006)** spanning test.

---

## 2. Linear CCA — `canonical_correlations`

### What it computes
Canonical correlations between the column space of the factor panel `F` (T×K) and the
target panel `M` (T×J), returned as a vector of length `min(K, J)`, descending, clipped
to `[0, 1]`.

### How (the whitened-cross-covariance / QR-SVD trick)
Rather than forming and inverting covariance matrices, it uses the standard numerically
stable route:

```text
1. Align F and M on common dates, mean-center each column.
2. Thin QR:  F = Qf Rf,  M = Qm Rm   →  Qf, Qm are orthonormal bases for the two spaces.
3. SVD of the cross-Gram:  s = svdvals(Qfᵀ Qm)
4. canonical correlations = clip(s, 0, 1)
```

The singular values of `Qfᵀ Qm` are exactly the cosines of the principal angles between
the two subspaces — i.e. the canonical correlations. Centering replaces the intercept;
QR whitening replaces the `Σff^{-1/2}`, `Σmm^{-1/2}` whitening in the textbook
generalized-eigenproblem form, and avoids explicitly inverting near-singular covariance
matrices.

### Key properties
- **No tuning parameters.** Deterministic, closed-form, O(T·max(K,J)²)-ish.
- **Invariant** to any invertible linear transform of either view (rotation, scaling,
  permutation, sign) — exactly the AE non-identifiability we want to quotient out.
- Wrapped by [`bai_ng_spanning_summary`](src/macro_mapping.py#L206), which reports
  `min`, `mean`, and the full vector.

### What `canon_min` / `canon_mean` mean in the study
On a clean linear DGP at K=6 over a true-rank-3 signal, per-factor `span_mean_r2`
collapses while `canon_min` stays ≈ 0.95 — the signature that the loss is **pure
coordinate rotation, recoverable**, not lost information.

---

## 3. Kernel CCA — `kernel_canonical_correlations`

The nonlinear analogue. Each view is lifted into an RBF reproducing-kernel Hilbert
space and **linear CCA is solved in that feature space** (Lai & Fyfe 2000; Bach & Jordan
2002; Hardoon et al. 2004). The canonical correlations then measure dependence
achievable by *arbitrary smooth nonlinear transforms* of `f` and `g` — a finite-sample,
capacity-controlled estimate of the Hirschfeld–Gebelein–Rényi maximal correlation.

### Pipeline
```text
1. Align, then standardize each view per-column (so the RBF bandwidth sees comparable scale).
2. RBF Gram matrices:  K = exp(-γ ||xᵢ - xⱼ||²),  γ from the median heuristic per view.
3. Double-center each Gram:  Kc = H K H,  H = I - 11ᵀ/n   (centers the feature map).
4. Eigendecompose each centered Gram:  K = U diag(λ) Uᵀ.
5. Regularized whitening per view:  d = sqrt( λ / (λ + κ) ),  κ = reg · mean(λ).
6. Inner matrix:  inner = (d_f ⊙ Ufᵀ Um) ⊙ d_m
7. canonical correlations = clip(svdvals(inner), 0, 1),  keep top_k (default J).
```

The whole thing reduces to **one n×n eigenproblem per view plus a small SVD** — the
derivation in the docstring shows the feature-space CCA collapses to the singular values
of `D_a (U_aᵀ U_b) D_b` with `D = diag(sqrt(λ/(λ+κ)))`.

### The regularization is the crux
Without the ridge `κ`, RBF kernels have enough high-frequency directions to drive *every*
canonical correlation to 1 — the well-known **KCCA degeneracy** (with enough capacity you
can always overfit two finite samples into perfect alignment). The factor `d = sqrt(λ/(λ+κ))`
shrinks low-eigenvalue (tail / high-frequency) directions toward 0 and leaves
high-eigenvalue (genuinely shared) directions near 1. `reg` (default `1.0`, i.e.
`κ = mean(λ)`) is the single most important knob.

### Other defaults worth knowing
- **`gamma` (RBF bandwidth):** median heuristic, `γ = 1/median(pairwise sq dist)`, per view.
- **`top_k`:** defaults to the number of target columns `J`, so the output is directly
  comparable to linear CCA's `min(K, J)` values. Without it, KCCA returns up to `n`
  mostly-spurious tail directions whose `min` is meaningless.
- **`eig_tol`:** drops eigenvalues below `1e-10 · λ_max` before whitening.

### What `kcca_min` / `kcca_mean` mean in the study
KCCA is the **space-level nonlinearity probe**. The tell is `kcca_min ≫ canon_min`:
the `f → g` relation is real but nonlinear, so the *linear* canonical correlations
understate recovery. Under the `tanh` DGP this shows up together with a positive E2
per-factor nonlinearity premium and a `canon_min` that drops to ≈ 0.88.

---

## 4. How the two compare

| | Linear CCA | Kernel CCA |
|---|---|---|
| Function | `canonical_correlations` | `kernel_canonical_correlations` |
| Detects | linear shared structure | linear **+ smooth nonlinear** shared structure |
| Tuning | none | `reg` (κ), `gamma_f/gamma_m`, `top_k`, `eig_tol` |
| Cost | O(T·max(K,J)²), one small SVD | one n×n eigendecomp per view (heavy in T) |
| Output length | `min(K, J)` | `top_k` (default J) |
| Failure mode | misses nonlinear maps (false "not spanned") | degenerates to all-1 if under-regularized (false "spanned") |
| Metric names | `canon_min`, `canon_mean` | `kcca_min`, `kcca_mean` |

Read them **together**: `canon` low + `kcca` high = real nonlinear relation. Both low =
genuinely unshared direction. Both high = linearly spanned.

---

## 5. Current limitations (read before moving to real data)

These are the things to revisit when `g` becomes the real macro panel rather than
synthetic ground truth.

**Shared to both**
1. **Linearly aligned dates only.** Both call `_align_xy`, an inner join on the index.
   Real macro is lower-frequency and has gaps/ragged edges; whatever forward-fill /
   resampling you do upstream determines the effective sample size. Mismatched frequency
   silently shrinks `T`.
2. **No leakage controls.** Unlike the E2 nonlinear mapping (which uses purged/embargoed
   CV), both CCA routines fit on the **full sample**. On autocorrelated real series the
   reported correlations are in-sample and optimistically biased. There is no held-out
   canonical correlation, no block bootstrap, no significance threshold — `min`/`mean`
   are point estimates with no error bar.
3. **Contemporaneous only.** They align `f_t` with `g_t`. Any lead/lag relationship
   (macro leading commodities or vice versa) is invisible; you would need to feed lagged
   columns explicitly.
4. **Stationarity assumed.** Centering/standardizing handles level and scale, not trends
   or regime shifts. Non-stationary real series can inflate correlations spuriously.

**Linear CCA specific**
5. **Misses nonlinearity** by construction — a real but nonlinear map reads as "not
   spanned." That's exactly why KCCA exists, but it means `canon_min` alone is not a
   verdict.
6. **No ridge.** If `K` or `J` is large relative to `T`, or the panel is highly
   collinear (the real macro FX/credit blocks are), the QR bases can be ill-conditioned
   and correlations biased upward. (Note `decompose_factors` *does* ridge for this exact
   reason; `canonical_correlations` does not.)

**Kernel CCA specific**
7. **Heavily tuning-dependent.** Results move with `reg` and `gamma`. The median
   heuristic and `reg=1.0` are reasonable defaults, **not** validated optima. On real
   data you should sweep `reg` and check the canonical correlations are stable, not
   pinned at 1 (under-regularized) or crushed to 0 (over-regularized).
8. **O(n²) memory / O(n³) eigendecomp.** Builds full n×n Gram matrices. Fine for the
   synthetic T; for multi-thousand-day real panels this is the binding cost. Consider
   Nyström / incomplete-Cholesky approximation or subsampling if T is large.
9. **No out-of-sample / permutation null.** The KCCA degeneracy means a high `kcca`
   needs a null to be credible. A permutation test (shuffle one view's rows, recompute,
   compare) is the natural addition and is currently absent.
10. **`top_k` truncation is a reporting choice.** It makes KCCA comparable to linear CCA
    but hides the tail; if you change the comparison you must revisit it.

---

## 6. Suggested additions for the real-data run

- Wrap both in a **block-bootstrap or permutation null** so `min`/`mean` carry a p-value.
- Add a **held-out canonical correlation** (fit directions on a train slice, evaluate
  correlation on a test slice) to detect overfitting, especially for KCCA.
- **Sweep `reg`/`gamma`** and report a stability curve, not a single KCCA number.
- Add **lagged target columns** to probe lead/lag.
- Ridge the linear CCA (or report its condition number) given the collinear real macro
  blocks.
- For large `T`, swap the exact Gram eigendecomp for a **Nyström approximation**.

---

### File map
- Implementations: [src/macro_mapping.py](src/macro_mapping.py)
  (`canonical_correlations`, `kernel_canonical_correlations`, `bai_ng_spanning_summary`,
  plus helpers `_rbf_gram`, `_median_gamma`, `_center_gram`).
- Consumers: [src/synthetic_recovery.py](src/synthetic_recovery.py) (`run_recovery` →
  `canon_*`, `kcca_*`), [scripts/run_synthetic_recovery.py](scripts/run_synthetic_recovery.py)
  (sweeps + report), [scripts/run_macro_mapping.py](scripts/run_macro_mapping.py)
  (real-data entry point).
- Conceptual framing: [AE_explainability_study.md](AE_explainability_study.md).
</content>
</invoke>
