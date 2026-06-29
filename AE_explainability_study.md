# Study: Where does the autoencoder lose explainability, and is it recoverable?

**Motivation.** The real commodity→macro maps (directive E1/E2) come out weak. On
real data that is unattributable: a weak map could mean the relation isn't there,
the AE destroyed it, or the probe is too blunt — three confounded explanations and
no ground truth to separate them. This study removes the confound by testing the
**same AE and the same E1/E2 probes** on data whose generating relation we *know*.
The AE becomes the only unknown in the chain, so any loss is attributable to it.

Scope is deliberately narrow: **only the two mapping methods we already have** —
E1 (linear spanning + Bai-Ng canonical correlations) and E2 (gradient-boosted
nonlinear mapping + premium). No E3/E4/E5. This is a separate track from the
commodity directive, which is left untouched.

---

## The core idea: decompose "explainability loss" into named, separable parts

"Explainability lost in the AE" is not one quantity. Encoding `x → f` and decoding
`f → x̂` can lose information in four distinct ways, and the contribution is to
**measure each separately**:

| Mechanism | What it is | Recoverable? | How we isolate it |
|---|---|---|---|
| **Coordinate rotation / splitting** | AE mixes, permutes, sign-flips, or splits the true factors across latents | **Yes** — by an invertible map | gap between **per-factor R²** (coordinate level) and **canonical corr** (space level) |
| **ReLU rectification** | encoder zeros negatives; a signed direction may need a +half and −half unit | partly | `activation="relu"` vs `"linear"` on identical training |
| **Rank mismatch** | bottleneck K ≠ true rank r | no, if K < r | sweep K ∈ {r−1, r, r+1, 2r} |
| **Reconstruction residual / nonlinearity** | what the decoder can't rebuild; nonlinear relation invisible to a linear probe | no / needs E2 | recon R²; E2 nonlinearity premium |

**The key conceptual claim:** most *apparent* explainability loss is coordinate
non-identifiability (rotation/splitting), which is recoverable by an invertible
transform. AE latents are identified only up to permutation, sign, and
reparametrization, so a **per-factor** map will look lossy even when no
information is lost. The right object is the **factor space** (canonical
correlations / Bai-Ng), not individual coordinates. Separating "lost to coordinate
choice" from "lost for real" is the deliverable.

---

## Data-generating process (`src/synthetic_recovery.py`)

We write the DGP, so we know everything:

```
g_t   : true latent factors, AR(1) (macro-like persistence), standardized  (T × r)
driver: D = relation(g)                                                     (T × r_eff)
x_t   : D @ B.T + noise,   noise scaled to a target SNR                     (T × n)
```

`relation ∈ {linear, interaction, tanh}`: `linear` → `D = g`; `interaction` adds a
hidden `g₀·g₁` dimension to the driver but **not** to the `g` handed to the probe
(so E2 must recover it); `tanh` passes each factor through a saturating
nonlinearity. Pipeline then mirrors the real one exactly: z-score `x` → train the
project `VanillaAutoencoder` → hand latents `f` and true factors `g` to E1/E2.

**Encoders compared:** `relu` (the project AE), `linear` (same AE, no ReLU), `pca`
(closed-form top-K — the interpretability gold standard). If the ReLU AE recovers a
*known* relation worse than PCA, the architecture is destroying explainability
rather than the data hiding it.

## Metrics (`RecoveryResult`)

- `recon_r2` — information surviving encode→decode.
- `span_mean_r2` / `span_max_r2` — **E1 coordinate level**: per-factor R²(g → fₖ).
- `canon_min` / `canon_mean` — **E1 space level**: Bai-Ng canonical correlations
  between the f-space and the g-space. High here + low `span_mean_r2` ⟹ loss is
  pure rotation, recoverable.
- `subspace_min` — decoder-column span vs true-loading span (linear DGP only).
- `n_active` — latents with non-trivial variance (ReLU dead-unit check).
- `nonlin_premium` — **E2**: mean (R²_nonlinear − R²_linear).

## Sweeps (`scripts/run_synthetic_recovery.py` → `results/synthetic/`)

- **A. Bottleneck rank** — K ∈ {r−1,…,2r} on a clean linear DGP. Rank mismatch and
  factor-splitting.
- **B. Encoder vs noise** — relu/linear/pca across SNR at K=r. Cost of ReLU and noise.
- **C. Linear vs nonlinear** — linear/interaction/tanh with E2 on. When the
  nonlinearity hides the relation from the linear probe.

---

## First findings (initial run)

1. **The loss is overwhelmingly coordinate rotation, and it is recoverable.** At
   K=6 on a true-rank-3 linear DGP, per-factor `span_mean_r2` collapses (ReLU 0.66,
   PCA 0.47) while `canon_min` stays ≈ 0.95. The factor *space* is intact; only the
   *coordinates* are scrambled. **This is the leading candidate explanation for the
   weak commodity maps: with K=5 over a likely rank-2/3 signal, each economic factor
   is split across latents, so per-factor E1 looks terrible while the space is fine.**
   Actionable: judge the real maps by canonical correlations / Bai-Ng, not per-factor R².
2. **Over-completeness trades reconstruction for interpretability, monotonically.**
   As K grows past r, `recon_r2` rises but `span_mean_r2` falls. More capacity =
   better reconstruction, worse per-factor identity.
3. **ReLU is not the villain on linear data.** relu ≈ linear ≈ pca at K=r; the
   damage is the bottleneck/over-completeness, not the rectification.
4. **E2 earns its keep exactly when the relation is nonlinear.** Under `tanh`, the
   E2 nonlinearity premium is clearly positive (~0.1–0.18) and `canon_min` drops to
   ~0.88 — the linear probe genuinely understates recovery, and E2 flags it.

---

## Bridge back to real data (anchor, deferred)

**Fisher triple** — `ust_10y`, `tips_10y`, `breakeven_10y` — obey the exact identity
`breakeven = nominal − real` (verified: residual std 0.008 pct pts over 3,288 days).
Three real series on a known 2-D plane: a real-data sanity check that the synthetic
conclusions aren't generation artifacts. Run after the synthetic sweeps settle.
