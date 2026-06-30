"""Synthetic AE-explainability study.

Controlled testbed for the question the real commodity->macro mapping cannot
answer: *how much explainability does the autoencoder lose during encode/decode,
and is it recoverable?* On real data there is no ground truth, so a weak E1/E2
map is unattributable (weak relation? AE destroyed it? probe too weak?). Here we
write the data-generating process ourselves: known latent factors ``g``, a known
map to observed inputs ``x``, known noise. The AE is then the only unknown in the
chain, so any loss is attributable to it.

Pipeline mirrors the real one: z-score ``x`` -> train the same ``VanillaAutoencoder``
-> hand the latent factors ``f`` and the *true* factors ``g`` to the same E1
(``spanning_regression`` / ``canonical_correlations``) and E2
(``nonlinear_macro_mapping``) machinery used on commodities.

Decomposition of "explainability loss" this exposes:
  * coordinate rotation  -> recoverable; caught by canonical corr (space-level)
                            being high while per-factor R^2 (coordinate-level) is low.
  * ReLU rectification    -> isolated by comparing activation="relu" vs "linear".
  * rank mismatch (K<r)   -> swept via ``n_factors``.
  * reconstruction residual / nonlinearity -> recon R^2 and the E2 premium.

The PCA / linear-AE baseline is the crux: if the ReLU AE recovers a *known*
relation worse than PCA does, the architecture is destroying explainability
rather than the data hiding it -- the candidate explanation for the poor real maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eqcp.cca.kernel import kernel_canonical_correlations
from eqcp.cca.linear import canonical_correlations
from eqcp.factors.autoencoder import AETrainConfig, train_vanilla_autoencoder
from eqcp.spanning.regression import nonlinear_macro_mapping, spanning_regression


# --------------------------------------------------------------------------- #
# Data-generating process
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticConfig:
    """Knobs for the synthetic DGP. Defaults give a clean linear rank-3 base."""

    n_obs: int = 1500
    n_inputs: int = 12  # observed series fed to the AE (the "commodities")
    true_rank: int = 3  # number of independent latent drivers g
    snr: float = 4.0  # signal variance / noise variance
    relation: str = "linear"  # "linear" | "interaction" | "tanh"
    ar1: float = 0.95  # persistence of each latent factor (macro-like)
    seed: int = 0


@dataclass
class SyntheticData:
    x: np.ndarray  # (T, n_inputs) observed inputs (pre-z-score)
    g: pd.DataFrame  # (T, true_rank) true latent factors -> the "macro"
    loadings: np.ndarray  # (n_inputs, r_eff) B; columns span the signal space
    driver: np.ndarray  # (T, r_eff) D such that signal = D @ B.T
    index: pd.DatetimeIndex


def _ar1_factors(n_obs: int, rank: int, ar1: float, rng: np.random.Generator) -> np.ndarray:
    """Persistent (AR(1)) standardized latent factors, mimicking macro dynamics."""
    g = np.zeros((n_obs, rank))
    eps = rng.standard_normal((n_obs, rank)) * np.sqrt(1 - ar1**2)
    g[0] = rng.standard_normal(rank)
    for t in range(1, n_obs):
        g[t] = ar1 * g[t - 1] + eps[t]
    g -= g.mean(axis=0, keepdims=True)
    g /= g.std(axis=0, ddof=0, keepdims=True)
    return g


def make_synthetic(cfg: SyntheticConfig) -> SyntheticData:
    """Build (x, g, B) for the requested relation, with SNR-controlled noise.

    ``g`` (the base factors handed to E1/E2) always has ``true_rank`` columns.
    The *driver* that actually generates ``x`` may have extra columns under a
    nonlinear relation (e.g. an interaction g0*g1) -- those are deliberately
    hidden from the linear probe so the E2 premium can reveal them.
    """
    rng = np.random.default_rng(cfg.seed)
    r = cfg.true_rank
    g = _ar1_factors(cfg.n_obs, r, cfg.ar1, rng)

    if cfg.relation == "linear":
        driver = g
    elif cfg.relation == "interaction":
        if r < 2:
            raise ValueError("interaction relation needs true_rank >= 2")
        inter = (g[:, 0] * g[:, 1])[:, None]
        inter = (inter - inter.mean()) / inter.std(ddof=0)
        driver = np.column_stack([g, inter])
    elif cfg.relation == "tanh":
        driver = np.tanh(1.5 * g)
        driver = (driver - driver.mean(axis=0)) / driver.std(axis=0, ddof=0)
    else:
        raise ValueError(f"unknown relation {cfg.relation!r}")

    r_eff = driver.shape[1]
    loadings = rng.standard_normal((cfg.n_inputs, r_eff))
    signal = driver @ loadings.T

    noise = rng.standard_normal(signal.shape)
    noise *= np.sqrt(signal.var() / (cfg.snr * noise.var()))
    x = signal + noise

    index = pd.bdate_range("2014-01-01", periods=cfg.n_obs)
    g_df = pd.DataFrame(g, index=index, columns=[f"g{i + 1}" for i in range(r)])
    return SyntheticData(x=x, g=g_df, loadings=loadings, driver=driver, index=index)


# --------------------------------------------------------------------------- #
# Encoders: ReLU AE, linear AE, PCA
# --------------------------------------------------------------------------- #
def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=0, keepdims=True)
    return (x - mu) / np.where(sd == 0, 1.0, sd)


def _encode_ae(xz: np.ndarray, n_factors: int, activation: str, seed: int):
    """Train the project's VanillaAutoencoder; return (factors, recon, decoder_W)."""
    import torch

    cfg = AETrainConfig(n_factors=n_factors, activation=activation, seed=seed)
    model, factors = train_vanilla_autoencoder(xz, cfg)
    with torch.no_grad():
        recon = model(torch.tensor(xz, dtype=torch.float32)).cpu().numpy()
    return factors, recon, model.decoder_weight()


def _encode_pca(xz: np.ndarray, n_factors: int):
    """Top-K PCA: the linear interpretability baseline."""
    u, s, vt = np.linalg.svd(xz, full_matrices=False)
    comps = vt[:n_factors]  # (K, n)
    factors = xz @ comps.T  # (T, K)
    recon = factors @ comps  # back-projection
    return factors, recon, comps.T  # decoder_W analogue (n, K)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _recon_r2(xz: np.ndarray, recon: np.ndarray) -> float:
    sse = float(((xz - recon) ** 2).sum())
    sst = float(((xz - xz.mean(axis=0)) ** 2).sum())
    return 1.0 - sse / sst


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Canonical correlations between the column spaces of A and B.

    Singular values of Q_a^T Q_b = cosines of the principal angles. All near 1
    means span(B) lies inside span(A): the AE's decoder subspace recovers the
    true loading subspace regardless of per-factor rotation/sign/permutation.
    """
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


@dataclass
class RecoveryResult:
    encoder: str
    n_factors: int
    relation: str
    true_rank: int
    snr: float
    recon_r2: float
    span_mean_r2: float  # E1: mean per-factor R^2(g -> f_k)  [coordinate level]
    span_max_r2: float
    canon_min: float  # E1 Bai-Ng: min canonical corr(f, g) [space level]
    canon_mean: float
    kcca_min: float  # E2 kernel-CCA: min nonlinear canonical corr(f, g) [space level]
    kcca_mean: float
    subspace_min: float  # decoder span vs true loading span (linear only; else NaN)
    n_active: int  # latents with non-negligible variance (ReLU dead-unit check)
    nonlin_premium: float  # E2: mean (R2_nonlin - R2_lin); NaN if E2 skipped
    extra: dict = field(default_factory=dict)


def run_recovery(
    cfg: SyntheticConfig,
    n_factors: int,
    encoder: str = "relu",
    run_e2: bool = False,
) -> RecoveryResult:
    """Generate data, encode, and score recovery of the known relation.

    ``encoder`` in {"relu", "linear", "pca"}. ``relu`` is the project AE; ``linear``
    is the same AE without the ReLU; ``pca`` is the closed-form baseline.
    """
    data = make_synthetic(cfg)
    xz = _zscore(data.x)

    if encoder == "pca":
        factors, recon, dec_w = _encode_pca(xz, n_factors)
    elif encoder in ("relu", "linear"):
        factors, recon, dec_w = _encode_ae(xz, n_factors, encoder, cfg.seed)
    else:
        raise ValueError(f"unknown encoder {encoder!r}")

    f_df = pd.DataFrame(factors, index=data.index, columns=[f"f{i + 1}" for i in range(n_factors)])

    # Drop dead/constant latents before the regressions (ReLU can kill units).
    active_mask = f_df.std(ddof=0).to_numpy() > 1e-8
    n_active = int(active_mask.sum())
    f_live = f_df.loc[:, active_mask] if n_active > 0 else f_df

    # E1 coordinate level: per-factor linear spanning by the true g.
    if n_active > 0:
        span = spanning_regression(f_live, data.g)
        r2s = np.array([r.r2 for r in span.values()])
        span_mean_r2, span_max_r2 = float(r2s.mean()), float(r2s.max())
        # E1 space level: Bai-Ng canonical correlations between f-space and g-space.
        cc = canonical_correlations(f_live, data.g)
        canon_min, canon_mean = float(cc.min()), float(cc.mean())
        # E2 space level: kernel-CCA -- the nonlinear analogue of canon. High here
        # while canon_min is low => the f->g relation is real but nonlinear, so the
        # linear probe understates recovery (the space-level nonlinearity premium).
        kcc = kernel_canonical_correlations(f_live, data.g)
        kcca_min, kcca_mean = float(kcc.min()), float(kcc.mean())
    else:
        span_mean_r2 = span_max_r2 = canon_min = canon_mean = float("nan")
        kcca_min = kcca_mean = float("nan")

    # Subspace recovery only well-defined when the signal is linear in g.
    if cfg.relation == "linear":
        sub = subspace_overlap(dec_w, data.loadings)
        subspace_min = float(sub.min())
    else:
        subspace_min = float("nan")

    nonlin_premium = float("nan")
    if run_e2 and n_active > 0:
        e2 = nonlinear_macro_mapping(f_live, data.g)
        nonlin_premium = float(np.mean([r.nonlinearity_premium for r in e2.values()]))

    return RecoveryResult(
        encoder=encoder,
        n_factors=n_factors,
        relation=cfg.relation,
        true_rank=cfg.true_rank,
        snr=cfg.snr,
        recon_r2=_recon_r2(xz, recon),
        span_mean_r2=span_mean_r2,
        span_max_r2=span_max_r2,
        canon_min=canon_min,
        canon_mean=canon_mean,
        kcca_min=kcca_min,
        kcca_mean=kcca_mean,
        subspace_min=subspace_min,
        n_active=n_active,
        nonlin_premium=nonlin_premium,
    )


def result_to_row(r: RecoveryResult) -> dict:
    return {
        "encoder": r.encoder,
        "n_factors": r.n_factors,
        "relation": r.relation,
        "true_rank": r.true_rank,
        "snr": r.snr,
        "recon_r2": r.recon_r2,
        "span_mean_r2": r.span_mean_r2,
        "span_max_r2": r.span_max_r2,
        "canon_min": r.canon_min,
        "canon_mean": r.canon_mean,
        "kcca_min": r.kcca_min,
        "kcca_mean": r.kcca_mean,
        "subspace_min": r.subspace_min,
        "n_active": r.n_active,
        "nonlin_premium": r.nonlin_premium,
    }
