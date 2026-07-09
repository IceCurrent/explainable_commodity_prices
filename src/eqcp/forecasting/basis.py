"""Train-frozen canonical-variate basis for factor-space attribution.

Why this exists. AE latents come with two distinct ambiguities: (i) the ReLU
architecture's own gauge (permutation x positive diagonal scaling leaves the
network's function unchanged), and (ii) the downstream pipeline's invariance —
OLS forecasting on [1, y_lag, s_t] and CCA cannot distinguish ANY invertible
affine reparametrization of the state block, and optimizer multiplicity makes
different seeds land in genuinely different latent coordinates. A per-raw-
coordinate attribution is therefore an attribution to arbitrary labels. The
fix: attribute in the CCA canonical-variate basis fit on the TRAIN segment
only and frozen. Canonical variates are invariant to invertible linear maps of
the factor block (up to sign, and up to rotation within blocks of tied
canonical correlations) PROVIDED the factor side is exactly whitened — so the
factor-side ridge is structurally pinned to 0 here; regularization is confined
to the 37-dim macro block, which does not disturb factor-side invariance.

Near-tied trailing correlations rotate freely under resampling, so individual
trailing directions are non-identified; the robust attribution level is the
grouped game over the {spanned, weakly-macro-correlated} blocks, with the
boundary placed at the largest adjacent gap in the train purged-CV rho
spectrum (falling back to the rho/p thresholds). Spanning labels use train
data only; the OOS segment is touched exclusively by forward diagnostics
(frozen-basis OOS rho, loading drift) computed after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eqcp.cca.inference import (
    perdim_perm_null_oos,
    purged_cv_canon,
    stationary_bootstrap_idx,
)
from eqcp.cca.linear import align_xy, linear_cca_full

RIDGE_F = 0.0  # structural: factor-side ridge must stay 0 or basis invariance is lost
FACTOR_EIG_TOL = 1e-8  # min eigenvalue of train factor correlation (dead-latent guard)
SIGN_ANCHOR_FLOOR = 0.10  # macro-side sign anchor only when |struct corr| clears this


@dataclass
class FrozenBasis:
    """CCA basis fit on the train segment; all fields are train-only objects."""

    A: np.ndarray  # (K, K) factor-side canonical vectors (columns)
    B: np.ndarray  # (J, K) macro-side canonical vectors
    mu_f: np.ndarray  # (K,) train mean of factors
    mu_m: np.ndarray  # (J,) train mean of macro
    rho_train: np.ndarray  # (K,) in-sample canonical correlations on train
    rho_cv_train: np.ndarray  # (K,) purged-CV canonical correlations within train
    perm_p_train: np.ndarray  # (K,) per-dim permutation p within train
    n_spanned: int  # leading block size (data-driven boundary)
    struct_m: np.ndarray  # (J, K) macro structure correlations on train
    struct_f: np.ndarray  # (K, K) factor structure correlations on train
    factor_cols: list[str]
    macro_cols: list[str]
    ridge_m: float
    factor_cond: float  # condition number of the train factor correlation

    @property
    def n_dims(self) -> int:
        return self.A.shape[1]

    @property
    def groups(self) -> list[tuple[int, ...]]:
        """Spanned block and weakly-macro-correlated block as grouped-game players."""
        return [
            tuple(range(self.n_spanned)),
            tuple(range(self.n_spanned, self.n_dims)),
        ]

    def variates(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Factor-side canonical variates V_t = (f_t - mu_f) A for any dates."""
        V = (factors[self.factor_cols].to_numpy(float) - self.mu_f) @ self.A
        cols = [f"V{k + 1}" for k in range(self.n_dims)]
        return pd.DataFrame(V, index=factors.index, columns=cols)

    def macro_variates(self, macro: pd.DataFrame) -> pd.DataFrame:
        """Macro-side canonical variates U_t = (m_t - mu_m) B for any dates."""
        U = (macro[self.macro_cols].to_numpy(float) - self.mu_m) @ self.B
        cols = [f"U{k + 1}" for k in range(self.n_dims)]
        return pd.DataFrame(U, index=macro.index, columns=cols)

    def top_loadings(self, k: int, n: int = 4) -> list[tuple[str, float]]:
        """Top-|structure-correlation| macro variables for canonical dim ``k``."""
        s = self.struct_m[:, k]
        order = np.argsort(-np.abs(s))[:n]
        return [(self.macro_cols[j], float(s[j])) for j in order]


def _spanned_boundary(
    rho_cv: np.ndarray,
    perm_p: np.ndarray,
    rho_min: float,
    p_max: float,
) -> int:
    """Leading-block size: largest adjacent gap in the train purged-CV spectrum.

    Falls back to the (rho > rho_min) & (p < p_max) threshold count when the
    spectrum is flat (max gap below 0.1) so the boundary is always defined.
    """
    gaps = rho_cv[:-1] - rho_cv[1:]
    if len(gaps) and float(np.max(gaps)) >= 0.1:
        return int(np.argmax(gaps)) + 1
    return int(np.sum((rho_cv > rho_min) & (perm_p < p_max)))


def fit_frozen_basis(
    factors_train: pd.DataFrame,
    macro_train: pd.DataFrame,
    ridge_grid: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 1.0),
    n_folds: int = 5,
    embargo: int = 10,
    n_perm: int = 200,
    seed: int = 0,
    spanned_rho_min: float = 0.3,
    spanned_p_max: float = 0.05,
) -> FrozenBasis:
    """Fit the canonical-variate basis and its spanning labels on train data only.

    The macro-side ridge is CV-selected on train; the factor-side ridge is
    pinned to ``RIDGE_F = 0`` (invariance requirement). Raises on a degenerate
    train factor block (dead/duplicated ReLU latents) instead of regularizing
    it away, because a factor-side ridge would silently void the invariance.
    """
    Fa, Ma = align_xy(factors_train, macro_train)
    K = Fa.shape[1]
    if len(Fa) < 10 * K:
        raise ValueError(f"train segment too short for a stable CCA basis (T={len(Fa)})")

    Fz = (Fa - Fa.mean()) / Fa.std(ddof=0).replace(0.0, np.nan)
    if Fz.isna().any().any():
        raise ValueError("degenerate (constant) latent factor in the train segment")
    eig = np.linalg.eigvalsh(np.corrcoef(Fz.to_numpy(float), rowvar=False))
    if eig.min() < FACTOR_EIG_TOL * eig.mean():
        raise ValueError(
            f"near-singular train factor block (min eig {eig.min():.2e}); "
            "dead or duplicated latents — refusing to build an attribution basis"
        )
    factor_cond = float(eig.max() / eig.min())

    scores: dict[float, float] = {}
    for rg in ridge_grid:
        oos = purged_cv_canon(Fa, Ma, n_folds=n_folds, embargo=embargo, ridge=rg, ridge_f=RIDGE_F)
        scores[rg] = float(np.nanmean(oos))
    ridge_m = max(scores, key=lambda k: scores[k])

    rho, A, B, _, _, struct_f, struct_m = linear_cca_full(Fa, Ma, ridge=ridge_m, ridge_f=RIDGE_F)
    if A.shape[1] != K or np.linalg.matrix_rank(A) < K:
        raise ValueError("factor-side canonical vectors are rank-deficient; basis not invertible")

    # Sign convention (cosmetic for PBSV — v(S) depends only on spans): anchor
    # on the largest macro structure correlation when it is informative,
    # otherwise on the factor side; ties broken by column order (argmax).
    for k in range(K):
        j = int(np.argmax(np.abs(struct_m[:, k])))
        anchor = struct_m[j, k] if abs(struct_m[j, k]) >= SIGN_ANCHOR_FLOOR else None
        if anchor is None:
            jf = int(np.argmax(np.abs(struct_f[:, k])))
            anchor = struct_f[jf, k]
        if anchor < 0:
            A[:, k] *= -1.0
            B[:, k] *= -1.0
            struct_m[:, k] *= -1.0
            struct_f[:, k] *= -1.0

    rho_cv = purged_cv_canon(
        Fa, Ma, n_folds=n_folds, embargo=embargo, ridge=ridge_m, ridge_f=RIDGE_F
    )
    null = perdim_perm_null_oos(
        Fa, Ma, n_perm, seed, K, n_folds, embargo, ridge_m, ridge_f=RIDGE_F
    )
    perm_p = (1 + np.sum(null >= rho_cv[None, :], axis=0)) / (n_perm + 1)
    n_spanned = _spanned_boundary(rho_cv, perm_p, spanned_rho_min, spanned_p_max)

    return FrozenBasis(
        A=A,
        B=B,
        mu_f=Fa.to_numpy(float).mean(axis=0),
        mu_m=Ma.to_numpy(float).mean(axis=0),
        rho_train=rho,
        rho_cv_train=rho_cv,
        perm_p_train=perm_p,
        n_spanned=n_spanned,
        struct_m=struct_m,
        struct_f=struct_f,
        factor_cols=list(Fa.columns),
        macro_cols=list(Ma.columns),
        ridge_m=float(ridge_m),
        factor_cond=factor_cond,
    )


def frozen_oos_diagnostics(
    basis: FrozenBasis,
    factors_oos: pd.DataFrame,
    macro_oos: pd.DataFrame,
) -> pd.DataFrame:
    """Strictly-forward validation of the frozen basis on the OOS segment.

    Per dimension: corr(v_k, u_k) with frozen A/B/means (the honest forward
    analogue of the train rho), and the cosine similarity between train
    structure correlations and OOS structure correlations of the frozen
    variates ("loading drift"; low cosine = the train-era macro name no longer
    describes the direction and must not be used in claims).
    """
    Fo, Mo = align_xy(factors_oos, macro_oos)
    V = basis.variates(Fo).to_numpy(float)
    U = basis.macro_variates(Mo).to_numpy(float)
    Mz = Mo.to_numpy(float)
    Mz = (Mz - Mz.mean(0)) / np.where(Mz.std(0) == 0, 1.0, Mz.std(0))
    rows = []
    for k in range(basis.n_dims):
        v = V[:, k] - V[:, k].mean()
        u = U[:, k] - U[:, k].mean()
        den = np.sqrt((v @ v) * (u @ u))
        rho_oos = float(v @ u / den) if den > 0 else np.nan
        vz = v / (v.std() if v.std() > 0 else 1.0)
        struct_oos = Mz.T @ vz / len(vz)
        tr = basis.struct_m[:, k]
        cos = float(tr @ struct_oos / (np.linalg.norm(tr) * np.linalg.norm(struct_oos) + 1e-12))
        rows.append(
            {
                "dim": f"V{k + 1}",
                "rho_train": float(basis.rho_train[k]),
                "rho_cv_train": float(basis.rho_cv_train[k]),
                "perm_p_train": float(basis.perm_p_train[k]),
                "rho_oos_frozen": rho_oos,
                "loading_cosine_oos": cos,
                "spanned": k < basis.n_spanned,
            }
        )
    return pd.DataFrame(rows)


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between column spaces of a and b."""
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    s = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)
    return np.arccos(s)[::-1]


def basis_stability_bootstrap(
    factors_train: pd.DataFrame,
    macro_train: pd.DataFrame,
    basis: FrozenBasis,
    n_boot: int = 200,
    mean_block: int = 21,
    seed: int = 0,
) -> pd.DataFrame:
    """Train-window block-bootstrap of the CCA basis (players' sampling noise).

    Refits the (ridge_m, ridge_f=0) CCA per stationary-bootstrap replicate of
    the train window and records the largest principal angle between the
    replicate's leading (spanned-block) subspace and the frozen one, plus the
    replicate's boundary. Basis noise is NOT propagated into phi CIs — this
    table quantifies it separately (phi CIs are conditional on the basis).
    """
    Fa, Ma = align_xy(factors_train, macro_train)
    Fn, Mn = Fa.to_numpy(float), Ma.to_numpy(float)
    T = len(Fa)
    ns = basis.n_spanned
    rng = np.random.default_rng(seed)
    rows = []
    for bi in range(n_boot):
        idx = stationary_bootstrap_idx(T, mean_block, rng)
        rho_b, A_b, *_ = linear_cca_full(Fn[idx], Mn[idx], ridge=basis.ridge_m, ridge_f=RIDGE_F)
        gaps = rho_b[:-1] - rho_b[1:]
        boundary_b = int(np.argmax(gaps)) + 1 if len(gaps) and gaps.max() >= 0.1 else ns
        max_angle = float(np.max(principal_angles(basis.A[:, :ns], A_b[:, :ns]))) if ns else np.nan
        rows.append(
            {
                "draw": bi,
                "max_principal_angle_deg": np.degrees(max_angle),
                "boundary": boundary_b,
                "boundary_matches": boundary_b == ns,
            }
        )
    return pd.DataFrame(rows)
