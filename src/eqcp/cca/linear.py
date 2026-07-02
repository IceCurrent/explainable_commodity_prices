"""Linear canonical correlation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_xy(factors: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = factors.index.intersection(macro.index)
    if len(idx) == 0:
        raise ValueError("factor and macro panels share no dates after alignment")
    return factors.loc[idx], macro.loc[idx]


def _as_array(X) -> np.ndarray:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy(dtype=float)
    return np.asarray(X, float)


def _ridged_inv_sqrt_from_data(Xc: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    T = Xc.shape[0]
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (s**2) / T
    r = ridge * float(lam.mean())
    inv = 1.0 / np.sqrt(lam + r)
    return (Vt.T * inv) @ Vt, r


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0


def _structure_corr(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    Xs = X - X.mean(0)
    Vs = V - V.mean(0)
    Xn = Xs / np.where(Xs.std(0) == 0, 1.0, Xs.std(0))
    Vn = Vs / np.where(Vs.std(0) == 0, 1.0, Vs.std(0))
    return (Xn.T @ Vn) / X.shape[0]


def canonical_correlations(factors: pd.DataFrame, macro: pd.DataFrame) -> np.ndarray:
    """Canonical correlations via SVD of whitened cross-covariance (ridge=0 anchor)."""
    f, m = align_xy(factors, macro)
    F = (f - f.mean()).to_numpy()
    M = (m - m.mean()).to_numpy()
    Qf, _ = np.linalg.qr(F)
    Qm, _ = np.linalg.qr(M)
    s = np.linalg.svd(Qf.T @ Qm, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def linear_cca_full(F, M, ridge: float = 0.0, ridge_f: float | None = None):
    """Full ridged linear CCA; ``ridge=0`` reproduces :func:`canonical_correlations`.

    ``ridge`` regularizes the macro (second) block; ``ridge_f`` overrides the
    factor-side ridge (default: same as ``ridge``). Factor-side canonical
    variates are invariant to invertible linear transforms of the factor block
    only when the factor side is exactly whitened (``ridge_f=0``) — the frozen
    attribution basis in :mod:`eqcp.forecasting.basis` relies on this.
    """
    Fa, Ma = _as_array(F), _as_array(M)
    if Fa.shape[0] != Ma.shape[0]:
        raise ValueError("F and M must be row-aligned before linear_cca_full")
    T, K = Fa.shape
    J = Ma.shape[1]
    Fc = Fa - Fa.mean(0)
    Mc = Ma - Ma.mean(0)
    Sfm = Fc.T @ Mc / T
    Sff_isq, _ = _ridged_inv_sqrt_from_data(Fc, ridge if ridge_f is None else ridge_f)
    Smm_isq, _ = _ridged_inv_sqrt_from_data(Mc, ridge)
    Omega = Sff_isq @ Sfm @ Smm_isq
    P, s, Qt = np.linalg.svd(Omega, full_matrices=False)
    r = min(K, J)
    corrs = np.clip(s[:r], 0.0, 1.0)
    A = Sff_isq @ P[:, :r]
    B = Smm_isq @ Qt.T[:, :r]
    V = Fc @ A
    U = Mc @ B
    struct_factor = _structure_corr(Fa, V)
    struct_macro = _structure_corr(Ma, U)
    return corrs, A, B, U, V, struct_factor, struct_macro


def bai_ng_spanning_summary(factors: pd.DataFrame, macro: pd.DataFrame) -> dict:
    cc = canonical_correlations(factors, macro)
    return {
        "canonical_correlations": cc.tolist(),
        "min_canonical_corr": float(cc.min()),
        "mean_canonical_corr": float(cc.mean()),
        "n_factors": int(factors.shape[1]),
        "n_macro": int(macro.shape[1]),
    }
