"""Kernel canonical correlation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqcp.cca.linear import _as_array, align_xy, linear_cca_full


def median_gamma(X: np.ndarray) -> float:
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    iu = np.triu_indices_from(d2, k=1)
    med = float(np.median(np.maximum(d2[iu], 0.0)))
    return 1.0 / med if med > 0 else 1.0


def _rbf_gram(X: np.ndarray, gamma: float) -> np.ndarray:
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _center_gram(K: np.ndarray) -> np.ndarray:
    rm = K.mean(axis=0, keepdims=True)
    Kc = K - rm - rm.T + K.mean()
    return 0.5 * (Kc + Kc.T)


def kernel_canonical_correlations(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    reg: float = 1.0,
    top_k: int | None = None,
    gamma_f: float | None = None,
    gamma_m: float | None = None,
    eig_tol: float = 1e-10,
) -> np.ndarray:
    f, m = align_xy(factors, macro)
    F = f.to_numpy(dtype=float)
    M = m.to_numpy(dtype=float)
    F = (F - F.mean(0)) / np.where(F.std(0, ddof=0) == 0, 1.0, F.std(0, ddof=0))
    M = (M - M.mean(0)) / np.where(M.std(0, ddof=0) == 0, 1.0, M.std(0, ddof=0))

    gamma_f = median_gamma(F) if gamma_f is None else gamma_f
    gamma_m = median_gamma(M) if gamma_m is None else gamma_m
    Kf = _center_gram(_rbf_gram(F, gamma_f))
    Km = _center_gram(_rbf_gram(M, gamma_m))

    def _whiten(K: np.ndarray):
        lam, U = np.linalg.eigh(K)
        keep = lam > eig_tol * float(lam.max())
        lam, U = lam[keep], U[:, keep]
        kappa = reg * float(lam.mean())
        d = np.sqrt(lam / (lam + kappa))
        return U, d

    Uf, df = _whiten(Kf)
    Um, dm = _whiten(Km)
    inner = (df[:, None] * (Uf.T @ Um)) * dm[None, :]
    cc = np.linalg.svd(inner, compute_uv=False)
    cc = np.clip(cc, 0.0, 1.0)
    k = m.shape[1] if top_k is None else top_k
    return cc[:k]


def _rbf_cross(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    aa = np.einsum("ij,ij->i", A, A)
    bb = np.einsum("ij,ij->i", B, B)
    d2 = aa[:, None] + bb[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _nystrom_features(
    X: np.ndarray, gamma: float, landmarks: np.ndarray, eig_tol: float = 1e-10
) -> np.ndarray:
    Xl = X[landmarks]
    W = _rbf_cross(Xl, Xl, gamma)
    E = _rbf_cross(X, Xl, gamma)
    lam, Q = np.linalg.eigh(0.5 * (W + W.T))
    keep = lam > eig_tol * float(lam.max())
    lam, Q = lam[keep], Q[:, keep]
    W_isq = (Q * (1.0 / np.sqrt(lam))) @ Q.T
    Z = E @ W_isq
    return Z - Z.mean(0)


def kcca_nystrom(
    F,
    M,
    reg: float = 1.0,
    gamma_f: float | None = None,
    gamma_m: float | None = None,
    n_landmarks: int = 400,
    top_k: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    Fa, Ma = _as_array(F), _as_array(M)
    Fs = (Fa - Fa.mean(0)) / np.where(Fa.std(0) == 0, 1.0, Fa.std(0))
    Ms = (Ma - Ma.mean(0)) / np.where(Ma.std(0) == 0, 1.0, Ma.std(0))
    gf = median_gamma(Fs) if gamma_f is None else gamma_f
    gm = median_gamma(Ms) if gamma_m is None else gamma_m
    T = Fs.shape[0]
    rng = np.random.default_rng(seed)
    L = min(n_landmarks, T)
    landmarks = rng.choice(T, size=L, replace=False)
    Zf = _nystrom_features(Fs, gf, landmarks)
    Zm = _nystrom_features(Ms, gm, landmarks)
    cc = linear_cca_full(Zf, Zm, ridge=reg)[0]
    cc = np.clip(cc, 0.0, 1.0)
    k = Ma.shape[1] if top_k is None else top_k
    return cc[:k]
