"""CCA inference: permutation nulls, purged CV, bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from eqcp.cca.linear import _as_array, linear_cca_full


def purged_time_series_folds(
    n: int, n_splits: int = 5, embargo: int = 21
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining CV folds with an embargo gap."""
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    size = max(n // (n_splits + 1), 1)
    for i in range(n_splits):
        t0 = (i + 1) * size
        t1 = min((i + 2) * size, n)
        if t0 >= n:
            break
        train_end = max(t0 - embargo, 0)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(t0, t1)
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


def purged_cv_canon(
    F,
    M,
    n_folds: int = 5,
    embargo: int = 10,
    ridge: float = 0.0,
    ridge_f: float | None = None,
) -> np.ndarray:
    Fa, Ma = _as_array(F), _as_array(M)
    T, K = Fa.shape
    r = min(K, Ma.shape[1])
    bounds = np.linspace(0, T, n_folds + 1).astype(int)
    per_fold: list[list[float]] = []
    for i in range(n_folds):
        ts, te = int(bounds[i]), int(bounds[i + 1])
        lo, hi = max(ts - embargo, 0), min(te + embargo, T)
        train = np.concatenate([np.arange(0, lo), np.arange(hi, T)])
        test = np.arange(ts, te)
        if len(train) <= K + 1 or len(test) < 3:
            continue
        muF, muM = Fa[train].mean(0), Ma[train].mean(0)
        _, A, B, _, _, _, _ = linear_cca_full(Fa[train], Ma[train], ridge=ridge, ridge_f=ridge_f)
        V_te = (Fa[test] - muF) @ A
        U_te = (Ma[test] - muM) @ B
        per_fold.append([_pearson(V_te[:, k], U_te[:, k]) for k in range(r)])
    if not per_fold:
        return np.full(r, np.nan)
    return np.nanmean(np.asarray(per_fold), axis=0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0


def select_ridge_oos(
    F,
    M,
    ridges: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 1.0),
    n_folds: int = 5,
    embargo: int = 10,
) -> tuple[float, dict[float, float]]:
    scores: dict[float, float] = {}
    for rg in ridges:
        oos = purged_cv_canon(F, M, n_folds=n_folds, embargo=embargo, ridge=rg)
        scores[rg] = float(np.nanmean(oos))
    best = max(scores, key=lambda k: scores[k])
    return best, scores


def circular_perm_null(
    score_fn: Callable,
    F,
    M,
    n_perm: int,
    seed: int,
) -> dict:
    """In-sample circular-shift permutation null (necessary-condition check)."""
    Fa, Ma = _as_array(F), _as_array(M)
    T = Ma.shape[0]
    obs = np.asarray(score_fn(Fa, Ma), float)
    obs_min, obs_mean = float(obs.min()), float(obs.mean())
    rng = np.random.default_rng(seed)
    null_min = np.empty(n_perm)
    null_mean = np.empty(n_perm)
    for b in range(n_perm):
        tau = int(rng.integers(1, T))
        s = np.asarray(score_fn(Fa, np.roll(Ma, tau, axis=0)), float)
        null_min[b] = s.min()
        null_mean[b] = s.mean()
    p_min = (1 + int(np.sum(null_min >= obs_min))) / (n_perm + 1)
    p_mean = (1 + int(np.sum(null_mean >= obs_mean))) / (n_perm + 1)
    return {
        "obs": obs,
        "obs_min": obs_min,
        "obs_mean": obs_mean,
        "null_min": null_min,
        "null_mean": null_mean,
        "null_min_p95": float(np.percentile(null_min, 95)),
        "null_mean_p95": float(np.percentile(null_mean, 95)),
        "null_min_mean": float(null_min.mean()),
        "null_mean_mean": float(null_mean.mean()),
        "p_min": p_min,
        "p_mean": p_mean,
    }


def circular_perm_null_oos(
    F,
    M,
    n_perm: int,
    seed: int,
    n_folds: int = 5,
    embargo: int = 10,
    ridge: float = 0.0,
) -> dict:
    """OOS circular-shift permutation null using purged-CV canonical correlations."""

    def oos_score(f_arr: np.ndarray, m_arr: np.ndarray) -> np.ndarray:
        return purged_cv_canon(f_arr, m_arr, n_folds=n_folds, embargo=embargo, ridge=ridge)

    Fa, Ma = _as_array(F), _as_array(M)
    T = Ma.shape[0]
    obs = np.asarray(oos_score(Fa, Ma), float)
    obs_min, obs_mean = float(np.nanmin(obs)), float(np.nanmean(obs))
    rng = np.random.default_rng(seed)
    null_min = np.empty(n_perm)
    null_mean = np.empty(n_perm)
    for b in range(n_perm):
        tau = int(rng.integers(1, T))
        s = np.asarray(oos_score(Fa, np.roll(Ma, tau, axis=0)), float)
        null_min[b] = float(np.nanmin(s))
        null_mean[b] = float(np.nanmean(s))
    p_min = (1 + int(np.sum(null_min >= obs_min))) / (n_perm + 1)
    p_mean = (1 + int(np.sum(null_mean >= obs_mean))) / (n_perm + 1)
    return {
        "obs": obs,
        "obs_min": obs_min,
        "obs_mean": obs_mean,
        "null_min": null_min,
        "null_mean": null_mean,
        "null_min_p95": float(np.percentile(null_min, 95)),
        "null_mean_p95": float(np.percentile(null_mean, 95)),
        "p_min": p_min,
        "p_mean": p_mean,
    }


def stationary_bootstrap_idx(T: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / max(mean_block, 1)
    idx = np.empty(T, dtype=int)
    t = 0
    cur = int(rng.integers(0, T))
    while t < T:
        idx[t] = cur
        t += 1
        if rng.random() < p:
            cur = int(rng.integers(0, T))
        else:
            cur = (cur + 1) % T
    return idx


def block_bootstrap_ci(
    F, M, ridge: float, mean_block: int = 21, n_boot: int = 1000, seed: int = 0
) -> dict:
    Fa, Ma = _as_array(F), _as_array(M)
    T = Fa.shape[0]
    rng = np.random.default_rng(seed)
    mins = np.empty(n_boot)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_idx(T, mean_block, rng)
        rho = linear_cca_full(Fa[idx], Ma[idx], ridge=ridge)[0]
        mins[b] = rho.min()
        means[b] = rho.mean()
    return {
        "min_lo": float(np.percentile(mins, 2.5)),
        "min_hi": float(np.percentile(mins, 97.5)),
        "mean_lo": float(np.percentile(means, 2.5)),
        "mean_hi": float(np.percentile(means, 97.5)),
    }


def perdim_perm_null(score_fn, F, M, n_perm: int, seed: int, r: int) -> np.ndarray:
    Fa, Ma = _as_array(F), _as_array(M)
    T = Ma.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_perm, r))
    for b in range(n_perm):
        tau = int(rng.integers(1, T))
        s = np.asarray(score_fn(Fa, np.roll(Ma, tau, axis=0)), float)
        draws[b] = s[:r]
    return draws


def perdim_perm_null_oos(
    F,
    M,
    n_perm: int,
    seed: int,
    r: int,
    n_folds: int,
    embargo: int,
    ridge: float,
    ridge_f: float | None = None,
) -> np.ndarray:
    def oos_score(f_arr, m_arr):
        return purged_cv_canon(
            f_arr, m_arr, n_folds=n_folds, embargo=embargo, ridge=ridge, ridge_f=ridge_f
        )

    Fa, Ma = _as_array(F), _as_array(M)
    T = Ma.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_perm, r))
    for b in range(n_perm):
        tau = int(rng.integers(1, T))
        s = np.asarray(oos_score(Fa, np.roll(Ma, tau, axis=0)), float)
        draws[b] = s[:r]
    return draws
