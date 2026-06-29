from __future__ import annotations

import numpy as np


def fit_ar1(y: np.ndarray) -> np.ndarray:
    """
    Fit AR(1): y_t = alpha + beta * y_{t-1} + eps.

    Returns coefficients [alpha, beta].
    """
    x_lag = y[:-1]
    y_curr = y[1:]
    design = np.column_stack([np.ones(len(x_lag)), x_lag])
    coef, _, _, _ = np.linalg.lstsq(design, y_curr, rcond=None)
    return coef


def forecast_ar1(coef: np.ndarray, y_t: float) -> float:
    alpha, beta = coef
    return alpha + beta * y_t


def fit_factor_augmented_ar1(
    y: np.ndarray,
    factors: np.ndarray,
) -> np.ndarray:
    """
    Fit x_i,t = alpha + beta * x_i,t-1 + gamma' * f_{t-1} + eps  (paper eq. 4, p=1).

    Parameters
    ----------
    y : ndarray, shape (M,)
        Commodity return series in the window.
    factors : ndarray, shape (M, K)
        Latent factors aligned with y.

    Returns
    -------
    coef : ndarray, shape (2 + K,)
        [alpha, beta, gamma_1, ..., gamma_K]
    """
    x_lag = y[:-1]
    f_lag = factors[:-1]
    y_curr = y[1:]
    design = np.column_stack([np.ones(len(x_lag)), x_lag, f_lag])
    coef, _, _, _ = np.linalg.lstsq(design, y_curr, rcond=None)
    return coef


def forecast_factor_augmented_ar1(
    coef: np.ndarray,
    y_t: float,
    f_t: np.ndarray,
) -> float:
    """One-step-ahead forecast (paper eq. 6, p=1)."""
    alpha = coef[0]
    beta = coef[1]
    gamma = coef[2:]
    return alpha + beta * y_t + float(np.dot(gamma, f_t))
