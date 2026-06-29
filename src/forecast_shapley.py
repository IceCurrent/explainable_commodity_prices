"""Forecast-based Shapley attribution of latent factors (phi_k).

This is the *forecast* Shapley layer referenced throughout the directive: it
ranks each aligned latent factor by its contribution to out-of-sample forecast
accuracy. Keep it conceptually separate from the E2 macro-SHAP layer, which
attributes the *mapping* f_k = h(M), not the forecast.

Value function. For a subset of factors S, the value is the out-of-sample MSE
reduction of the factor-augmented AR(1) restricted to S relative to the plain
AR(1) baseline:

    v(S) = MSE(AR1)  -  MSE(AR1 + factors in S)

so that v(empty) = 0 and the Shapley values satisfy the efficiency property
sum_k phi_k = v(full) = MSE(AR1) - MSE(AR1 + all K factors).

A positive phi_k means factor k reduces forecast error.

Implementation. With K factors there are only 2^K subsets, so the Shapley value
is computed *exactly* (no sampling). For each rolling window and commodity we
form the full normal-equations Gram once and solve each subset by indexing into
it -- avoiding 2^K independent least-squares fits. Factors are aligned to the
reference identity (E0) first, so "factor k" denotes the same object in every
window.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Callable

import numpy as np
import pandas as pd

from src.data import ReturnPanel, load_return_panel
from src.factor_alignment import Alignment
from src.factor_extraction import FactorBundle


def _aligned_window_factors(bundle: FactorBundle, alignment: Alignment, w: int) -> np.ndarray:
    """Aligned in-window factor series for window w, shape (M, K)."""
    return alignment.sign[w] * bundle.factors[w][:, alignment.perm[w]]


def _subset_mse_per_window(
    y_window: np.ndarray,
    f_window: np.ndarray,
    y_t: float,
    f_t: np.ndarray,
    actual: float,
    subsets: list[tuple[int, ...]],
) -> dict[tuple[int, ...], float]:
    """Squared forecast error for one (window, commodity) over all factor subsets.

    Uses one shared Gram matrix; each subset is a small symmetric solve.
    """
    y_lag = y_window[:-1]
    y_curr = y_window[1:]
    f_lag = f_window[:-1]
    n = y_lag.shape[0]
    K = f_lag.shape[1]

    # Full design D = [1, y_lag, f_lag] and its normal equations.
    D = np.empty((n, 2 + K))
    D[:, 0] = 1.0
    D[:, 1] = y_lag
    D[:, 2:] = f_lag
    G = D.T @ D
    b = D.T @ y_curr
    x_pred_full = np.concatenate([[1.0, y_t], f_t])

    out: dict[tuple[int, ...], float] = {}
    for S in subsets:
        idx = [0, 1] + [2 + j for j in S]
        Gs = G[np.ix_(idx, idx)]
        bs = b[idx]
        try:
            coef = np.linalg.solve(Gs, bs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(Gs, bs, rcond=None)[0]
        pred = float(coef @ x_pred_full[idx])
        out[S] = (actual - pred) ** 2
    return out


def _shapley_from_values(values: dict[tuple[int, ...], float], K: int) -> np.ndarray:
    """Exact Shapley values from a full subset->value table. v keyed by sorted tuple."""
    phi = np.zeros(K)
    factors = list(range(K))
    for k in factors:
        others = [j for j in factors if j != k]
        acc = 0.0
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(K - r - 1) / factorial(K)
            for combo in combinations(others, r):
                S = tuple(sorted(combo))
                S_k = tuple(sorted(combo + (k,)))
                acc += w * (values[S_k] - values[S])
        phi[k] = acc
    return phi


@dataclass
class ShapleyResult:
    factors: list[str]
    commodities: list[str]
    phi_per_commodity: np.ndarray   # (N, K)  MSE-reduction Shapley
    phi_aggregate: np.ndarray       # (K,)    pooled across all commodities
    mse_ar1: np.ndarray             # (N,)    baseline per commodity
    mse_full: np.ndarray            # (N,)    all-factor model per commodity
    mse_ar1_aggregate: float
    mse_full_aggregate: float

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.phi_per_commodity, index=self.commodities, columns=self.factors)
        df.index.name = "commodity"
        df["mse_ar1"] = self.mse_ar1
        df["mse_full"] = self.mse_full
        agg = pd.Series(
            list(self.phi_aggregate) + [self.mse_ar1_aggregate, self.mse_full_aggregate],
            index=self.factors + ["mse_ar1", "mse_full"],
            name="ALL",
        )
        return pd.concat([df, agg.to_frame().T])


def compute_forecast_shapley(
    bundle: FactorBundle,
    alignment: Alignment,
    panel: ReturnPanel | None = None,
    window_factor_provider: Callable[[int], np.ndarray] | None = None,
    verbose: bool = True,
) -> ShapleyResult:
    """Compute exact forecast-Shapley phi_k per commodity and pooled.

    Parameters
    ----------
    window_factor_provider : callable, optional
        ``w -> (M, K)`` aligned in-window factor series for window w. Defaults to
        the E0-aligned encoder factors. Override to feed the macro-spanned part
        ``f_hat`` or the orthogonal residual ``u`` for the E1 incremental test.
    """
    panel = panel or load_return_panel()
    W = bundle.n_windows
    K = bundle.n_factors
    N = bundle.n_assets
    provider = window_factor_provider or (lambda w: _aligned_window_factors(bundle, alignment, w))

    subsets = [tuple(sorted(c)) for r in range(K + 1) for c in combinations(range(K), r)]

    # Accumulate squared errors per (commodity, subset) and pooled per subset.
    sse_per_comm = {i: {S: 0.0 for S in subsets} for i in range(N)}
    count_per_comm = np.zeros(N, dtype=int)

    for w in range(W):
        if verbose and (w == 0 or (w + 1) % 200 == 0 or w == W - 1):
            print(f"  shapley window {w + 1}/{W}", flush=True)
        start, end = int(bundle.window_start_idx[w]), int(bundle.window_end_idx[w])
        f_window = provider(w)  # (M, K)
        f_t = f_window[-1]
        y_t_vec = panel.raw[end - 1]
        actual_vec = panel.raw[end]
        for i in range(N):
            y_window = panel.raw[start:end, i]
            errs = _subset_mse_per_window(
                y_window, f_window, float(y_t_vec[i]), f_t, float(actual_vec[i]), subsets
            )
            d = sse_per_comm[i]
            for S, e in errs.items():
                d[S] += e
            count_per_comm[i] += 1

    factor_names = [f"f{k+1}" for k in range(K)]
    phi_per_comm = np.zeros((N, K))
    mse_ar1 = np.zeros(N)
    mse_full = np.zeros(N)
    empty = tuple()
    full = tuple(range(K))

    pooled_sse = {S: 0.0 for S in subsets}
    pooled_count = 0
    for i in range(N):
        cnt = max(count_per_comm[i], 1)
        mse_vals = {S: sse_per_comm[i][S] / cnt for S in subsets}
        # value = MSE(AR1) - MSE(S)
        baseline = mse_vals[empty]
        values = {S: baseline - mse_vals[S] for S in subsets}
        phi_per_comm[i] = _shapley_from_values(values, K)
        mse_ar1[i] = mse_vals[empty]
        mse_full[i] = mse_vals[full]
        for S in subsets:
            pooled_sse[S] += sse_per_comm[i][S]
        pooled_count += cnt

    pooled_mse = {S: pooled_sse[S] / max(pooled_count, 1) for S in subsets}
    base = pooled_mse[empty]
    pooled_values = {S: base - pooled_mse[S] for S in subsets}
    phi_agg = _shapley_from_values(pooled_values, K)

    return ShapleyResult(
        factors=factor_names,
        commodities=bundle.commodities,
        phi_per_commodity=phi_per_comm,
        phi_aggregate=phi_agg,
        mse_ar1=mse_ar1,
        mse_full=mse_full,
        mse_ar1_aggregate=pooled_mse[empty],
        mse_full_aggregate=pooled_mse[full],
    )
