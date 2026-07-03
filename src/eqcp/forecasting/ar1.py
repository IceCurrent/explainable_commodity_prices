"""Factor-augmented AR(1) forecasting with expanding-window subset refits.

Model (direct h-step, per commodity i):

    y^(h)_{i,t+h} = alpha_i + beta_i y^(h)_{i,t} + delta_i' c_{i,t} + gamma_i' s_t + eps

where y^(h) is the trailing h-day cumulative log return, c_{i,t} are optional
always-included controls (e.g. EWMA vol, 12m momentum), and s_t is the latent
factor state. The OLS fit on the span of the regressors is invariant to
invertible affine transforms of s_t, so the FULL-model forecast is basis-free;
only the per-coordinate decomposition depends on the basis, which is why the
attribution happens in the frozen canonical-variate basis
(:mod:`eqcp.forecasting.basis`).

For the forecast-based Shapley values every factor subset S must be refit, so
the engine maintains one expanding Gram matrix per commodity and solves every
model in one batched masked solve per origin (excluded coefficients pinned to
zero via an identity block) — 2^K exact refits per origin at small-linear-solve
cost.

Leakage contract: a forecast issued at origin t uses only design rows tau <= t
and training pairs whose TARGET is fully realized by t (tau + h <= t). For
h > 1 this means the last usable regressor date is t - h; rows enter the Gram
only once their cumulative target completes.

Benchmarks: ZERO (driftless random walk), MEAN (expanding mean), AR (own-lag
AR(1) + controls). The PBSV value baseline is the empty factor subset ().
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations

import numpy as np
from scipy import stats

ZERO_KEY: tuple[int, ...] = (-3,)  # zero forecast (driftless random walk)
MEAN_KEY: tuple[int, ...] = (-2,)  # expanding-mean benchmark (intercept only)
AR_KEY: tuple[int, ...] = ()  # AR(1) + controls, no factors: the PBSV baseline
BENCH_KEYS = (ZERO_KEY, MEAN_KEY)


def all_subsets(k: int) -> list[tuple[int, ...]]:
    """All 2^k factor subsets, sorted by size then lexicographically."""
    return list(chain.from_iterable(combinations(range(k), r) for r in range(k + 1)))


def h_period_returns(returns: np.ndarray, h: int) -> np.ndarray:
    """Trailing h-day cumulative returns; row t sums rows (t-h+1 .. t), NaN before."""
    if h == 1:
        return returns.astype(float).copy()
    T = returns.shape[0]
    out = np.full(returns.shape, np.nan)
    c = np.cumsum(returns.astype(float), axis=0)
    out[h - 1 :] = c[h - 1 :] - np.vstack([np.zeros((1, returns.shape[1])), c[: T - h]])
    return out


def ewma_vol(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """(T, N) RiskMetrics EWMA daily volatility, seeded with the first observation."""
    T, N = returns.shape
    var = np.empty((T, N))
    var[0] = returns[0] ** 2
    for t in range(1, T):
        var[t] = lam * var[t - 1] + (1 - lam) * returns[t] ** 2
    return np.sqrt(var)


def staleness_stats(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-commodity (zero-return fraction, longest flat run) staleness diagnostics."""
    zero = np.isclose(returns, 0.0)
    frac = zero.mean(axis=0)
    longest = np.zeros(returns.shape[1], dtype=int)
    for i in range(returns.shape[1]):
        run = best = 0
        for z in zero[:, i]:
            run = run + 1 if z else 0
            best = max(best, run)
        longest[i] = best
    return frac, longest


@dataclass
class SubsetForecasts:
    """Expanding-window forecasts for every requested model on the OOS origins."""

    keys: list[tuple[int, ...]]  # ZERO_KEY, MEAN_KEY, AR_KEY, factor subsets
    origins: np.ndarray  # (T_oos,) forecast-origin indices into the full calendar
    targets: np.ndarray  # (T_oos,) target indices (= origins + h)
    preds: np.ndarray  # (n_models, T_oos, N)
    actual: np.ndarray  # (T_oos, N) realized h-period returns
    horizon: int

    def key_index(self, key: tuple[int, ...]) -> int:
        return self.keys.index(key if key in BENCH_KEYS else tuple(sorted(key)))

    def sqerr(self, key: tuple[int, ...]) -> np.ndarray:
        """(T_oos, N) squared errors for one model."""
        return (self.actual - self.preds[self.key_index(key)]) ** 2

    def mse(self, key: tuple[int, ...], weights: np.ndarray | None = None) -> float:
        """Pooled MSE; ``weights`` (N,) rescales each commodity (e.g. 1/train-var)."""
        e2 = self.sqerr(key)
        if weights is not None:
            e2 = e2 * weights[None, :]
        return float(e2.mean())

    def per_day_loss(self, key: tuple[int, ...], weights: np.ndarray | None = None) -> np.ndarray:
        """(T_oos,) cross-commodity mean loss per origin (bootstrap/placebo input)."""
        e2 = self.sqerr(key)
        if weights is not None:
            e2 = e2 * weights[None, :]
        return e2.mean(axis=1)

    def select(self, cols: np.ndarray) -> SubsetForecasts:
        """Restrict to a subset of commodities (e.g. the non-stale headline panel)."""
        return SubsetForecasts(
            keys=self.keys,
            origins=self.origins,
            targets=self.targets,
            preds=self.preds[:, :, cols],
            actual=self.actual[:, cols],
            horizon=self.horizon,
        )

    def select_origins(self, mask: np.ndarray) -> SubsetForecasts:
        """Restrict to a subset of OOS origins (e.g. non-overlapping targets)."""
        return SubsetForecasts(
            keys=self.keys,
            origins=self.origins[mask],
            targets=self.targets[mask],
            preds=self.preds[:, mask],
            actual=self.actual[mask],
            horizon=self.horizon,
        )

    def nonoverlap_mask(self) -> np.ndarray:
        """Greedy mask of origins whose h-period targets do NOT overlap.

        For h > 1 the trailing-h cumulative targets of adjacent origins share
        h-1 daily returns, so pooled tests on all origins over-count
        information; this mask keeps the largest left-to-right set of origins
        spaced >= h apart in the target calendar, giving an autocorrelation-free
        sample for honest long-horizon inference (at the cost of efficiency).
        """
        mask = np.zeros(len(self.origins), dtype=bool)
        last = -(10**9)
        for i, tgt in enumerate(self.targets):
            if int(tgt) - last >= self.horizon:
                mask[i] = True
                last = int(tgt)
        return mask


def expanding_subset_forecasts(
    returns: np.ndarray,
    state: np.ndarray,
    oos_start: int,
    horizon: int = 1,
    subsets: list[tuple[int, ...]] | None = None,
    controls: np.ndarray | None = None,
    available: np.ndarray | None = None,
    min_train: int = 60,
) -> SubsetForecasts:
    """Run the expanding FAAR for the benchmarks and each factor subset.

    ``oos_start`` is the first TARGET index evaluated. ``controls`` (T, N, C)
    are per-commodity always-included regressors added to every model except
    ZERO/MEAN. ``available`` masks calendar rows usable as design rows and
    origins (e.g. macro availability for substitution arms); comparisons
    across runs must share it.
    """
    T, N = returns.shape
    K = state.shape[1]
    n_ctrl = 0 if controls is None else controls.shape[2]
    subsets = all_subsets(K) if subsets is None else [tuple(sorted(s)) for s in subsets]
    keys: list[tuple[int, ...]] = [ZERO_KEY, MEAN_KEY, AR_KEY]
    keys += [s for s in subsets if s != AR_KEY]
    if available is None:
        available = np.ones(T, dtype=bool)
    if not 0 < oos_start < T:
        raise ValueError(f"oos_start {oos_start} outside calendar of length {T}")

    y = h_period_returns(returns, horizon)  # (T, N), NaN before h-1
    P = 2 + n_ctrl + K  # [1, y_lag, controls..., state...]
    state_off = 2 + n_ctrl
    masks = np.zeros((len(keys), P), dtype=bool)
    for m_i, key in enumerate(keys):
        if key == ZERO_KEY:
            continue  # never solved; prediction is 0
        masks[m_i, 0] = True
        if key != MEAN_KEY:
            masks[m_i, 1 : 2 + n_ctrl] = True
            masks[m_i, [state_off + j for j in key]] = True

    origins = np.array(
        [
            t
            for t in range(horizon, T - horizon)
            if available[t] and t + horizon >= oos_start and np.isfinite(y[t]).all()
        ],
        dtype=int,
    )
    if len(origins) == 0:
        raise ValueError("no valid OOS origins; check oos_start/horizon/availability")

    G = np.zeros((N, P, P))
    b = np.zeros((N, P))
    n_pairs = 0
    pair_ptr = horizon  # next candidate pair origin tau

    # Masked-solve scaffolding: excluded coefficients are pinned to zero by
    # replacing their Gram rows/cols with an identity block and zero rhs.
    pin = np.zeros((len(keys), P, P))
    for m_i in range(len(keys)):
        off = ~masks[m_i]
        pin[m_i][np.ix_(off, off)] = np.eye(int(off.sum()))

    preds = np.full((len(keys), len(origins), N), np.nan)
    actual = np.full((len(origins), N), np.nan)
    zero_i = keys.index(ZERO_KEY)

    def design_rows(t: int) -> np.ndarray:
        d = np.empty((N, P))
        d[:, 0] = 1.0
        d[:, 1] = y[t]
        if controls is not None:
            d[:, 2 : 2 + n_ctrl] = controls[t]
        d[:, state_off:] = state[t][None, :]
        return d

    mask_f = masks.astype(float)  # (M, P)
    for out_i, t in enumerate(origins):
        # Absorb every training pair whose target y[tau + horizon] is realized by t.
        while pair_ptr + horizon <= t:
            tau = pair_ptr
            pair_ptr += 1
            if not available[tau] or not np.isfinite(y[tau]).all():
                continue
            d = design_rows(tau)
            G += d[:, :, None] * d[:, None, :]
            b += d * y[tau + horizon][:, None]
            n_pairs += 1

        actual[out_i] = y[t + horizon]
        if n_pairs < min_train:
            continue
        d_t = design_rows(t)
        # (M, N, P, P) masked Grams solved in one batched call.
        Gm = G[None] * (mask_f[:, None, :, None] * mask_f[:, None, None, :]) + pin[:, None]
        bm = b[None] * mask_f[:, None, :]
        try:
            coef = np.linalg.solve(Gm, bm[..., None])[..., 0]  # (M, N, P)
        except np.linalg.LinAlgError:
            coef = np.stack(
                [
                    np.stack(
                        [np.linalg.lstsq(Gm[m, i], bm[m, i], rcond=None)[0] for i in range(N)]
                    )
                    for m in range(len(keys))
                ]
            )
        preds[:, out_i] = np.einsum("mnp,np->mn", coef * mask_f[:, None, :], d_t)
        preds[zero_i, out_i] = 0.0

    fitted = np.isfinite(preds[keys.index(MEAN_KEY)]).all(axis=1)
    return SubsetForecasts(
        keys=keys,
        origins=origins[fitted],
        targets=origins[fitted] + horizon,
        preds=preds[:, fitted],
        actual=actual[fitted],
        horizon=horizon,
    )


def r2_oos(fr: SubsetForecasts, key, bench, weights: np.ndarray | None = None) -> float:
    """Out-of-sample R^2 of ``key`` against ``bench`` (pooled)."""
    return 1.0 - fr.mse(key, weights) / fr.mse(bench, weights)


def r2_oos_per_commodity(fr: SubsetForecasts, key, bench) -> np.ndarray:
    """(N,) per-commodity out-of-sample R^2 of ``key`` against ``bench``."""
    return 1.0 - fr.sqerr(key).mean(axis=0) / fr.sqerr(bench).mean(axis=0)


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett) standard error of the mean of ``x``."""
    x = np.asarray(x, float)
    x = x - x.mean()
    T = len(x)
    v = float(x @ x) / T
    for lag in range(1, min(lags, T - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        v += 2.0 * w * float(x[:-lag] @ x[lag:]) / T
    return float(np.sqrt(max(v, 0.0) / T))


def default_hac_lags(horizon: int) -> int:
    """Conservative Bartlett bandwidth: covers the h-1 target overlap plus slack."""
    return max(2 * horizon, 5)


def clark_west(
    fr: SubsetForecasts,
    key,
    bench,
    hac_lags: int | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Clark-West MSPE-adjusted test of ``key`` vs the nested ``bench``.

    Pools the adjusted differential across commodities per origin FIRST (the
    time series of cross-sectional means is robust to arbitrary cross-sectional
    error correlation), then a one-sided NW t-test over origins. Returns
    (stat, p).
    """
    e_b = fr.actual - fr.preds[fr.key_index(bench)]
    e_m = fr.actual - fr.preds[fr.key_index(key)]
    adj = e_b**2 - e_m**2 + (fr.preds[fr.key_index(bench)] - fr.preds[fr.key_index(key)]) ** 2
    if weights is not None:
        adj = adj * weights[None, :]
    f_t = adj.mean(axis=1)
    se = newey_west_se(f_t, hac_lags if hac_lags is not None else default_hac_lags(fr.horizon))
    if se == 0:
        return 0.0, 1.0
    stat = float(f_t.mean() / se)
    return stat, float(1.0 - stats.norm.cdf(stat))


def clark_west_per_commodity(
    fr: SubsetForecasts, key, bench, hac_lags: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(N,) Clark-West stats and one-sided p-values, one commodity at a time."""
    if hac_lags is None:
        hac_lags = default_hac_lags(fr.horizon)
    e_b = fr.actual - fr.preds[fr.key_index(bench)]
    e_m = fr.actual - fr.preds[fr.key_index(key)]
    d_pred = fr.preds[fr.key_index(bench)] - fr.preds[fr.key_index(key)]
    adj = e_b**2 - e_m**2 + d_pred**2
    stats_, ps = np.zeros(adj.shape[1]), np.ones(adj.shape[1])
    for i in range(adj.shape[1]):
        se = newey_west_se(adj[:, i], hac_lags)
        if se > 0:
            stats_[i] = adj[:, i].mean() / se
            ps[i] = 1.0 - stats.norm.cdf(stats_[i])
    return stats_, ps


def bh_fdr(pvals: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg rejection mask at FDR level ``q``."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    out = np.zeros(n, dtype=bool)
    if passed.any():
        cut = int(np.max(np.nonzero(passed)[0]))
        out[order[: cut + 1]] = True
    return out


def timing_utility_gain(
    fr: SubsetForecasts,
    key,
    bench,
    vol: np.ndarray,
    gamma: float = 3.0,
    w_cap: float = 1.5,
) -> tuple[float, float]:
    """Campbell-Thompson-style economic value of ``key`` vs ``bench`` forecasts.

    A mean-variance investor sizes each commodity position w = mu_hat/(gamma
    sigma^2), capped at |w| <= w_cap, using each model's forecast and the
    day-t EWMA vol (scaled to the horizon). Returns (annualized utility gain,
    annualized timing-strategy Sharpe difference), averaged across commodities.
    """
    sig2 = np.clip(vol[fr.origins] ** 2 * fr.horizon, 1e-12, None)  # (T_oos, N)
    rets = []
    for kk in (key, bench):
        w = np.clip(fr.preds[fr.key_index(kk)] / (gamma * sig2), -w_cap, w_cap)
        rets.append(w * fr.actual)
    ann = 252.0 / fr.horizon
    utils = [ann * (r.mean() - 0.5 * gamma * r.var()) for r in rets]
    sharpes = [
        float(r.mean(axis=1).mean() / r.mean(axis=1).std() * np.sqrt(ann))
        if r.mean(axis=1).std() > 0
        else 0.0
        for r in rets
    ]
    return float(utils[0] - utils[1]), float(sharpes[0] - sharpes[1])
