"""Forecast-based Shapley values (PBSV) over factor-state directions.

The players are directions of the latent state (canonical-variate basis for
the headline attribution; raw AE coordinates only as a basis-dependence
diagnostic). The value of a coalition S is its out-of-sample MSE reduction
relative to the AR(1)(+controls) baseline:

    v(S) = MSE_oos(AR1) - MSE_oos(AR1 + state_S),   v(empty) = 0,

so exact Shapley values (2^K enumeration, no sampling) satisfy efficiency:
sum_k phi_k = v(full). A positive phi_k means direction k reduces OOS error.

Grouped attribution: v(S) for a union of whole groups depends only on the SPAN
of the included directions, so grouped values are invariant to rotations
inside a group — the correct quotient when trailing canonical correlations
are near-tied. Note grouped Shapley is NOT the within-block sum of the
per-direction values (Shapley lacks group consistency); the full coalition
table is exported so readers can see the interaction term.

Calibration: every marginal contribution pays a mechanical estimation-cost
penalty (~sigma^2 * ln(T/T_train)/T_oos per regressor) even under zero signal,
so small negative phi is expected noise, not harm. ``placebo_pbsv`` reruns the
identical machinery on jointly circular-shifted state to produce a
cardinality-matched zero-signal band; phi inside the band must not be
interpreted. Bootstrap CIs are descriptive and conditional on the frozen basis
and the realized expanding-parameter path — significance claims belong to the
Clark-West contrasts, not these intervals.
"""

from __future__ import annotations

from itertools import chain, combinations
from math import factorial

import numpy as np

from eqcp.cca.inference import stationary_bootstrap_idx
from eqcp.forecasting.ar1 import (
    AR_KEY,
    BENCH_KEYS,
    SubsetForecasts,
    all_subsets,
    expanding_subset_forecasts,
)


def exact_shapley(values: dict[tuple[int, ...], float], players: list[int]) -> np.ndarray:
    """Exact Shapley values from a complete coalition->value table.

    ``values`` must contain every subset of ``players`` as a sorted tuple key.
    """
    k = len(players)
    phi = np.zeros(k)
    for i, p in enumerate(players):
        others = [q for q in players if q != p]
        for r in range(k):
            w = factorial(r) * factorial(k - r - 1) / factorial(k)
            for combo in combinations(others, r):
                s = tuple(sorted(combo))
                s_p = tuple(sorted(combo + (p,)))
                phi[i] += w * (values[s_p] - values[s])
    return phi


def _subset_keys(fr: SubsetForecasts) -> list[tuple[int, ...]]:
    return [k for k in fr.keys if k not in BENCH_KEYS]


def _n_dims(subset_keys: list[tuple[int, ...]]) -> int:
    return max((max(s) for s in subset_keys if s), default=-1) + 1


def subset_values_from_forecasts(
    fr: SubsetForecasts, weights: np.ndarray | None = None
) -> dict[tuple[int, ...], float]:
    """v(S) = MSE(AR1) - MSE(S) for every factor subset present in ``fr``."""
    mse_ar = fr.mse(AR_KEY, weights)
    return {key: mse_ar - fr.mse(key, weights) for key in _subset_keys(fr)}


def grouped_values(
    values: dict[tuple[int, ...], float], groups: list[tuple[int, ...]]
) -> dict[tuple[int, ...], float]:
    """Coalition table over group indices, mapping each group union to v(union)."""
    out: dict[tuple[int, ...], float] = {}
    n_groups = len(groups)
    for gs in chain.from_iterable(combinations(range(n_groups), r) for r in range(n_groups + 1)):
        union = tuple(sorted(chain.from_iterable(groups[g] for g in gs)))
        out[tuple(gs)] = values[union]
    return out


def pbsv(
    fr: SubsetForecasts,
    weights: np.ndarray | None = None,
    groups: list[tuple[int, ...]] | None = None,
) -> dict:
    """Point estimates: per-direction phi, grouped phi, and the value table."""
    values = subset_values_from_forecasts(fr, weights)
    players = list(range(_n_dims(list(values))))
    phi = exact_shapley(values, players)
    out = {"values": values, "phi": phi, "v_full": values[tuple(players)]}
    if groups is not None:
        gv = grouped_values(values, groups)
        out["phi_grouped"] = exact_shapley(gv, list(range(len(groups))))
        out["group_table"] = gv
    return out


def pbsv_per_commodity(fr: SubsetForecasts) -> np.ndarray:
    """(N, K) per-commodity Shapley values (raw squared-error units)."""
    subset_keys = _subset_keys(fr)
    players = list(range(_n_dims(subset_keys)))
    n = fr.actual.shape[1]
    mse_ar = fr.sqerr(AR_KEY).mean(axis=0)
    phis = np.zeros((n, len(players)))
    for i in range(n):
        values = {key: float(mse_ar[i] - fr.sqerr(key)[:, i].mean()) for key in subset_keys}
        phis[i] = exact_shapley(values, players)
    return phis


def pbsv_on_mask(
    fr: SubsetForecasts, day_mask: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """phi recomputed on a subset of OOS origins (per-year / burn-in robustness)."""
    subset_keys = _subset_keys(fr)
    players = list(range(_n_dims(subset_keys)))
    losses = {k: fr.per_day_loss(k, weights)[day_mask].mean() for k in subset_keys}
    values = {k: losses[AR_KEY] - losses[k] for k in subset_keys}
    return exact_shapley(values, players)


def bootstrap_pbsv(
    fr: SubsetForecasts,
    weights: np.ndarray | None = None,
    groups: list[tuple[int, ...]] | None = None,
    mean_block: int = 21,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Stationary-bootstrap draws of phi (and grouped phi) over OOS origins.

    The SAME day blocks are resampled jointly across all subsets, preserving
    the cross-subset covariance of per-day losses (phi is a linear functional
    of the 2^K-vector of subset means).
    """
    subset_keys = _subset_keys(fr)
    players = list(range(_n_dims(subset_keys)))
    losses = np.stack([fr.per_day_loss(k, weights) for k in subset_keys])  # (S, T)
    ar_row = subset_keys.index(AR_KEY)
    t_oos = losses.shape[1]
    rng = np.random.default_rng(seed)

    draws = np.zeros((n_boot, len(players)))
    g_draws = np.zeros((n_boot, len(groups))) if groups is not None else None
    for bi in range(n_boot):
        idx = stationary_bootstrap_idx(t_oos, mean_block, rng)
        m = losses[:, idx].mean(axis=1)
        values = {key: float(m[ar_row] - m[si]) for si, key in enumerate(subset_keys)}
        draws[bi] = exact_shapley(values, players)
        if g_draws is not None and groups is not None:
            g_draws[bi] = exact_shapley(grouped_values(values, groups), list(range(len(groups))))

    out = {
        "phi_draws": draws,
        "phi_lo": np.percentile(draws, 2.5, axis=0),
        "phi_hi": np.percentile(draws, 97.5, axis=0),
    }
    if g_draws is not None:
        out["phi_grouped_draws"] = g_draws
        out["phi_grouped_lo"] = np.percentile(g_draws, 2.5, axis=0)
        out["phi_grouped_hi"] = np.percentile(g_draws, 97.5, axis=0)
    return out


def placebo_pbsv(
    returns: np.ndarray,
    state: np.ndarray,
    oos_start: int,
    horizon: int,
    weights: np.ndarray | None = None,
    groups: list[tuple[int, ...]] | None = None,
    controls: np.ndarray | None = None,
    available: np.ndarray | None = None,
    min_train: int = 60,
    n_placebo: int = 100,
    min_shift: int = 63,
    seed: int = 0,
    stat_fn=None,
) -> dict:
    """Cardinality-matched zero-signal phi distribution via joint circular shifts.

    The whole state block is rolled by one common tau per draw (|tau| >=
    ``min_shift`` from both calendar ends), preserving each variate's
    autocorrelation and the block's covariance while destroying alignment with
    the targets. Each draw pays the same per-regressor estimation cost as the
    real game, so the resulting per-phi bands calibrate how much |phi| pure
    overfitting produces at matched subset cardinality.
    """
    T = state.shape[0]
    rng = np.random.default_rng(seed)
    k = state.shape[1]
    draws = np.zeros((n_placebo, k))
    g_draws = np.zeros((n_placebo, len(groups))) if groups is not None else None
    stat_draws: list[float] = []
    for b in range(n_placebo):
        tau = int(rng.integers(min_shift, T - min_shift))
        fr_b = expanding_subset_forecasts(
            returns,
            np.roll(state, tau, axis=0),
            oos_start,
            horizon=horizon,
            controls=controls,
            available=available,
            min_train=min_train,
        )
        res = pbsv(fr_b, weights=weights, groups=groups)
        draws[b] = res["phi"]
        if g_draws is not None:
            g_draws[b] = res["phi_grouped"]
        if stat_fn is not None:
            stat_draws.append(float(stat_fn(fr_b)))
    out = {
        "phi_draws": draws,
        "phi_lo": np.percentile(draws, 2.5, axis=0),
        "phi_hi": np.percentile(draws, 97.5, axis=0),
        "abs_p95": np.percentile(np.abs(draws), 95, axis=0),
    }
    if stat_fn is not None:
        out["stat_draws"] = np.asarray(stat_draws)
    if g_draws is not None:
        out["phi_grouped_draws"] = g_draws
        out["phi_grouped_lo"] = np.percentile(g_draws, 2.5, axis=0)
        out["phi_grouped_hi"] = np.percentile(g_draws, 97.5, axis=0)
        out["grouped_abs_p95"] = np.percentile(np.abs(g_draws), 95, axis=0)
    return out


def boundary_sensitivity(values: dict[tuple[int, ...], float], k: int) -> list[dict]:
    """Grouped 2-player PBSV for every adjacent spanned|orphan boundary."""
    rows = []
    for cut in range(1, k):
        groups = [tuple(range(cut)), tuple(range(cut, k))]
        phi_g = exact_shapley(grouped_values(values, groups), [0, 1])
        rows.append(
            {
                "boundary": f"{'.'.join(str(i + 1) for i in groups[0])}|"
                f"{'.'.join(str(i + 1) for i in groups[1])}",
                "phi_lead_block": float(phi_g[0]),
                "phi_tail_block": float(phi_g[1]),
                "v_lead": values[groups[0]],
                "v_tail": values[groups[1]],
                "interaction": values[tuple(range(k))] - values[groups[0]] - values[groups[1]],
            }
        )
    return rows


def efficiency_gap(result: dict) -> float:
    """|sum(phi) - v(full)|; exact Shapley must satisfy this to numerical precision."""
    return float(abs(result["phi"].sum() - result["v_full"]))


__all__ = [
    "AR_KEY",
    "all_subsets",
    "exact_shapley",
    "subset_values_from_forecasts",
    "grouped_values",
    "pbsv",
    "pbsv_per_commodity",
    "pbsv_on_mask",
    "bootstrap_pbsv",
    "placebo_pbsv",
    "boundary_sensitivity",
    "efficiency_gap",
]
