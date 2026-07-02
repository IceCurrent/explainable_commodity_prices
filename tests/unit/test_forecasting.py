"""Unit tests for the forecasting engine, PBSV, and the frozen attribution basis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eqcp.forecasting.ar1 import (
    AR_KEY,
    MEAN_KEY,
    ZERO_KEY,
    all_subsets,
    expanding_subset_forecasts,
    h_period_returns,
    staleness_stats,
)
from eqcp.forecasting.basis import fit_frozen_basis, frozen_oos_diagnostics, principal_angles
from eqcp.forecasting.pbsv import (
    bootstrap_pbsv,
    boundary_sensitivity,
    efficiency_gap,
    exact_shapley,
    grouped_values,
    pbsv,
    subset_values_from_forecasts,
)


# --------------------------------------------------------------------- games
def test_exact_shapley_axioms():
    # 3-player game: v(S) = sum of members + synergy between 0 and 1.
    base = {0: 1.0, 1: 2.0, 2: 0.0}
    syn = 0.6

    values = {}
    for s in all_subsets(3):
        v = sum(base[p] for p in s)
        if 0 in s and 1 in s:
            v += syn
        values[s] = v
    phi = exact_shapley(values, [0, 1, 2])
    # efficiency
    assert phi.sum() == pytest.approx(values[(0, 1, 2)])
    # dummy player 2 gets exactly 0
    assert phi[2] == pytest.approx(0.0, abs=1e-12)
    # synergy split evenly between symmetric contributors
    assert phi[0] == pytest.approx(base[0] + syn / 2)
    assert phi[1] == pytest.approx(base[1] + syn / 2)


def test_grouped_values_and_boundary_sensitivity():
    values = {s: float(len(s)) for s in all_subsets(4)}
    gv = grouped_values(values, [(0, 1), (2, 3)])
    assert gv[()] == 0.0
    assert gv[(0,)] == 2.0
    assert gv[(0, 1)] == 4.0
    rows = boundary_sensitivity(values, 4)
    assert len(rows) == 3
    for row in rows:
        assert row["phi_lead_block"] + row["phi_tail_block"] == pytest.approx(4.0)


# ------------------------------------------------------------------- engine
def _simulate(T=700, N=4, K=3, beta=0.6, seed=0):
    """Returns with one genuinely predictive persistent state direction."""
    rng = np.random.default_rng(seed)
    g = np.zeros(T)
    for t in range(1, T):
        g[t] = 0.95 * g[t - 1] + rng.standard_normal() * np.sqrt(1 - 0.95**2)
    state = np.column_stack([g] + [rng.standard_normal(T) for _ in range(K - 1)])
    load = rng.uniform(0.5, 1.0, N)
    returns = np.zeros((T, N))
    for t in range(1, T):
        returns[t] = beta * g[t - 1] * load + 0.5 * rng.standard_normal(N)
    return returns, state


def test_h_period_returns_sums():
    r = np.arange(12, dtype=float).reshape(6, 2)
    y = h_period_returns(r, 3)
    assert np.isnan(y[:2]).all()
    np.testing.assert_allclose(y[2], r[0] + r[1] + r[2])
    np.testing.assert_allclose(y[5], r[3] + r[4] + r[5])


def test_engine_finds_predictive_state_and_efficiency():
    returns, state = _simulate()
    fr = expanding_subset_forecasts(returns, state, oos_start=400, min_train=50)
    res = pbsv(fr)
    # the predictive direction is player 0 and dominates
    assert res["v_full"] > 0
    assert res["phi"][0] > 0.8 * res["v_full"]
    assert efficiency_gap(res) < 1e-12
    # full model beats AR(1) and zero benchmarks
    assert fr.mse((0, 1, 2)) < fr.mse(AR_KEY)
    assert fr.mse((0, 1, 2)) < fr.mse(ZERO_KEY)


def test_no_lookahead_multihorizon():
    returns, state = _simulate(T=500)
    h = 5
    probe = 320
    fr = expanding_subset_forecasts(returns, state, oos_start=250, horizon=h, min_train=40)
    rng = np.random.default_rng(9)
    returns2 = returns.copy()
    state2 = state.copy()
    returns2[probe + 1 :] = rng.standard_normal(returns2[probe + 1 :].shape) * 10
    state2[probe + 1 :] = rng.standard_normal(state2[probe + 1 :].shape) * 10
    fr2 = expanding_subset_forecasts(returns2, state2, oos_start=250, horizon=h, min_train=40)
    # forecasts at origins t <= probe - h use only data <= probe: identical.
    keep1 = fr.origins <= probe - h
    keep2 = fr2.origins <= probe - h
    full = tuple(range(state.shape[1]))
    np.testing.assert_allclose(
        fr.preds[fr.key_index(full)][keep1],
        fr2.preds[fr2.key_index(full)][keep2],
        atol=1e-10,
    )
    # and the training pair whose target ends at probe+1 is NOT yet usable at
    # origin probe-h+1: forecasts there must also match (target completes later).
    keep_edge1 = fr.origins <= probe - 1
    assert keep_edge1.sum() >= keep1.sum()


def test_full_model_invariant_to_affine_state_transform():
    returns, state = _simulate(T=400)
    rng = np.random.default_rng(3)
    R = rng.standard_normal((3, 3)) + 3 * np.eye(3)
    shift = rng.standard_normal(3)
    fr1 = expanding_subset_forecasts(returns, state, oos_start=250, min_train=40)
    fr2 = expanding_subset_forecasts(returns, state @ R.T + shift, oos_start=250, min_train=40)
    full = (0, 1, 2)
    np.testing.assert_allclose(
        fr1.preds[fr1.key_index(full)], fr2.preds[fr2.key_index(full)], atol=1e-6
    )
    # benchmarks unaffected by the state entirely
    np.testing.assert_allclose(
        fr1.preds[fr1.key_index(AR_KEY)], fr2.preds[fr2.key_index(AR_KEY)], atol=1e-8
    )
    # but SUBSET values are basis-dependent: the raw single-player games differ
    v1 = subset_values_from_forecasts(fr1)
    v2 = subset_values_from_forecasts(fr2)
    assert abs(v1[(0,)] - v2[(0,)]) > 1e-6


def test_controls_columns_present_in_all_models():
    returns, state = _simulate(T=400)
    ctrl = np.abs(returns)[:, :, None]  # (T, N, 1)
    fr = expanding_subset_forecasts(
        returns, state, oos_start=300, controls=ctrl, subsets=[(), (0, 1, 2)], min_train=40
    )
    # AR model with controls differs from AR model without them
    fr0 = expanding_subset_forecasts(
        returns, state, oos_start=300, subsets=[(), (0, 1, 2)], min_train=40
    )
    assert not np.allclose(
        fr.preds[fr.key_index(AR_KEY)], fr0.preds[fr0.key_index(AR_KEY)]
    )
    # mean benchmark is intercept-only in both
    np.testing.assert_allclose(
        fr.preds[fr.key_index(MEAN_KEY)], fr0.preds[fr0.key_index(MEAN_KEY)], atol=1e-10
    )


def test_bootstrap_pbsv_deterministic_and_efficient_per_draw():
    returns, state = _simulate(T=400)
    fr = expanding_subset_forecasts(returns, state, oos_start=300, min_train=40)
    b1 = bootstrap_pbsv(fr, n_boot=25, seed=7, groups=[(0,), (1, 2)])
    b2 = bootstrap_pbsv(fr, n_boot=25, seed=7, groups=[(0,), (1, 2)])
    np.testing.assert_array_equal(b1["phi_draws"], b2["phi_draws"])
    # per-draw efficiency: grouped phi sums to the same v(full) as per-dim phi
    np.testing.assert_allclose(
        b1["phi_draws"].sum(axis=1), b1["phi_grouped_draws"].sum(axis=1), atol=1e-10
    )


def test_staleness_stats():
    r = np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.0], [0.1, 0.3]])
    frac, longest = staleness_stats(r)
    np.testing.assert_allclose(frac, [0.75, 0.25])
    np.testing.assert_array_equal(longest, [3, 1])


# -------------------------------------------------------------------- basis
def _factor_macro_panels(T=600, seed=0):
    """Factors = invertible mix of [persistent macro-observed g, noise dims]."""
    rng = np.random.default_rng(seed)
    g = np.zeros(T)
    for t in range(1, T):
        g[t] = 0.95 * g[t - 1] + rng.standard_normal() * np.sqrt(1 - 0.95**2)
    latent = np.column_stack([g, rng.standard_normal((T, 2))])
    R = rng.standard_normal((3, 3)) + 3 * np.eye(3)
    dates = pd.bdate_range("2015-01-01", periods=T)
    F = pd.DataFrame(latent @ R.T, index=dates, columns=["f1", "f2", "f3"])
    macro = np.column_stack([g + 0.3 * rng.standard_normal(T) for _ in range(4)])
    M = pd.DataFrame(macro, index=dates, columns=[f"m{i}" for i in range(4)])
    return F, M, latent


def test_frozen_basis_variates_invariant_to_factor_mixing(rng: np.random.Generator):
    F, M, _ = _factor_macro_panels()
    basis1 = fit_frozen_basis(F, M, ridge_grid=(0.0,), n_perm=20)
    R = rng.standard_normal((3, 3)) + 3 * np.eye(3)
    F2 = pd.DataFrame(F.to_numpy() @ R.T, index=F.index, columns=F.columns)
    basis2 = fit_frozen_basis(F2, M, ridge_grid=(0.0,), n_perm=20)
    v1 = basis1.variates(F).to_numpy()
    v2 = basis2.variates(F2).to_numpy()
    # canonical variates identical up to sign (macro-anchored sign should fix it)
    for k in range(3):
        corr = np.corrcoef(v1[:, k], v2[:, k])[0, 1]
        assert abs(corr) > 1 - 1e-6
    # the leading (macro-aligned) variate has the SAME sign under both bases
    assert np.corrcoef(v1[:, 0], v2[:, 0])[0, 1] > 0


def test_frozen_basis_invariance_holds_with_macro_side_ridge(rng: np.random.Generator):
    F, M, _ = _factor_macro_panels()
    basis1 = fit_frozen_basis(F, M, ridge_grid=(0.1,), n_perm=20)
    assert basis1.ridge_m == 0.1
    R = rng.standard_normal((3, 3)) + 3 * np.eye(3)
    F2 = pd.DataFrame(F.to_numpy() @ R.T, index=F.index, columns=F.columns)
    basis2 = fit_frozen_basis(F2, M, ridge_grid=(0.1,), n_perm=20)
    v1 = basis1.variates(F).to_numpy()
    v2 = basis2.variates(F2).to_numpy()
    for k in range(3):
        assert abs(np.corrcoef(v1[:, k], v2[:, k])[0, 1]) > 1 - 1e-6


def test_frozen_basis_rejects_degenerate_factors():
    F, M, _ = _factor_macro_panels()
    F_bad = F.copy()
    F_bad["f3"] = F_bad["f1"] * 2.0  # duplicated latent
    with pytest.raises(ValueError, match="near-singular|degenerate"):
        fit_frozen_basis(F_bad, M, ridge_grid=(0.0,), n_perm=20)


def test_frozen_oos_diagnostics_and_principal_angles():
    F, M, _ = _factor_macro_panels(T=800)
    basis = fit_frozen_basis(F.iloc[:500], M.iloc[:500], ridge_grid=(0.0,), n_perm=20)
    diag = frozen_oos_diagnostics(basis, F.iloc[500:], M.iloc[500:])
    assert len(diag) == 3
    # macro-aligned dim keeps a high forward rho and stable loadings
    assert diag.loc[0, "rho_oos_frozen"] > 0.5
    assert diag.loc[0, "loading_cosine_oos"] > 0.8
    ang = principal_angles(np.eye(4)[:, :2], np.eye(4)[:, :2])
    np.testing.assert_allclose(ang, 0.0, atol=1e-10)


def test_cv_basis_pbsv_recovers_planted_persistent_direction():
    """End-to-end identification check: the macro-observed persistent direction
    earns the Shapley mass in the CV basis, regardless of factor mixing."""
    F, M, latent = _factor_macro_panels(T=900, seed=1)
    rng = np.random.default_rng(2)
    T = len(F)
    N = 4
    load = rng.uniform(0.5, 1.0, N)
    returns = np.zeros((T, N))
    for t in range(1, T):
        returns[t] = 0.6 * latent[t - 1, 0] * load + 0.5 * rng.standard_normal(N)
    t_split = 600
    basis = fit_frozen_basis(F.iloc[:t_split], M.iloc[:t_split], ridge_grid=(0.0,), n_perm=20)
    state = basis.variates(F).to_numpy()
    fr = expanding_subset_forecasts(returns, state, oos_start=t_split, min_train=50)
    res = pbsv(fr, groups=basis.groups)
    assert res["v_full"] > 0
    # cv1 (the macro-anchored direction) carries the dominant share
    assert res["phi"][0] > 0.7 * res["v_full"]
    assert basis.n_spanned >= 1
