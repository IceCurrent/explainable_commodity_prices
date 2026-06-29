"""Synthetic validation of the AE-factor mapping infrastructure.

These tests use controlled synthetic data with a known ground truth so each
component can be checked against an exact answer:

* alignment must recover an injected permutation + sign per window;
* forecast-Shapley must satisfy the efficiency identity sum phi_k = v(full);
* the decoder Jacobian must recover the true loading matrix after alignment;
* the linear spanning machinery must report R^2 ~ 1 / residual ~ 0 when factors
  are an exact linear function of macro, and a positive nonlinearity premium when
  they are a nonlinear function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import ReturnPanel
from src.factor_extraction import FactorBundle


def build_synthetic_bundle(
    seed: int = 0,
    n_windows: int = 8,
    M: int = 200,
    K: int = 4,
    N: int = 10,
    stride: int = 1,
):
    """Synthetic bundle: each window is a permuted/sign-flipped view of shared truth.

    Returns (bundle, true_perm, true_sign, Z_global, D_true).
    """
    rng = np.random.default_rng(seed)
    T_global = M + stride * (n_windows - 1)
    Z = rng.standard_normal((T_global, K))          # independent true factors
    D_true = rng.standard_normal((N, K))            # true decoder loadings

    factors = np.empty((n_windows, M, K), dtype=np.float32)
    decoder = np.empty((n_windows, N, K), dtype=np.float32)
    perms = np.empty((n_windows, K), dtype=int)
    signs = np.empty((n_windows, K), dtype=float)

    for w in range(n_windows):
        if w == 0:
            perm = np.arange(K)        # reference = true identity
            sign = np.ones(K)
        else:
            perm = rng.permutation(K)
            sign = rng.choice([-1.0, 1.0], size=K)
        perms[w] = perm
        signs[w] = sign
        z_win = Z[w * stride : w * stride + M]
        # raw[:, perm[r]] = sign[r] * truth[:, r]
        raw = np.empty((M, K))
        draw = np.empty((N, K))
        for r in range(K):
            raw[:, perm[r]] = sign[r] * z_win[:, r]
            draw[:, perm[r]] = sign[r] * D_true[:, r]
        factors[w] = raw
        decoder[w] = draw

    dates = pd.bdate_range("2010-01-01", periods=T_global)
    start_idx = np.array([w * stride for w in range(n_windows)])
    end_idx = start_idx + M
    bundle = FactorBundle(
        commodities=[f"c{i}" for i in range(N)],
        forecast_dates=pd.DatetimeIndex([dates[min(e, T_global - 1)] for e in end_idx]),
        as_of_dates=pd.DatetimeIndex([dates[e - 1] for e in end_idx]),
        window_start_idx=start_idx,
        window_end_idx=end_idx,
        factors=factors,
        decoder_weight=decoder,
        decoder_bias=np.zeros((n_windows, N), dtype=np.float32),
        config={"stride": stride, "window_size": M, "n_factors": K},
    )
    return bundle, perms, signs, Z, D_true


def test_alignment_recovers_permutation_and_sign():
    from src.factor_alignment import align_by_correlation, align_by_decoder_cosine, stability_report

    bundle, true_perm, true_sign, Z, _ = build_synthetic_bundle(stride=5)
    corr = align_by_correlation(bundle, reference=0)
    cos = align_by_decoder_cosine(bundle, reference=0)

    np.testing.assert_array_equal(corr.perm, true_perm)
    np.testing.assert_array_equal(corr.sign, true_sign)
    np.testing.assert_array_equal(cos.perm, true_perm)
    np.testing.assert_array_equal(cos.sign, true_sign)

    report = stability_report(bundle, corr, cos)
    assert report["median_adjacent_abs_corr"] > 0.99
    assert report["verdict"] == "stable"
    assert report["anchor_agreement_rate"] == 1.0


def test_aligned_live_factors_match_truth():
    from src.factor_alignment import align_by_correlation, apply_alignment_live

    bundle, _, _, Z, _ = build_synthetic_bundle(stride=5)
    corr = align_by_correlation(bundle, reference=0)
    aligned = apply_alignment_live(bundle, corr)
    # live truth = last in-window row of Z per window
    truth = np.array([Z[int(e) - 1] for e in bundle.window_end_idx])
    np.testing.assert_allclose(aligned, truth, atol=1e-4)


def test_jacobian_recovers_true_loadings():
    from src.factor_alignment import align_by_correlation
    from src.jacobian import aligned_mean_jacobian

    bundle, _, _, _, D_true = build_synthetic_bundle(stride=5)
    corr = align_by_correlation(bundle, reference=0)
    J = aligned_mean_jacobian(bundle, corr)
    np.testing.assert_allclose(J, D_true, atol=1e-4)


def test_shapley_efficiency_identity():
    from src.factor_alignment import align_by_correlation
    from src.forecast_shapley import compute_forecast_shapley

    bundle, _, _, _, _ = build_synthetic_bundle(seed=1, n_windows=6, M=150, K=3, N=4, stride=3)
    # Build a return panel consistent with the bundle's window indices.
    rng = np.random.default_rng(2)
    T = int(bundle.window_end_idx.max()) + 1
    raw = rng.standard_normal((T, bundle.n_assets)) * 0.01
    panel = ReturnPanel(
        dates=pd.bdate_range("2010-01-01", periods=T),
        commodities=bundle.commodities,
        raw=raw,
        standardized=raw,
    )
    corr = align_by_correlation(bundle, reference=0)
    res = compute_forecast_shapley(bundle, corr, panel=panel, verbose=False)
    # Efficiency: sum_k phi_k == v(full) == MSE(AR1) - MSE(full)
    assert res.phi_aggregate.sum() == pytest.approx(
        res.mse_ar1_aggregate - res.mse_full_aggregate, rel=1e-6, abs=1e-12
    )
    # Per-commodity efficiency too.
    lhs = res.phi_per_commodity.sum(axis=1)
    rhs = res.mse_ar1 - res.mse_full
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_linear_spanning_is_exact():
    from src.macro_mapping import bai_ng_spanning_summary, decompose_factors, spanning_regression

    rng = np.random.default_rng(3)
    n, J, K = 600, 5, 3
    dates = pd.bdate_range("2012-01-01", periods=n)
    macro = pd.DataFrame(rng.standard_normal((n, J)), index=dates,
                         columns=[f"m{j}" for j in range(J)])
    B = rng.standard_normal((J, K))
    factors = pd.DataFrame(macro.to_numpy() @ B, index=dates,
                           columns=[f"f{k+1}" for k in range(K)])

    reg = spanning_regression(factors, macro)
    for r in reg.values():
        assert r.r2 > 0.999

    # Unregularized projection must reproduce an exactly-linear factor.
    decomp = decompose_factors(factors, macro, ridge_alpha=0.0)
    assert np.abs(decomp.residual.to_numpy()).max() < 1e-6

    bn = bai_ng_spanning_summary(factors, macro)
    assert bn["min_canonical_corr"] > 0.999


def test_nonlinearity_premium_positive_for_nonlinear_factor():
    from src.macro_mapping import nonlinear_macro_mapping

    rng = np.random.default_rng(4)
    n, J = 800, 4
    dates = pd.bdate_range("2012-01-01", periods=n)
    macro = pd.DataFrame(rng.standard_normal((n, J)), index=dates,
                         columns=[f"m{j}" for j in range(J)])
    # Strongly nonlinear, low-noise factor.
    f = np.sin(3 * macro["m0"].to_numpy()) * (macro["m1"].to_numpy() ** 2)
    factors = pd.DataFrame({"f1": f}, index=dates)
    res = nonlinear_macro_mapping(factors, macro, n_splits=4, embargo=10)
    assert res["f1"].nonlinearity_premium > 0.05
