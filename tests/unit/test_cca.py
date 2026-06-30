"""Unit tests for CCA core, inference, bloc reduction, and macro transforms."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from eqcp.cca.inference import circular_perm_null, purged_cv_canon
from eqcp.cca.linear import canonical_correlations, linear_cca_full
from eqcp.cca.reduce import bloc_reduce
from eqcp.cca.verdict import kcca_nonlinearity_verdict
from eqcp.macro_processing.process_macro import apply_transform


def _synthetic_panels(rng: np.random.Generator, n: int = 200, k: int = 3, j: int = 5):
    dates = pd.bdate_range("2015-01-01", periods=n)
    g = rng.standard_normal((n, k))
    b = rng.standard_normal((j, k))
    factors = pd.DataFrame(g, index=dates, columns=[f"f{i}" for i in range(k)])
    macro = pd.DataFrame(
        g @ b.T + 0.05 * rng.standard_normal((n, j)),
        index=dates,
        columns=[f"m{i}" for i in range(j)],
    )
    return factors, macro


def test_linear_cca_full_matches_canonical_correlations(rng: np.random.Generator):
    factors, macro = _synthetic_panels(rng)
    cc_ref = canonical_correlations(factors, macro)
    cc_full, *_ = linear_cca_full(factors.to_numpy(), macro.to_numpy(), ridge=0.0)
    np.testing.assert_allclose(cc_full, cc_ref, atol=1e-8)


def test_cca_invariant_to_column_scaling(rng: np.random.Generator):
    factors, macro = _synthetic_panels(rng, n=150, k=3, j=4)
    cc0 = canonical_correlations(factors, macro)
    scales = np.array([2.0, 0.5, 3.0])[: factors.shape[1]]
    scaled = factors * scales
    cc1 = canonical_correlations(scaled, macro)
    np.testing.assert_allclose(cc0, cc1, atol=1e-8)


def test_cca_invariant_to_orthogonal_rotation(rng: np.random.Generator):
    factors, macro = _synthetic_panels(rng, n=150, k=3, j=4)
    cc0 = canonical_correlations(factors, macro)
    k = factors.shape[1]
    q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    rotated = pd.DataFrame(factors.to_numpy() @ q, index=factors.index, columns=factors.columns)
    cc1 = canonical_correlations(rotated, macro)
    np.testing.assert_allclose(cc0, cc1, atol=1e-8)


def test_circular_perm_null_deterministic_and_pvalue_formula(rng: np.random.Generator):
    factors, macro = _synthetic_panels(rng, n=120, k=2, j=3)
    f_arr, m_arr = factors.to_numpy(), macro.to_numpy()
    n_perm = 40

    def score_fn(f, m):
        return linear_cca_full(f, m, ridge=0.0)[0]

    r1 = circular_perm_null(score_fn, f_arr, m_arr, n_perm=n_perm, seed=7)
    r2 = circular_perm_null(score_fn, f_arr, m_arr, n_perm=n_perm, seed=7)
    np.testing.assert_array_equal(r1["null_min"], r2["null_min"])
    np.testing.assert_array_equal(r1["null_mean"], r2["null_mean"])
    expected_p_min = (1 + int(np.sum(r1["null_min"] >= r1["obs_min"]))) / (n_perm + 1)
    expected_p_mean = (1 + int(np.sum(r1["null_mean"] >= r1["obs_mean"]))) / (n_perm + 1)
    assert r1["p_min"] == pytest.approx(expected_p_min)
    assert r1["p_mean"] == pytest.approx(expected_p_mean)


def test_purged_cv_canon_folds_respect_embargo(rng: np.random.Generator):
    n, k, j = 120, 2, 3
    embargo = 12
    n_folds = 4
    row_ids = np.arange(n)[:, None]
    f_arr = np.hstack([row_ids, rng.standard_normal((n, k - 1))])
    m_arr = rng.standard_normal((n, j))
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    captured: list[np.ndarray] = []

    def recording_cca(f_train, m_train, ridge=0.0):
        captured.append(f_train[:, 0].copy())
        return linear_cca_full(f_train, m_train, ridge=ridge)

    with patch("eqcp.cca.inference.linear_cca_full", side_effect=recording_cca):
        purged_cv_canon(f_arr, m_arr, n_folds=n_folds, embargo=embargo, ridge=0.0)

    assert len(captured) > 0
    for i, train_ids in enumerate(captured):
        ts, te = int(bounds[i]), int(bounds[i + 1])
        lo, hi = max(ts - embargo, 0), min(te + embargo, n)
        forbidden = set(range(lo, hi))
        assert not forbidden.intersection(set(train_ids.astype(int)))


def test_bloc_reduce_sign_fixing(rng: np.random.Generator):
    n = 60
    base = rng.standard_normal((n, 3))
    # Flip all columns so the raw first PC tends to oppose the row mean.
    m_df = pd.DataFrame(-base, columns=["a", "b", "c"])
    out = bloc_reduce(m_df, {"blk": ["a", "b", "c"]})
    xs = (m_df - m_df.mean()) / m_df.std(ddof=0)
    row_mean = xs.mean(axis=1).to_numpy()
    corr = np.corrcoef(out["blk"].to_numpy(), row_mean)[0, 1]
    assert corr >= 0.0


def test_kcca_nonlinearity_verdict_nonlinear():
    stab = pd.DataFrame({"kcca_min": [0.50, 0.52, 0.51, 0.53]})
    label, expl = kcca_nonlinearity_verdict(stab, canon_min=0.30, kcca_min=0.55)
    assert label == "nonlinear"
    assert "stable" in expl.lower()


def test_kcca_nonlinearity_verdict_inconclusive_when_unstable():
    stab = pd.DataFrame({"kcca_min": [0.30, 0.60, 0.35, 0.70]})
    label, expl = kcca_nonlinearity_verdict(stab, canon_min=0.30, kcca_min=0.55)
    assert label == "inconclusive / degenerate"
    assert "unstable" in expl.lower()


def test_apply_transform_level():
    idx = pd.bdate_range("2020-01-01", periods=5)
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="x")
    out = apply_transform(s, "level")
    pd.testing.assert_series_equal(out, s)


def test_apply_transform_diff():
    idx = pd.bdate_range("2020-01-01", periods=5)
    s = pd.Series([1.0, 2.0, 4.0, 7.0, 11.0], index=idx, name="y")
    out = apply_transform(s, "diff")
    pd.testing.assert_series_equal(out, s.diff())


def test_apply_transform_log_return():
    idx = pd.bdate_range("2020-01-01", periods=5)
    s = pd.Series([100.0, 101.0, 102.0, 104.0, 103.0], index=idx, name="z")
    out = apply_transform(s, "log_return")
    expected = np.log(s).diff()
    pd.testing.assert_series_equal(out, expected)


def test_apply_transform_rejects_non_positive_log_return():
    idx = pd.bdate_range("2020-01-01", periods=3)
    s = pd.Series([1.0, -0.5, 2.0], index=idx, name="bad")
    with pytest.raises(AssertionError):
        apply_transform(s, "log_return")


def test_apply_transform_unknown_kind():
    s = pd.Series([1.0, 2.0], index=pd.bdate_range("2020-01-01", periods=2))
    with pytest.raises(ValueError, match="unknown transform"):
        apply_transform(s, "fft")
