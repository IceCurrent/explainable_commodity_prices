"""Synthetic validation of linear / nonlinear macro spanning machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqcp.cca.linear import align_xy, bai_ng_spanning_summary
from eqcp.spanning.regression import nonlinear_macro_mapping, spanning_regression


def _macro_projection_residual(factors: pd.DataFrame, macro: pd.DataFrame) -> np.ndarray:
    """Unregularized macro projection residual (ridge_alpha=0 analogue)."""
    f, m = align_xy(factors, macro)
    mu = m.mean()
    sd = m.std(ddof=0).replace(0.0, 1.0)
    z = ((m - mu) / sd).to_numpy()
    zc = np.column_stack([np.ones(len(z)), z])
    coef = np.linalg.lstsq(zc, f.to_numpy(), rcond=None)[0]
    return f.to_numpy() - zc @ coef


def test_linear_spanning_is_exact():
    rng = np.random.default_rng(3)
    n, j, k = 600, 5, 3
    dates = pd.bdate_range("2012-01-01", periods=n)
    macro = pd.DataFrame(
        rng.standard_normal((n, j)), index=dates, columns=[f"m{j_}" for j_ in range(j)]
    )
    b = rng.standard_normal((j, k))
    factors = pd.DataFrame(
        macro.to_numpy() @ b, index=dates, columns=[f"f{k_ + 1}" for k_ in range(k)]
    )

    reg = spanning_regression(factors, macro)
    for r in reg.values():
        assert r.r2 > 0.999

    residual = _macro_projection_residual(factors, macro)
    assert np.abs(residual).max() < 1e-6

    bn = bai_ng_spanning_summary(factors, macro)
    assert bn["min_canonical_corr"] > 0.999


def test_nonlinearity_premium_positive_for_nonlinear_factor():
    rng = np.random.default_rng(4)
    n, j = 800, 4
    dates = pd.bdate_range("2012-01-01", periods=n)
    macro = pd.DataFrame(
        rng.standard_normal((n, j)), index=dates, columns=[f"m{j_}" for j_ in range(j)]
    )
    f = np.sin(3 * macro["m0"].to_numpy()) * (macro["m1"].to_numpy() ** 2)
    factors = pd.DataFrame({"f1": f}, index=dates)
    res = nonlinear_macro_mapping(factors, macro, n_splits=4, embargo=10)
    assert res["f1"].nonlinearity_premium > 0.05
