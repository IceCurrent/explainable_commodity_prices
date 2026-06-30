"""Optional end-to-end smoke tests on tiny fixture panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eqcp.cca.inference import circular_perm_null, purged_cv_canon
from eqcp.cca.kernel import kernel_canonical_correlations
from eqcp.cca.linear import bai_ng_spanning_summary, canonical_correlations, linear_cca_full
from eqcp.cca.reduce import bloc_reduce
from eqcp.spanning.regression import spanning_regression


def _load_fixture_pair(fixtures_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = pd.read_csv(fixtures_dir / "tiny_factors.csv", index_col=0, parse_dates=True)
    macro = pd.read_csv(fixtures_dir / "tiny_macro.csv", index_col=0, parse_dates=True)
    return factors, macro


@pytest.mark.slow
def test_fixture_cca_pipeline_smoke(fixtures_dir: Path):
    factors, macro = _load_fixture_pair(fixtures_dir)
    cc = canonical_correlations(factors, macro)
    assert len(cc) == min(factors.shape[1], macro.shape[1])
    assert (cc >= 0).all() and (cc <= 1).all()

    corrs, *_ = linear_cca_full(factors.to_numpy(), macro.to_numpy(), ridge=0.0)
    assert len(corrs) == len(cc)

    bn = bai_ng_spanning_summary(factors, macro)
    assert bn["n_factors"] == factors.shape[1]
    assert bn["n_macro"] == macro.shape[1]

    span = spanning_regression(factors, macro)
    assert len(span) == factors.shape[1]

    bloc_map = {"fx": list(macro.columns[:2]), "rates": list(macro.columns[2:])}
    reduced = bloc_reduce(macro, bloc_map)
    assert reduced.shape[1] == 2

    oos = purged_cv_canon(factors.to_numpy(), macro.to_numpy(), n_folds=3, embargo=3)
    assert len(oos) == min(factors.shape[1], macro.shape[1])

    kcc = kernel_canonical_correlations(factors, macro, top_k=2)
    assert len(kcc) == 2

    null = circular_perm_null(
        lambda f, m: linear_cca_full(f, m, ridge=0.0)[0],
        factors.to_numpy(),
        macro.to_numpy(),
        n_perm=20,
        seed=0,
    )
    assert 0.0 < null["p_min"] <= 1.0
