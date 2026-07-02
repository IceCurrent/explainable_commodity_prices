"""Unit tests for panel curation (exclusion) and the sector-wise CCA framework."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqcp.io.commodities import (
    EXCLUDED_COMMODITIES,
    SECTORS,
    load_return_panel,
    sector_of,
)
from eqcp.sectors import _factor_count, spanning_analysis, spanning_summary_table


def test_panel_excludes_poisonous_series() -> None:
    panel = load_return_panel()
    assert panel.n_assets == 21
    for bad in EXCLUDED_COMMODITIES:
        assert bad not in panel.commodities
    # the sector partition covers exactly the retained panel, no overlaps
    covered = [c for members in SECTORS.values() for c in members]
    assert sorted(covered) == sorted(panel.commodities)
    assert len(covered) == len(set(covered))


def test_sector_of() -> None:
    assert sector_of("Brent") == "energy"
    assert sector_of("Corn") == "agriculture"
    assert sector_of("Gold") == "metals"
    assert sector_of("Lithium") is None


def test_load_subset() -> None:
    panel = load_return_panel(commodities=SECTORS["energy"])
    assert panel.n_assets == 5
    assert list(panel.commodities) == SECTORS["energy"]


def test_factor_count_is_a_genuine_bottleneck() -> None:
    assert _factor_count(5, 3) == 3
    assert _factor_count(3, 5) == 2  # never >= number of series
    assert _factor_count(1, 3) == 1  # floored at 1


def test_spanning_analysis_recovers_known_structure() -> None:
    rng = np.random.default_rng(0)
    t = 400
    idx = pd.date_range("2019-01-01", periods=t, freq="B")
    latent = rng.standard_normal((t, 2))
    macro = pd.DataFrame(
        {
            "a1": latent[:, 0] + 0.1 * rng.standard_normal(t),
            "a2": latent[:, 0] + 0.1 * rng.standard_normal(t),
            "b1": latent[:, 1] + 0.1 * rng.standard_normal(t),
            "b2": latent[:, 1] + 0.1 * rng.standard_normal(t),
        },
        index=idx,
    )
    factors = pd.DataFrame(
        {
            "f1": latent[:, 0] + 0.2 * rng.standard_normal(t),
            "f2": rng.standard_normal(t),
            "f3": rng.standard_normal(t),
        },
        index=idx,
    )
    bloc_map = {"A": ["a1", "a2"], "B": ["b1", "b2"]}
    res = spanning_analysis(
        "energy", factors, macro, n_commodities=5, bloc_map=bloc_map, n_perm=40, seed=0
    )
    assert res.rho_oos.shape == (3,)
    assert res.perm_p_oos.shape == (3,)
    assert res.struct_macro.shape == (4, 3)
    # bloc structure has min(n_factors, n_blocs) canonical columns
    assert res.bloc_struct.shape == (2, 2)
    # the planted direction is strongly recoverable in-sample
    assert res.rho_insample[0] > 0.5
    assert 0 <= res.n_spanned <= 3
    assert len(res.top_macro) == 3 and len(res.top_bloc) == 3


def test_spanning_summary_table_shape() -> None:
    rng = np.random.default_rng(1)
    t = 200
    idx = pd.date_range("2019-01-01", periods=t, freq="B")
    macro = pd.DataFrame(rng.standard_normal((t, 4)), columns=list("abcd"), index=idx)
    factors = pd.DataFrame(rng.standard_normal((t, 2)), columns=["f1", "f2"], index=idx)
    res = spanning_analysis(
        "x", factors, macro, n_commodities=3, bloc_map={"g": ["a", "b"]}, n_perm=10, seed=0
    )
    table = spanning_summary_table({"x": res})
    assert len(table) == 2
    assert {"block", "dim", "rho_oos", "perm_p_oos", "spanned", "top_bloc"}.issubset(table.columns)
