"""Unit tests for the rolling-window forecast engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eqcp.config import FactorModelConfig, RollingForecastConfig
from eqcp.forecasting.ar1 import AR_KEY
from eqcp.forecasting.pbsv import efficiency_gap, pbsv
from eqcp.forecasting.rolling import rank_stability, rolling_forecasts


def _synthetic(T: int = 420, n: int = 6, j: int = 8, seed: int = 0):
    """Returns with a persistent latent factor that also drives the macro panel."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)
    z = rng.standard_normal((T, 2))  # two shared latent drivers
    loadings = rng.standard_normal((2, n))
    returns = z @ loadings + 0.5 * rng.standard_normal((T, n))
    macro_load = rng.standard_normal((2, j))
    macro = pd.DataFrame(
        z @ macro_load + 0.5 * rng.standard_normal((T, j)),
        index=dates,
        columns=[f"m{i}" for i in range(j)],
    )
    return returns, dates, macro


def _cfg() -> tuple[RollingForecastConfig, FactorModelConfig]:
    cfg = RollingForecastConfig(
        train_window=120,
        test_block=20,
        step=60,
        horizons=(1, 5),
        min_train_pairs=20,
        ae_epochs=4,
        n_boot=50,
    )
    fcfg = FactorModelConfig(n_factors=3, epochs=4, patience=2, seed=0)
    return cfg, fcfg


def test_rolling_produces_multihorizon_forecasts():
    returns, dates, macro = _synthetic()
    cfg, fcfg = _cfg()
    roll = rolling_forecasts(returns, dates, macro, cfg, fcfg, "vanilla", seed=0)
    assert set(roll.forecasts) == {1, 5}
    assert roll.n_windows >= 2
    for h, fr in roll.forecasts.items():
        # every origin is genuinely out-of-sample (>= first train window length)
        assert fr.origins.min() >= cfg.train_window
        assert fr.horizon == h
        assert fr.preds.shape[1] == len(fr.origins)


def test_pbsv_efficiency_holds_on_pooled_forecasts():
    returns, dates, macro = _synthetic(seed=1)
    cfg, fcfg = _cfg()
    roll = rolling_forecasts(returns, dates, macro, cfg, fcfg, "vanilla", seed=0)
    fr = roll.forecasts[1]
    res = pbsv(fr)
    # exact Shapley must satisfy efficiency to numerical precision
    assert efficiency_gap(res) < 1e-9
    # AR baseline value is 0 by construction (v is measured against AR_KEY)
    assert res["values"][AR_KEY] == 0.0


def test_rank_stability_bounded_and_shaped():
    returns, dates, macro = _synthetic(seed=2)
    cfg, fcfg = _cfg()
    roll = rolling_forecasts(returns, dates, macro, cfg, fcfg, "vanilla", seed=0)
    stab = rank_stability(roll, cos_stable=0.7, n_perm=50, seed=0)
    assert len(stab) == roll.n_dims
    assert stab["median_abs_cos"].between(0.0, 1.0).all()
    assert stab["macro_stable"].dtype == bool
    # rigor add-ons present and well-formed
    for col in ("null_p95", "chance_abs_cos", "p_value", "median_rho_oos", "stability_class"):
        assert col in stab.columns
    assert stab["p_value"].between(0.0, 1.0).all()
    assert stab["stability_class"].isin(["stable", "partial", "weak", "rotating"]).all()
