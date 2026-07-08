"""Shared, cached pipeline context for deep-analysis scripts.

Rebuilds exactly the forecast-PBSV pipeline's leak-free state (train-only AE,
train-frozen CCA basis, canonical-variate state, macro variates) once and
caches the pieces under results/deep_analysis/cache/.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eqcp.config import load_factor_model_config, load_forecast_pbsv_config
from eqcp.forecasting.basis import FrozenBasis, fit_frozen_basis
from eqcp.io.commodities import load_return_panel
from eqcp.io.macro import load_macro_stationary
from eqcp.pipelines.forecast_pbsv import _standardized_weights, train_leakfree_factors
from eqcp.pipelines.macro_mapping import RunLogger

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "deep_analysis" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class Ctx:
    returns: np.ndarray  # (T, N)
    dates: pd.DatetimeIndex
    commodities: list[str]
    t_split: int
    F: pd.DataFrame  # leak-free AE factors, all days
    basis: FrozenBasis
    state: np.ndarray  # (T, K) canonical-variate state
    M_full: pd.DataFrame  # stationary macro panel
    weights: np.ndarray  # 1/train-var pooling weights (h=1)
    cfg: object
    factor_cfg: object


def load_ctx(seed: int = 0) -> Ctx:
    cfg = load_forecast_pbsv_config(None)
    factor_cfg = load_factor_model_config(None)
    panel = load_return_panel()
    returns, dates = panel.raw, panel.dates
    T, N = returns.shape
    t_split = int(cfg.train_frac * T)
    manifest = pd.read_csv(ROOT / "data/processed/transform_manifest.csv")
    M_full = load_macro_stationary(ROOT / "data/processed/macro_stationary.csv", manifest)

    f_path = CACHE / f"factors_seed{seed}.csv"
    b_path = CACHE / f"basis_seed{seed}.pkl"
    if f_path.exists() and b_path.exists():
        F = pd.read_csv(f_path, parse_dates=["date"], index_col="date")
        with open(b_path, "rb") as fh:
            basis = pickle.load(fh)
    else:
        F = train_leakfree_factors(returns, dates, t_split, factor_cfg, seed, RunLogger())
        basis = fit_frozen_basis(
            F.iloc[:t_split],
            M_full.loc[M_full.index < dates[t_split]],
            n_perm=50,
            seed=seed,
        )
        F.to_csv(f_path)
        with open(b_path, "wb") as fh:
            pickle.dump(basis, fh)

    state = basis.variates(F).to_numpy(float)
    weights = _standardized_weights(returns[:t_split], np.ones(N, dtype=bool))
    return Ctx(
        returns=returns, dates=dates, commodities=list(panel.commodities),
        t_split=t_split, F=F, basis=basis, state=state, M_full=M_full,
        weights=weights, cfg=cfg, factor_cfg=factor_cfg,
    )
