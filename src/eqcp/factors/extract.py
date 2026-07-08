"""Unified factor extraction for vanilla AE and beta-VAE."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch.nn as nn

from eqcp.config import FactorModelConfig
from eqcp.factors.autoencoder import AETrainConfig, train_vanilla_autoencoder
from eqcp.factors.beta_vae import BetaVAETrainConfig, train_beta_vae
from eqcp.io.commodities import load_return_panel

FactorModelType = Literal["vanilla", "beta_vae"]

FACTOR_CSV_NAMES: dict[FactorModelType, str] = {
    "vanilla": "ae_factors_vanilla.csv",
    "beta_vae": "ae_factors_beta.csv",
}

RESULTS_SUBDIRS: dict[FactorModelType, str] = {
    "vanilla": "macro_mapping",
    "beta_vae": "macro_mapping_beta",
}

FORECAST_SUBDIRS: dict[FactorModelType, str] = {
    "vanilla": "forecast_pbsv",
    "beta_vae": "forecast_pbsv_beta",
}


def train_factors(
    window_returns: np.ndarray,
    factor_cfg: FactorModelConfig,
    model_type: FactorModelType = "vanilla",
    seed: int | None = None,
) -> tuple[nn.Module, np.ndarray]:
    """Train the requested factor model on standardized returns."""
    sd = seed if seed is not None else factor_cfg.seed
    if model_type == "vanilla":
        cfg = AETrainConfig(
            n_factors=factor_cfg.n_factors,
            epochs=factor_cfg.epochs,
            batch_size=factor_cfg.batch_size,
            learning_rate=factor_cfg.learning_rate,
            patience=factor_cfg.patience,
            seed=sd,
            activation=factor_cfg.activation,
        )
        return train_vanilla_autoencoder(window_returns, cfg)
    if model_type == "beta_vae":
        cfg = BetaVAETrainConfig(
            n_factors=factor_cfg.n_factors,
            epochs=factor_cfg.epochs,
            batch_size=factor_cfg.batch_size,
            learning_rate=factor_cfg.learning_rate,
            patience=factor_cfg.patience,
            seed=sd,
            activation=factor_cfg.activation,
            beta=factor_cfg.beta,
        )
        return train_beta_vae(window_returns, cfg)
    raise ValueError(f"unknown factor model type: {model_type!r}")


def factors_to_frame(
    factors: np.ndarray,
    dates: pd.DatetimeIndex,
    n_factors: int,
) -> pd.DataFrame:
    F = pd.DataFrame(
        factors,
        index=dates,
        columns=[f"f{i + 1}" for i in range(n_factors)],
    )
    F.index.name = "date"
    return F


def regenerate_factors(
    factor_cfg: FactorModelConfig,
    model_type: FactorModelType = "vanilla",
    activation: str | None = None,
) -> pd.DataFrame:
    """Train on the full commodity panel and encode all days."""
    act = activation or factor_cfg.activation
    cfg = FactorModelConfig(
        n_factors=factor_cfg.n_factors,
        epochs=factor_cfg.epochs,
        batch_size=factor_cfg.batch_size,
        learning_rate=factor_cfg.learning_rate,
        patience=factor_cfg.patience,
        seed=factor_cfg.seed,
        activation=act,
        beta=factor_cfg.beta,
    )
    panel = load_return_panel()
    _, factors = train_factors(
        panel.standardized,
        cfg,
        model_type=model_type,
        seed=cfg.seed,
    )
    return factors_to_frame(factors, panel.dates, cfg.n_factors)
