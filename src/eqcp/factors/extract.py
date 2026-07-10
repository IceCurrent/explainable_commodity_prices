"""Unified factor extraction for the vanilla AE and the beta-VAE.

The rolling engine calls :func:`train_factors` once per window on the window's
standardized train block — there is no full-sample fit/encode path (that was the
leakage-prone design this project removed).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch.nn as nn

from eqcp.config import FactorModelConfig
from eqcp.factors.autoencoder import AETrainConfig, train_vanilla_autoencoder
from eqcp.factors.beta_vae import BetaVAETrainConfig, train_beta_vae

FactorModelType = Literal["vanilla", "beta_vae"]


def train_factors(
    window_returns: np.ndarray,
    factor_cfg: FactorModelConfig,
    model_type: FactorModelType = "vanilla",
    seed: int | None = None,
) -> tuple[nn.Module, np.ndarray]:
    """Train the requested factor model on standardized returns.

    Returns the trained module (for encoding held-out days through the same
    encoder) and the train-block latent factors.
    """
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
        vae_cfg = BetaVAETrainConfig(
            n_factors=factor_cfg.n_factors,
            epochs=factor_cfg.epochs,
            batch_size=factor_cfg.batch_size,
            learning_rate=factor_cfg.learning_rate,
            patience=factor_cfg.patience,
            seed=sd,
            activation=factor_cfg.activation,
            beta=factor_cfg.beta,
        )
        return train_beta_vae(window_returns, vae_cfg)
    raise ValueError(f"unknown factor model type: {model_type!r}")
