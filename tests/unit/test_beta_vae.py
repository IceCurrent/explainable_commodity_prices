"""Unit tests for beta-VAE factor extraction."""

from __future__ import annotations

import numpy as np

from eqcp.config import FactorModelConfig
from eqcp.factors.beta_vae import BetaVAETrainConfig, train_beta_vae
from eqcp.factors.extract import train_factors


def test_beta_vae_trains_and_returns_correct_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 8)).astype(np.float64)
    cfg = BetaVAETrainConfig(n_factors=3, epochs=5, patience=2, batch_size=32, seed=0)
    model, factors = train_beta_vae(x, cfg)
    assert factors.shape == (120, 3)
    assert np.isfinite(factors).all()
    torch = __import__("torch")
    recon = model.decoder(model.encode(torch.tensor(x, dtype=torch.float32)))
    assert recon.shape == (120, 8)


def test_train_factors_beta_matches_vanilla_api():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((80, 6)).astype(np.float64)
    fc = FactorModelConfig(n_factors=2, epochs=4, patience=2, batch_size=16, seed=1, beta=2.0)
    _, f_beta = train_factors(x, fc, model_type="beta_vae")
    _, f_van = train_factors(x, fc, model_type="vanilla")
    assert f_beta.shape == f_van.shape == (80, 2)
