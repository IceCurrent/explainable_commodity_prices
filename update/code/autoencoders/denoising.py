"""Denoising autoencoder: reconstruct clean inputs from corrupted ones."""

import torch
from torch import nn

from .base import BaseAutoencoder, MLPAEModule


class DenoisingAE(BaseAutoencoder):
    """Gaussian input corruption.

    Inputs are standardized within each training window, so a fixed noise
    std of 0.3 corresponds to ~30% of a typical daily move -- enough to force
    the encoder toward the cross-sectional common component without drowning
    the signal. Clean inputs are used at inference (encode), as usual.
    """

    name = "DenoisingAE"

    def __init__(self, *args, noise_std: float = 0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_std = noise_std

    def _build(self) -> nn.Module:
        return MLPAEModule(self.n_inputs, self.n_factors, self.hidden_dim)

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        x_noisy = x + self.noise_std * torch.randn_like(x)
        return nn.functional.mse_loss(self.model(x_noisy), x)
