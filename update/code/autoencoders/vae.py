"""Variational autoencoder with Gaussian posterior and reparameterization."""

import torch
from torch import nn

from .base import BaseAutoencoder


class VAEModule(nn.Module):
    def __init__(self, n_inputs: int, n_factors: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_inputs, hidden_dim), nn.Tanh())
        self.mu_head = nn.Linear(hidden_dim, n_factors)
        self.logvar_head = nn.Linear(hidden_dim, n_factors)
        self.decoder = nn.Sequential(
            nn.Linear(n_factors, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_inputs),
        )

    def encode_params(self, x: torch.Tensor):
        h = self.trunk(x)
        # Clamp keeps the KL term finite early in training on tiny windows.
        return self.mu_head(h), self.logvar_head(h).clamp(-8.0, 8.0)


class VAE(BaseAutoencoder):
    """Standard VAE (beta = 1).

    Loss follows the Gaussian-likelihood convention: squared error summed
    over the N features plus beta * KL summed over the K latents, averaged
    over the batch (scaled by 1/N for a stable learning rate across
    architectures). At inference `encode` returns the posterior MEAN, which
    is the deterministic, minimum-variance factor estimate appropriate for
    the downstream OLS forecasting step.
    """

    name = "VAE"

    def __init__(self, *args, beta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def _build(self) -> nn.Module:
        return VAEModule(self.n_inputs, self.n_factors, self.hidden_dim)

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.model.encode_params(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        recon = (self.model.decoder(z) - x).pow(2).sum(dim=1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)
        return (recon + self.beta * kl).mean() / self.n_inputs

    def _encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.model.encode_params(x)
        return mu
