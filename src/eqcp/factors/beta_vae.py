"""Beta-VAE for commodity factor extraction with disentangled latents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from eqcp.factors.autoencoder import VALID_ACTIVATIONS


class BetaVAE(nn.Module):
    """Single-layer beta-VAE with Gaussian latent and reparameterization."""

    def __init__(
        self,
        n_assets: int,
        n_factors: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in VALID_ACTIVATIONS:
            raise ValueError(f"activation must be one of {VALID_ACTIVATIONS}, got {activation!r}")
        self.activation = activation
        enc_layers: list[nn.Module] = [nn.Linear(n_assets, n_factors)]
        if activation == "relu":
            enc_layers.append(nn.ReLU())
        elif activation == "tanh":
            enc_layers.append(nn.Tanh())
        self.encoder_body = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(n_factors, n_factors)
        self.fc_logvar = nn.Linear(n_factors, n_factors)
        self.decoder = nn.Sequential(nn.Linear(n_factors, n_assets))

    def encode_params(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent means (used for downstream factor analysis)."""
        mu, _ = self.encode_params(x)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_params(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decoder_weight(self) -> np.ndarray:
        layer = self.decoder[0]
        assert isinstance(layer, nn.Linear)
        return layer.weight.detach().cpu().numpy().astype(np.float64)

    def decoder_bias(self) -> np.ndarray:
        layer = self.decoder[0]
        assert isinstance(layer, nn.Linear)
        return layer.bias.detach().cpu().numpy().astype(np.float64)


@dataclass
class BetaVAETrainConfig:
    n_factors: int = 5
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 5
    seed: int = 42
    activation: str = "relu"
    beta: float = 4.0


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def train_beta_vae(
    window_returns: np.ndarray,
    config: BetaVAETrainConfig | None = None,
    init_state: dict[str, torch.Tensor] | None = None,
) -> tuple[BetaVAE, np.ndarray]:
    """Train a beta-VAE; return model and latent means on all observations."""
    config = config or BetaVAETrainConfig()
    torch.manual_seed(config.seed)

    n_obs, n_assets = window_returns.shape
    device = torch.device("cpu")
    model = BetaVAE(n_assets, config.n_factors, activation=config.activation).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    recon_criterion = nn.MSELoss(reduction="sum")

    x = torch.tensor(window_returns, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=min(config.batch_size, n_obs),
        shuffle=True,
    )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _ in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            recon_loss = recon_criterion(recon, batch_x)
            kl = _kl_divergence(mu, logvar).sum()
            loss = recon_loss + config.beta * kl
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        if epoch_loss < best_loss - 1e-8:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        factors = model.encode(x.to(device)).cpu().numpy()
    return model, factors
