"""Vanilla bottleneck autoencoder for commodity factor extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

VALID_ACTIVATIONS = ("relu", "linear", "tanh")


class VanillaAutoencoder(nn.Module):
    """Single-layer bottleneck autoencoder.

    ``activation`` controls the encoder nonlinearity:
    - ``relu`` (default): one-sided latents, project baseline
    - ``linear``: PCA-like linear bottleneck
    - ``tanh``: symmetric bounded latents (experiment knob for one-sidedness)
    """

    def __init__(self, n_assets: int, n_factors: int, activation: str = "relu") -> None:
        super().__init__()
        if activation not in VALID_ACTIVATIONS:
            raise ValueError(f"activation must be one of {VALID_ACTIVATIONS}, got {activation!r}")
        self.activation = activation
        enc_layers: list[nn.Module] = [nn.Linear(n_assets, n_factors)]
        if activation == "relu":
            enc_layers.append(nn.ReLU())
        elif activation == "tanh":
            enc_layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(nn.Linear(n_factors, n_assets))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))

    def decoder_weight(self) -> np.ndarray:
        layer = self.decoder[0]
        assert isinstance(layer, nn.Linear)
        return layer.weight.detach().cpu().numpy().astype(np.float64)

    def decoder_bias(self) -> np.ndarray:
        layer = self.decoder[0]
        assert isinstance(layer, nn.Linear)
        return layer.bias.detach().cpu().numpy().astype(np.float64)


@dataclass
class AETrainConfig:
    n_factors: int = 5
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 5
    seed: int = 42
    activation: str = "relu"


def train_vanilla_autoencoder(
    window_returns: np.ndarray,
    config: AETrainConfig | None = None,
    init_state: dict[str, torch.Tensor] | None = None,
) -> tuple[VanillaAutoencoder, np.ndarray]:
    config = config or AETrainConfig()
    torch.manual_seed(config.seed)

    n_obs, n_assets = window_returns.shape
    device = torch.device("cpu")
    model = VanillaAutoencoder(n_assets, config.n_factors, activation=config.activation).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

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
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
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
