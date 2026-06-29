"""Standard deterministic autoencoder. Serves as the comparison baseline."""

import torch
from torch import nn

from .base import BaseAutoencoder, MLPAEModule


class VanillaAE(BaseAutoencoder):
    name = "VanillaAE"

    def _build(self) -> nn.Module:
        return MLPAEModule(self.n_inputs, self.n_factors, self.hidden_dim)

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(self.model(x), x)
