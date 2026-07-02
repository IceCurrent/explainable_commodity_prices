"""Sparse autoencoder: L1 penalty on bottleneck activations."""

import torch
from torch import nn

from .base import BaseAutoencoder, MLPAEModule


class SparseAE(BaseAutoencoder):
    """L1 regularization on the K-dimensional bottleneck.

    With K=5 the bottleneck is already tight, so the L1 term acts as
    shrinkage that discourages weak, noisy factors rather than producing
    classical over-complete sparsity. The weight (1e-2) is moderate by
    design: large enough to matter relative to an O(0.5) reconstruction
    MSE, small enough to avoid collapsing all factors to zero.
    """

    name = "SparseAE"

    def __init__(self, *args, l1_weight: float = 1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1_weight = l1_weight

    def _build(self) -> nn.Module:
        return MLPAEModule(self.n_inputs, self.n_factors, self.hidden_dim)

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encoder(x)
        recon = nn.functional.mse_loss(self.model.decoder(z), x)
        return recon + self.l1_weight * z.abs().mean()
