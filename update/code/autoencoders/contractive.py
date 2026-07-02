"""Contractive autoencoder: Frobenius penalty on the encoder Jacobian."""

import torch
from torch import nn

from .base import BaseAutoencoder, MLPAEModule


class ContractiveAE(BaseAutoencoder):
    """Penalizes ||d z / d x||_F^2 averaged over the batch.

    The shared encoder is exactly Linear -> Tanh -> Linear, so the Jacobian
    has the closed form  J = W2 @ diag(1 - a^2) @ W1  per sample (a = hidden
    activations). We compute it exactly with an einsum instead of autograd
    double-backward: at (T~252, N=22, H=16, K=5) this is ~0.4M multiply-adds
    per epoch, i.e. essentially free, and avoids the approximation of
    penalizing only the hidden layer as in the original Rifai et al. recipe.
    """

    name = "ContractiveAE"

    def __init__(self, *args, jacobian_weight: float = 1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        self.jacobian_weight = jacobian_weight

    def _build(self) -> nn.Module:
        return MLPAEModule(self.n_inputs, self.n_factors, self.hidden_dim)

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.model.encoder
        lin1, act, lin2 = enc[0], enc[1], enc[2]
        a = act(lin1(x))                      # (T, H)
        z = lin2(a)                           # (T, K)
        recon = nn.functional.mse_loss(self.model.decoder(z), x)

        dadx = 1.0 - a**2                     # tanh'(pre-activation), (T, H)
        # J[t] = W2 @ diag(dadx[t]) @ W1  -> (T, K, N)
        jac = torch.einsum("kh,th,hn->tkn", lin2.weight, dadx, lin1.weight)
        penalty = jac.pow(2).sum(dim=(1, 2)).mean()
        return recon + self.jacobian_weight * penalty
