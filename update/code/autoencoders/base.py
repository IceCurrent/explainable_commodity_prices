"""Shared infrastructure for all autoencoder architectures.

Research-validity note: every architecture inherits the same bottleneck size,
hidden width, optimizer, learning rate, epoch budget and weight decay. We
deliberately do NOT tune these per architecture -- with a shared budget, any
out-of-sample difference is attributable to the architectural/regularization
principle itself rather than to unequal tuning effort.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


class BaseAutoencoder:
    """Common interface consumed by the rolling evaluation pipeline.

    Subclasses implement:
      - ``_build()``: construct and return the ``nn.Module``
      - ``_loss(x)``: full-batch training loss on a (T, N) float tensor
    and may override ``_encode_tensor`` if their encoder is not simply
    ``self.model.encoder``.

    Training is full-batch Adam: each rolling window holds only ~252 rows of
    22 features, so mini-batching adds gradient noise without any memory
    benefit. A fixed epoch count (rather than early stopping on reconstruction
    loss) keeps fits deterministic and avoids selecting models on a criterion
    (reconstruction) that is not the downstream objective (forecast MSE).
    """

    name = "Base"

    def __init__(
        self,
        n_inputs: int,
        n_factors: int = 5,
        hidden_dim: int = 16,
        epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.n_inputs = n_inputs
        self.n_factors = n_factors
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = torch.device(device)
        self.model: nn.Module | None = None

    # ---- interface -------------------------------------------------------

    def get_name(self) -> str:
        return self.name

    def fit(self, X: np.ndarray) -> None:
        assert X.ndim == 2 and X.shape[1] == self.n_inputs
        seed_everything(self.seed)
        self.model = self._build().to(self.device)
        self.model.train()
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = self._loss(x)
            loss.backward()
            # Clipping mainly protects the LSTM variant; it is a no-op for
            # well-behaved MLP fits and keeps the training loop uniform.
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            opt.step()

    def encode(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "fit() must be called before encode()"
        self.model.eval()
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self._encode_tensor(x)
        return z.cpu().numpy()

    def decode(self, Z: np.ndarray) -> np.ndarray:
        assert self.model is not None, "fit() must be called before decode()"
        self.model.eval()
        z = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            x_hat = self._decode_tensor(z)
        return x_hat.cpu().numpy()

    # ---- subclass hooks ---------------------------------------------------

    def _build(self) -> nn.Module:
        raise NotImplementedError

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encoder(x)

    def _decode_tensor(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decoder(z)


class MLPAEModule(nn.Module):
    """Symmetric one-hidden-layer autoencoder: N -> H -> K -> H -> N.

    A single Tanh hidden layer (~900 parameters at N=22, H=16, K=5) is sized
    for ~252 training rows per window; deeper stacks overfit badly at this
    scale. Tanh is used everywhere (including ContractiveAE, which needs a
    smooth activation for its analytic Jacobian) so the comparison is not
    confounded by activation choice.
    """

    def __init__(self, n_inputs: int, n_factors: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_factors),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_factors, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_inputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
