"""Masked autoencoder trained with contiguous temporal block masking."""

import torch
from torch import nn

from .base import BaseAutoencoder, MLPAEModule


class MaskedAE(BaseAutoencoder):
    """Reconstruction loss computed ONLY on masked positions.

    Masking design: for each commodity independently we zero out random
    CONTIGUOUS temporal blocks (default: 10-day blocks, ~25% of the sample).
    Because the encoder is pointwise (row-wise MLP), masking entire rows
    would leave the model with no information to reconstruct from; masking
    per-commodity blocks instead means that at a masked (t, i) the model
    must impute commodity i's return from the *unmasked cross-section* at
    time t. Temporal contiguity makes the task hard in the right way: a
    commodity disappears for two trading weeks at a stretch, so the model
    cannot exploit splatter-pattern correlations and is forced to learn the
    persistent cross-sectional co-movement structure -- exactly the common
    factors the downstream forecasting model needs.

    A fresh mask is drawn every epoch. Masked entries are set to 0 (the
    within-window mean after standardization), and no mask indicator channel
    is appended so that the encoder input at inference (fully observed data)
    matches the training distribution at unmasked positions.
    """

    name = "MaskedAE"

    def __init__(self, *args, mask_frac: float = 0.25, block_len: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_frac = mask_frac
        self.block_len = block_len

    def _build(self) -> nn.Module:
        return MLPAEModule(self.n_inputs, self.n_factors, self.hidden_dim)

    def _sample_mask(self, T: int, N: int, device) -> torch.Tensor:
        n_blocks = max(1, int(round(self.mask_frac * T / self.block_len)))
        mask = torch.zeros(T, N, dtype=torch.bool, device=device)
        starts = torch.randint(0, max(1, T - self.block_len + 1), (n_blocks, N))
        for i in range(N):
            for b in range(n_blocks):
                s = int(starts[b, i])
                mask[s : s + self.block_len, i] = True
        return mask

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._sample_mask(x.shape[0], x.shape[1], x.device)
        recon = self.model(x * (~mask))
        return ((recon - x).pow(2) * mask).sum() / mask.sum().clamp(min=1)
