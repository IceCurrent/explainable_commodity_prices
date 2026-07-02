"""LSTM encoder-decoder producing one factor vector per time step."""

import torch
from torch import nn

from .base import BaseAutoencoder


class LSTMAEModule(nn.Module):
    def __init__(self, n_inputs: int, n_factors: int, hidden_dim: int):
        super().__init__()
        self.enc_lstm = nn.LSTM(n_inputs, hidden_dim, batch_first=True)
        self.enc_head = nn.Linear(hidden_dim, n_factors)
        self.dec_lstm = nn.LSTM(n_factors, hidden_dim, batch_first=True)
        self.dec_head = nn.Linear(hidden_dim, n_inputs)

    def encode_seq(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.enc_lstm(x.unsqueeze(0))      # (1, T, H)
        return self.enc_head(h).squeeze(0)        # (T, K)

    def decode_seq(self, z: torch.Tensor) -> torch.Tensor:
        h, _ = self.dec_lstm(z.unsqueeze(0))
        return self.dec_head(h).squeeze(0)


class RecurrentAE(BaseAutoencoder):
    """Sequence autoencoder over the time dimension.

    Design choices that affect temporal factor structure:

    1. CAUSALITY: the encoder LSTM is unidirectional, so the factor F_t is a
       function of X_1..X_t only. A bidirectional encoder would reconstruct
       better but would leak future information into F_t, invalidating the
       one-step-ahead forecasting evaluation. This is non-negotiable for
       research validity.
    2. PER-STEP FACTORS: instead of compressing the whole window into a
       single vector (the seq2seq convention), we read a factor vector off
       the hidden state at EVERY step, giving the (T, K) output the pipeline
       requires. Factors therefore carry recursive memory: unlike the
       pointwise architectures, F_t can encode recent history (e.g. local
       volatility), not just the date-t cross-section.
    3. Training is on the full window as one sequence (300 full-sequence
       gradient steps). The LSTM (~13k parameters) is over-parameterized for
       252 observations; weight decay and gradient clipping are the guard
       rails. Recurrent models being data-hungry at this window size is a
       known handicap and part of what the comparison is measuring.
    """

    name = "RecurrentAE"

    def __init__(self, *args, hidden_dim: int = 32, lr: float = 5e-3, **kwargs):
        # Higher lr than the MLPs: 300 full-sequence steps at 1e-3 underfit.
        kwargs["hidden_dim"] = hidden_dim
        kwargs["lr"] = lr
        super().__init__(*args, **kwargs)

    def _build(self) -> nn.Module:
        return LSTMAEModule(self.n_inputs, self.n_factors, self.hidden_dim)

    def _loss(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encode_seq(x)
        recon = self.model.decode_seq(z)
        return nn.functional.mse_loss(recon, x)

    def _encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_seq(x)

    def _decode_tensor(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode_seq(z)
