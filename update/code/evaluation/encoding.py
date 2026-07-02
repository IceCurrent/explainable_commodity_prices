"""Autoencoder fit and latent factor extraction with train/test split.

The AE is fit on the training set only. The scaler (mean/std) is also fit on
training data and applied to both splits to avoid look-ahead bias. Factors and
reconstruction errors are produced for both train and test periods.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.autoencoders.base import BaseAutoencoder


def run_encoding_experiment(
    returns: pd.DataFrame,
    model_factory: Callable[[int], BaseAutoencoder],
    train_end: str = "2023-12-31",
    base_seed: int = 0,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fit one AE on the training period and encode both train and test.

    The scaler is fit on train only (no look-ahead). Returns a dict with keys
    'train' and 'test', each containing:
      'factors'     — (T, K) DataFrame, columns F1 … FK
      'recon'       — (T, N) decoded reconstructions (standardised units)
      'recon_error' — (T, N) actual - recon (standardised units)
    """
    train = returns.loc[:train_end]
    test = returns.loc[str(int(train_end[:4]) + 1):]

    X_train = train.to_numpy(dtype=np.float64)
    X_test = test.to_numpy(dtype=np.float64)

    mu = X_train.mean(axis=0)
    sd = np.maximum(X_train.std(axis=0), 1e-8)

    Z_train = (X_train - mu) / sd
    Z_test = (X_test - mu) / sd

    model = model_factory(base_seed)
    model.fit(Z_train)

    def _encode_decode(Z: np.ndarray, index: pd.Index) -> dict[str, pd.DataFrame]:
        factors = model.encode(Z)
        recon = model.decode(factors)
        factor_cols = [f"F{k + 1}" for k in range(factors.shape[1])]
        return {
            "factors": pd.DataFrame(factors, index=index, columns=factor_cols),
            "recon": pd.DataFrame(recon, index=index, columns=returns.columns),
            "recon_error": pd.DataFrame(Z - recon, index=index, columns=returns.columns),
        }

    return {
        "train": _encode_decode(Z_train, train.index),
        "test": _encode_decode(Z_test, test.index),
    }
