# src/evaluation/hparam_cv.py

from __future__ import annotations

import numpy as np
import torch
from typing import Callable


def walk_forward_cv(
    X: np.ndarray,
    factory: Callable,
    n_splits: int = 5,
    val_size: int = 21,
    gap: int = 5,
    train_window: int = 252,
) -> float:
    """
    Walk-forward CV for Autoencoder hyperparameter tuning.

    Evaluates Out-of-Sample reconstruction MSE.

    Fold layout:

        [------ TRAIN ------][gap][-- VAL --]

    Parameters
    ----------
    X : np.ndarray
        Shape (T, N)

    factory : Callable
        Returns a fresh Autoencoder model wrapper.

    n_splits : int
        Number of validation folds.

    val_size : int
        Number of observations in each validation fold.

    gap : int
        Purge gap to reduce leakage.

    train_window : int
        Rolling training window length.

    Returns
    -------
    float
        Mean validation reconstruction MSE.
    """

    T = len(X)

    min_required = train_window + gap + n_splits * val_size

    if T < min_required:
        raise ValueError(
            f"Need at least {min_required} observations, got {T}"
        )

    fold_mses = []

    # oldest fold -> newest fold
    for fold in range(n_splits):

        val_start = (
            T
            - n_splits * val_size
            + fold * val_size
        )

        val_end = val_start + val_size

        train_end = val_start - gap
        train_start = max(0, train_end - train_window)

        X_train = X[train_start:train_end]
        X_val = X[val_start:val_end]

        # --------------------------------------------------
        # Fit AE
        # --------------------------------------------------
        model = factory()
        model.fit(X_train)

        # --------------------------------------------------
        # Reconstruction Error
        # --------------------------------------------------
        model.model.eval()

        device = next(model.model.parameters()).device

        with torch.no_grad():

            x_val = torch.as_tensor(
                X_val,
                dtype=torch.float32,
                device=device
            )

            output = model.model(x_val)

            if isinstance(output, tuple):
                recon = output[0]
            else:
                recon = output

            recon = recon.detach().cpu().numpy()

        mse = float(np.mean((recon - X_val) ** 2))

        fold_mses.append(mse)

    return float(np.mean(fold_mses))