# scripts/tune_hyperparams.py

from __future__ import annotations


import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.autoencoders import VanillaAE, BetaVAE
from src.evaluation.hparam_tuner import (
    grid_search,
    random_search,
)

# ==========================================================
# Search Spaces
# ==========================================================

SHARED_GRID = {
    "hidden_dim": [8, 16, 32, 64],
    "n_factors": [1, 3, 5, 8],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "epochs": [150, 300],
}

CAE_GRID = {
    **SHARED_GRID,
    "jacobian_weight": [1e-3, 1e-2, 5e-2, 1e-1],
}

# ==========================================================
# CV Settings
# ==========================================================

N_SPLITS = 5
VAL_SIZE = 21
GAP = 5
TRAIN_WINDOW = 252

MIN_OBS_REQUIRED = (
    TRAIN_WINDOW
    + GAP
    + N_SPLITS * VAL_SIZE
)

SEED = 42


# ==========================================================
# Main
# ==========================================================

def main():

    returns_path = REPO_ROOT / "data" / "returns.csv"

    print(f"\nLoading: {returns_path}")

    returns = pd.read_csv(
        returns_path,
        index_col=0,
        parse_dates=True,
    )

    print(f"Total observations: {len(returns):,}")
    print(f"Assets/features:    {returns.shape[1]}")

    # ------------------------------------------------------
    # Use first 60% for tuning
    # ------------------------------------------------------

    tune_cutoff = int(len(returns) * 0.60)

    X_tune = returns.iloc[:tune_cutoff].to_numpy(
        dtype="float32"
    )

    print(f"\nTuning observations: {len(X_tune):,}")

    if len(X_tune) < MIN_OBS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_OBS_REQUIRED} observations "
            f"for CV. Got {len(X_tune)}."
        )

    # ======================================================
    # Vanilla AE
    # ======================================================

    print("\n" + "=" * 60)
    print("TUNING VANILLA AUTOENCODER")
    print("=" * 60)

    vanilla_result = grid_search(
        X=X_tune,
        arch_cls=VanillaAE,
        param_grid=SHARED_GRID,
        n_splits=N_SPLITS,
        val_size=VAL_SIZE,
        gap=GAP,
        train_window=TRAIN_WINDOW,
        seed=SEED,
        verbose=True,
    )

    print("\nBest VanillaAE Parameters")
    print(vanilla_result.best_params)
    print(
        f"Best CV MSE: "
        f"{vanilla_result.best_score:.8f}"
    )

    # ======================================================
    # Contractive AE
    # ======================================================

    print("\n" + "=" * 60)
    print("TUNING CONTRACTIVE AUTOENCODER")
    print("=" * 60)

    cae_result = random_search(
        X=X_tune,
        arch_cls=BetaVAE,
        param_distributions=CAE_GRID,
        n_iter=40,
        n_splits=N_SPLITS,
        val_size=VAL_SIZE,
        gap=GAP,
        train_window=TRAIN_WINDOW,
        seed=SEED,
        verbose=True,
    )

    print("\nBest ContractiveAE Parameters")
    print(cae_result.best_params)
    print(
        f"Best CV MSE: "
        f"{cae_result.best_score:.8f}"
    )

    # ======================================================
    # Save Results
    # ======================================================

    output = {
        "VanillaAE": vanilla_result.best_params,
        "ContractiveAE": cae_result.best_params,
        "scores": {
            "VanillaAE": float(
                vanilla_result.best_score
            ),
            "ContractiveAE": float(
                cae_result.best_score
            ),
        },
        "cv_settings": {
            "n_splits": N_SPLITS,
            "val_size": VAL_SIZE,
            "gap": GAP,
            "train_window": TRAIN_WINDOW,
        },
    }

    output_dir = REPO_ROOT / "results"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir / "best_hparams.json"
    )

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print("\n" + "=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)
    print(output_file)

    print("\nSummary")
    print("-" * 40)
    print(
        f"VanillaAE     MSE: "
        f"{vanilla_result.best_score:.8f}"
    )
    print(
        f"ContractiveAE MSE: "
        f"{cae_result.best_score:.8f}"
    )


if __name__ == "__main__":
    main()