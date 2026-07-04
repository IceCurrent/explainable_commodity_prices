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

Tier_1_AE = {
    "n_factors": [3, 4, 6, 8, 10],
    "weight_decay": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    "dropout_rate": [0.0, 0.1, 0.2, 0.3],
    "lr": [1e-4, 1e-3, 3e-3],
}

Tier_2_AE = {
    "hidden_dim": [8, 16, 32, 64],
    "depth": [1, 2, 3],
    "epochs": [150, 300],
}

SHARED_GRID = {
    "n_factors": [1, 3, 5, 8],
    "weight_decay": [0.0, 1e-4, 1e-3],
    "hidden_dim": [8, 16, 32, 64],
    "epochs": [150, 300],
}

Tier_1_BVAE = {
    **Tier_1_AE,
    "beta": [0.2, 0.3, 0.5, 0.8],
}

Tier_2_BVAE = {
    **Tier_2_AE
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

    returns_path = REPO_ROOT / "data" / "final_returns.csv"

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

    cv_kwargs = dict(
        n_splits=N_SPLITS,
        val_size=VAL_SIZE,
        gap=GAP,
        train_window=TRAIN_WINDOW,
        seed=SEED,
        verbose=True,
    )

    # ======================================================
    # Vanilla AE - Tier 1 (regularization), Tier 2 (capacity)
    # ======================================================

    print("\n" + "=" * 60)
    print("TUNING VANILLA AUTOENCODER -- TIER 1")
    print("=" * 60)

    vanilla_tier1 = grid_search(
        X=X_tune,
        arch_cls=VanillaAE,
        param_grid=Tier_1_AE,
        **cv_kwargs,
    )

    print("\nBest VanillaAE Tier-1 Parameters")
    print(vanilla_tier1.best_params)

    print("\n" + "=" * 60)
    print("TUNING VANILLA AUTOENCODER -- TIER 2")
    print("=" * 60)

    vanilla_tier2 = grid_search(
        X=X_tune,
        arch_cls=VanillaAE,
        param_grid=Tier_2_AE,
        fixed_params=vanilla_tier1.best_params,
        **cv_kwargs,
    )

    print("\nBest VanillaAE Tier-2 Parameters")
    print(vanilla_tier2.best_params)

    vanilla_best_params = {**vanilla_tier1.best_params, **vanilla_tier2.best_params}
    vanilla_best_score = vanilla_tier2.best_score

    print(f"\nBest CV MSE: {vanilla_best_score:.8f}")

    # ======================================================
    # Beta VAE - Tier 1 (regularization + beta), Tier 2 (capacity)
    # ======================================================

    print("\n" + "=" * 60)
    print("TUNING BETA VAE -- TIER 1")
    print("=" * 60)

    bvae_tier1 = random_search(
        X=X_tune,
        arch_cls=BetaVAE,
        param_distributions=Tier_1_BVAE,
        n_iter=40,
        **cv_kwargs,
    )

    print("\nBest BetaVAE Tier-1 Parameters")
    print(bvae_tier1.best_params)

    print("\n" + "=" * 60)
    print("TUNING BETA VAE -- TIER 2")
    print("=" * 60)

    bvae_tier2 = grid_search(
        X=X_tune,
        arch_cls=BetaVAE,
        param_grid=Tier_2_BVAE,
        fixed_params=bvae_tier1.best_params,
        **cv_kwargs,
    )

    print("\nBest BetaVAE Tier-2 Parameters")
    print(bvae_tier2.best_params)

    bvae_best_params = {**bvae_tier1.best_params, **bvae_tier2.best_params}
    bvae_best_score = bvae_tier2.best_score

    print(f"\nBest CV MSE: {bvae_best_score:.8f}")

    # ======================================================
    # Save Results
    # ======================================================

    output = {
        "VanillaAE": vanilla_best_params,
        "BetaVAE": bvae_best_params,
        "scores": {
            "VanillaAE": float(
                vanilla_best_score
            ),
            "BetaVAE": float(
                bvae_best_score
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
        f"{vanilla_best_score:.8f}"
    )
    print(
        f"BetaVAE MSE: "
        f"{bvae_best_score:.8f}"
    )


if __name__ == "__main__":
    main()