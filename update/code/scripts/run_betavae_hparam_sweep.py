#!/usr/bin/env python3
"""BetaVAE hyperparameter sweep: find the best (lr, weight_decay, epochs, n_factors, beta)
combination by minimising out-of-sample reconstruction MSE.

hidden_dim is fixed at 16 throughout.

Usage:
    python scripts/run_betavae_hparam_sweep.py
    python scripts/run_betavae_hparam_sweep.py --factors 3 4 5 --betas 1.0 4.0
    python scripts/run_betavae_hparam_sweep.py --output results/hparam_sweep
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.autoencoders import BetaVAE          # noqa: E402
from src.evaluation.encoding import run_encoding_experiment  # noqa: E402

HIDDEN_DIM = 16

DEFAULT_LRS           = [1e-4, 5e-4, 1e-3, 5e-3]
DEFAULT_WEIGHT_DECAYS = [0.0, 1e-4, 1e-3]
DEFAULT_EPOCHS        = [50, 100, 200, 300]
DEFAULT_FACTORS       = [3, 4, 5, 6, 7, 8, 9]
DEFAULT_BETAS         = [0.3, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",      default=str(REPO_ROOT / "data" / "returns.csv"))
    p.add_argument("--output",    default=str(REPO_ROOT / "results" / "beta_hparam"))
    p.add_argument("--train-end", default="2023-12-31")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--lr",           nargs="+", type=float, default=DEFAULT_LRS,           metavar="LR")
    p.add_argument("--weight-decay", nargs="+", type=float, default=DEFAULT_WEIGHT_DECAYS, metavar="WD")
    p.add_argument("--epochs",       nargs="+", type=int,   default=DEFAULT_EPOCHS,        metavar="E")
    p.add_argument("--factors",      nargs="+", type=int,   default=DEFAULT_FACTORS,       metavar="K")
    p.add_argument("--betas",        nargs="+", type=float, default=DEFAULT_BETAS,         metavar="BETA")
    return p.parse_args()


def mse(result: dict, split: str) -> float:
    err = result[split]["recon_error"].to_numpy()
    return float((err ** 2).mean())


def main() -> None:
    args = parse_args()

    returns = pd.read_csv(args.data, index_col=0, parse_dates=True)
    assert not returns.isna().any().any(), "panel must be clean"

    N = returns.shape[1]
    train = returns.loc[:args.train_end]
    test  = returns.loc[str(int(args.train_end[:4]) + 1):]
    print(
        f"Panel: N={N} | train={len(train)} ({train.index.min().date()}–{train.index.max().date()}) "
        f"test={len(test)} ({test.index.min().date()}–{test.index.max().date()})\n"
        f"Fixed: hidden_dim={HIDDEN_DIM}\n"
        f"Sweeping: lr={args.lr} | wd={args.weight_decay} | epochs={args.epochs} "
        f"| factors={args.factors} | betas={args.betas}\n"
    )

    grid = list(itertools.product(args.factors, args.lr, args.weight_decay, args.epochs, args.betas))
    print(f"Total runs: {len(grid)}\n")

    rows = []
    for i, (k, lr, wd, ep, beta) in enumerate(grid, 1):
        tag = f"K={k} lr={lr:g} wd={wd:g} ep={ep} beta={beta:g}"
        factory = lambda seed, _k=k, _lr=lr, _wd=wd, _ep=ep, _b=beta: BetaVAE(
            n_inputs=N,
            n_factors=_k,
            hidden_dim=HIDDEN_DIM,
            epochs=_ep,
            lr=_lr,
            weight_decay=_wd,
            seed=seed,
            beta=_b,
        )
        t0 = time.time()
        result = run_encoding_experiment(
            returns, factory, train_end=args.train_end, base_seed=args.seed
        )
        elapsed = time.time() - t0
        train_mse = mse(result, "train")
        test_mse  = mse(result, "test")
        rows.append(
            dict(n_factors=k, lr=lr, weight_decay=wd, epochs=ep, beta=beta,
                 train_mse=train_mse, test_mse=test_mse)
        )
        print(f"[{i:>{len(str(len(grid)))}}/{len(grid)}] {tag} | "
              f"train={train_mse:.5f}  test={test_mse:.5f}  ({elapsed:.0f}s)")

    results_df = pd.DataFrame(rows).sort_values("test_mse").reset_index(drop=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "betavae_hparam_sweep.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print(f"\n=== Top 10 configs by test MSE ===")
    print(results_df.head(10).to_string(index=False))

    best = results_df.iloc[0]
    print(
        f"\nBest config:\n"
        f"  n_factors    = {int(best.n_factors)}\n"
        f"  lr           = {best.lr:g}\n"
        f"  weight_decay = {best.weight_decay:g}\n"
        f"  epochs       = {int(best.epochs)}\n"
        f"  beta         = {best.beta:g}\n"
        f"  test MSE     = {best.test_mse:.5f}\n"
        f"  train MSE    = {best.train_mse:.5f}\n"
    )


if __name__ == "__main__":
    main()
