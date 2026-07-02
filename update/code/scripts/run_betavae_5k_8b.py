#!/usr/bin/env python3
"""BetaVAE beta sweep: fit BetaVAE with multiple beta values and save results.

Results are saved to results/beta_sweep/ (separate from the main architecture
comparison in results/).

Usage:
    python scripts/run_beta_sweep.py
    python scripts/run_beta_sweep.py --beta-values 0.1 0.5 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.autoencoders import BetaVAE  # noqa: E402
from src.evaluation.encoding import run_encoding_experiment  # noqa: E402
from src.evaluation.reporting import write_all_outputs  # noqa: E402

DEFAULT_BETAS = [0.3, 0.5, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "returns.csv"))
    p.add_argument("--output", default=str(REPO_ROOT / "results" / "beta_sweep"))
    p.add_argument("--factors", type=int, default=5, help="latent dimension K")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-end", default="2023-12-31", help="last date of training set (inclusive)")
    p.add_argument(
        "--beta-values",
        nargs="+",
        type=float,
        default=DEFAULT_BETAS,
        metavar="BETA",
        help="beta values to sweep (default: 0.3 0.5 0.8 1 2 3 4 5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    returns = pd.read_csv(args.data, index_col=0, parse_dates=True)
    assert not returns.isna().any().any(), "panel must be clean"

    train = returns.loc[:args.train_end]
    test = returns.loc[str(int(args.train_end[:4]) + 1):]
    T, N = returns.shape
    print(
        f"Panel: T={T}, N={N} | train={len(train)} ({train.index.min().date()}–{train.index.max().date()}) "
        f"test={len(test)} ({test.index.min().date()}–{test.index.max().date()}) | "
        f"K={args.factors}, epochs={args.epochs}, seed={args.seed}"
    )
    print(f"Beta values: {args.beta_values}\n")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_results: dict[str, dict] = {}
    for beta in args.beta_values:
        name = f"BetaVAE_b{beta:g}"
        factory = lambda seed, _b=beta: BetaVAE(
            n_inputs=N, n_factors=args.factors, epochs=args.epochs, seed=seed, beta=_b
        )
        print(f"[{name}] fitting and encoding...", flush=True)
        t0 = time.time()
        result = run_encoding_experiment(returns, factory, train_end=args.train_end, base_seed=args.seed)
        experiment_results[name] = result
        train_mse = float((result["train"]["recon_error"].to_numpy() ** 2).mean())
        test_mse = float((result["test"]["recon_error"].to_numpy() ** 2).mean())
        print(f"[{name}] done in {time.time() - t0:.0f}s | train MSE = {train_mse:.5f} | test MSE = {test_mse:.5f}")

    # Use beta=1 as baseline (standard VAE) if present, otherwise first beta
    betas_sorted = sorted(args.beta_values)
    baseline_beta = min(betas_sorted, key=lambda b: abs(b - 1.0))
    baseline = f"BetaVAE_b{baseline_beta:g}"
    summary = write_all_outputs(experiment_results, out_dir, baseline=baseline)

    print(f"\n=== Beta sweep results (ratio vs {baseline}) ===")
    print(summary.round(4).to_string())
    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()
