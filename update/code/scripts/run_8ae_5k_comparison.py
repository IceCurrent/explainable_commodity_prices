#!/usr/bin/env python3
"""Entry point: autoencoder comparison for commodity latent factor extraction.

Each architecture is fit once on the full standardised return panel. Latent
factors (T × K) are saved to results/factors/. Architectures are ranked by
reconstruction MSE relative to the VanillaAE baseline.

Usage:
    python scripts/run_comparison.py
    python scripts/run_comparison.py --archs VanillaAE VAE ContractiveAE
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.autoencoders import ALL_ARCHITECTURES  # noqa: E402
from src.evaluation.reporting import write_all_outputs  # noqa: E402
from src.evaluation.encoding import run_encoding_experiment  # noqa: E402

ARCH_BY_NAME = {cls.name: cls for cls in ALL_ARCHITECTURES}
BASELINE = "VanillaAE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(REPO_ROOT / "data" / "returns.csv"))
    p.add_argument("--output", default=str(REPO_ROOT / "results"))
    p.add_argument("--factors", type=int, default=5, help="latent dimension K")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-end", default="2023-12-31", help="last date of training set (inclusive)")
    p.add_argument(
        "--archs",
        nargs="+",
        default=list(ARCH_BY_NAME),
        choices=list(ARCH_BY_NAME),
        help="subset of architectures to run",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if BASELINE not in args.archs:
        args.archs = [BASELINE] + args.archs

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

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment_results: dict[str, dict] = {}
    for name in args.archs:
        cls = ARCH_BY_NAME[name]
        factory = lambda seed, cls=cls: cls(  # noqa: E731
            n_inputs=N, n_factors=args.factors, epochs=args.epochs, seed=seed
        )
        print(f"[{name}] fitting and encoding...", flush=True)
        t0 = time.time()
        result = run_encoding_experiment(returns, factory, train_end=args.train_end, base_seed=args.seed)
        experiment_results[name] = result
        train_mse = float((result["train"]["recon_error"].to_numpy() ** 2).mean())
        test_mse = float((result["test"]["recon_error"].to_numpy() ** 2).mean())
        print(f"[{name}] done in {time.time() - t0:.0f}s | train MSE = {train_mse:.5f} | test MSE = {test_mse:.5f}")

    summary = write_all_outputs(experiment_results, out_dir, baseline=BASELINE)
    print("\n=== Architecture ranking (reconstruction MSE ratio vs. VanillaAE) ===")
    print(summary.round(4).to_string())
    print(f"\nOutputs written to {out_dir}/")


if __name__ == "__main__":
    main()
