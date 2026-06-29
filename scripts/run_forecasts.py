#!/usr/bin/env python3
"""Run rolling-window AR(1) and vanilla-AE factor-augmented forecasts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_return_panel
from src.rolling_forecast import RollingForecastConfig, run_rolling_forecasts, save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling-window one-step-ahead forecasts using AR(1) and "
            "vanilla autoencoder factor-augmented AR(1), following "
            "Cerqueti et al. (2024) core methodology."
        )
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1260,
        help="Estimation window M in trading days (default: 1260 ≈ 5 years).",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=5,
        help="Number of latent factors K (default: 5, per paper).",
    )
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=50,
        help="Max training epochs for the autoencoder per window.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on out-of-sample windows (for quick runs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for forecast outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for autoencoder training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_return_panel()
    config = RollingForecastConfig(
        window_size=args.window_size,
        n_factors=args.n_factors,
        ae_epochs=args.ae_epochs,
        max_windows=args.max_windows,
        seed=args.seed,
    )

    n_oos = panel.n_obs - config.window_size
    if config.max_windows is not None:
        n_oos = min(n_oos, config.max_windows)

    print(
        f"Panel: {panel.n_obs} days × {panel.n_assets} commodities "
        f"({panel.dates.min().date()} → {panel.dates.max().date()})",
        flush=True,
    )
    print(
        f"Rolling forecast: M={config.window_size}, K={config.n_factors}, "
        f"out-of-sample windows={n_oos}",
        flush=True,
    )

    t0 = time.perf_counter()
    results = run_rolling_forecasts(panel=panel, config=config)
    elapsed = time.perf_counter() - t0

    save_results(results, args.output_dir)
    print(f"Finished in {elapsed / 60:.1f} min")
    print(f"Wrote forecasts to {args.output_dir / 'forecasts.csv'}")
    print(f"Wrote errors to {args.output_dir / 'errors.csv'}")
    print(f"Wrote summary to {args.output_dir / 'summary.csv'}")

    overall = results.summary[results.summary["commodity"] == "ALL"].iloc[0]
    print(
        "Overall MSE — AR(1): "
        f"{overall['mse_ar1']:.6e}, Vanilla AE: {overall['mse_vanilla_ae']:.6e}"
    )


if __name__ == "__main__":
    main()
