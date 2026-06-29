#!/usr/bin/env python3
"""Compute forecast-based Shapley phi_k for the aligned AE factors.

Consumes ``results/factor_bundle.npz``; aligns factors (E0) then attributes OOS
forecast-MSE reduction to each factor. Writes:
  * results/forecast_shapley.csv          - phi_k per commodity + pooled (ALL)
  * results/forecast_shapley_ranking.csv  - pooled phi_k ranked with shares
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data import load_return_panel
from src.factor_alignment import align_by_correlation
from src.factor_extraction import FactorBundle
from src.forecast_shapley import compute_forecast_shapley


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "results" / "factor_bundle.npz")
    p.add_argument("--reference", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = FactorBundle.load(args.bundle)
    panel = load_return_panel()
    alignment = align_by_correlation(bundle, reference=args.reference)

    print(f"Computing forecast-Shapley over {bundle.n_windows} windows, "
          f"K={bundle.n_factors} factors, {bundle.n_assets} commodities...", flush=True)
    result = compute_forecast_shapley(bundle, alignment, panel=panel)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = result.to_frame()
    frame.to_csv(args.output_dir / "forecast_shapley.csv")

    phi = result.phi_aggregate
    total = np.sum(np.abs(phi))
    ranking = pd.DataFrame(
        {
            "factor": result.factors,
            "phi_aggregate": phi,
            "abs_share": np.abs(phi) / total if total > 0 else np.zeros_like(phi),
        }
    ).sort_values("phi_aggregate", ascending=False, key=np.abs)
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking.to_csv(args.output_dir / "forecast_shapley_ranking.csv", index=False)

    print("\nPooled forecast-Shapley (MSE reduction vs AR(1)):")
    print(ranking.to_string(index=False))
    print(f"\nBaseline pooled MSE (AR1): {result.mse_ar1_aggregate:.6e}")
    print(f"Full-factor pooled MSE:    {result.mse_full_aggregate:.6e}")
    print(f"Sum phi_k = {phi.sum():.6e}  (efficiency check = "
          f"{result.mse_ar1_aggregate - result.mse_full_aggregate:.6e})")
    print(f"\nWrote -> {args.output_dir / 'forecast_shapley.csv'}")


if __name__ == "__main__":
    main()
