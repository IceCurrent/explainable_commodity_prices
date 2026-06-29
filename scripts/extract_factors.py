#!/usr/bin/env python3
"""Re-estimate the rolling AE and persist the latent-factor bundle.

This is the prerequisite artifact for every macro-mapping experiment (E0-E4).
Output is a single compressed ``results/factor_bundle.npz`` plus a CSV of the
live (most-recent) factor per window for quick inspection.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data import load_return_panel
from src.factor_extraction import FactorExtractionConfig, extract_factors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window-size", type=int, default=1260)
    p.add_argument("--n-factors", type=int, default=5)
    p.add_argument("--ae-epochs", type=int, default=50)
    p.add_argument("--stride", type=int, default=1,
                   help="Window stride. 1 reproduces the forecaster; larger strides "
                        "give a faster stability audit at coarser resolution.")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--warm-start", action="store_true",
                   help="Warm-start each window's AE from the previous window (E0 step 3).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "factor_bundle.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    panel = load_return_panel()
    config = FactorExtractionConfig(
        window_size=args.window_size,
        n_factors=args.n_factors,
        ae_epochs=args.ae_epochs,
        stride=args.stride,
        max_windows=args.max_windows,
        warm_start=args.warm_start,
        seed=args.seed,
    )
    print(
        f"Panel: {panel.n_obs} days x {panel.n_assets} commodities "
        f"({panel.dates.min().date()} -> {panel.dates.max().date()})",
        flush=True,
    )
    t0 = time.perf_counter()
    bundle = extract_factors(panel=panel, config=config)
    elapsed = time.perf_counter() - t0

    bundle.save(args.output)
    live = pd.DataFrame(
        bundle.live_factors(),
        index=bundle.forecast_dates,
        columns=[f"f{k+1}" for k in range(bundle.n_factors)],
    )
    live.index.name = "date"
    live_path = args.output.with_name("factors_live_raw.csv")
    live.to_csv(live_path)

    print(f"Extracted {bundle.n_windows} windows in {elapsed / 60:.1f} min", flush=True)
    print(f"Wrote factor bundle to {args.output}")
    print(f"  (factors {bundle.factors.shape}, decoder {bundle.decoder_weight.shape})")
    print(f"Wrote raw live factors to {live_path}")
    print("NOTE: these are UNALIGNED. Run scripts/run_alignment.py next (E0).")


if __name__ == "__main__":
    main()
