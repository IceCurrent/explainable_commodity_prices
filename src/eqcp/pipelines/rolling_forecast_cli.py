"""CLI entry point for the rolling-window forecast pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from eqcp.config import PROJECT_ROOT
from eqcp.pipelines.rolling_forecast import run_rolling_forecast


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=run_rolling_forecast.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dp = PROJECT_ROOT / "data" / "processed"
    cfg_dir = PROJECT_ROOT / "configs"
    p.add_argument("--macro", type=Path, default=dp / "macro_stationary.csv")
    p.add_argument("--manifest", type=Path, default=dp / "transform_manifest.csv")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path, default=cfg_dir / "rolling_forecast.yaml")
    p.add_argument("--factor-config", type=Path, default=cfg_dir / "factor_model.yaml")
    p.add_argument(
        "--factor-model",
        choices=("vanilla", "beta_vae"),
        default="vanilla",
        help="Factor extraction backend (vanilla AE or beta-VAE)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_rolling_forecast(parse_args(argv))


if __name__ == "__main__":
    main()
