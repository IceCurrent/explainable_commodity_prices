"""CLI entry point for the forecast-PBSV pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from eqcp.config import PROJECT_ROOT
from eqcp.pipelines.forecast_pbsv import run_forecast_pbsv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=run_forecast_pbsv.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dp = PROJECT_ROOT / "data" / "processed"
    cfg_dir = PROJECT_ROOT / "configs"
    p.add_argument("--macro", type=Path, default=dp / "macro_stationary.csv")
    p.add_argument("--manifest", type=Path, default=dp / "transform_manifest.csv")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path, default=cfg_dir / "forecast_pbsv.yaml")
    p.add_argument("--factor-config", type=Path, default=cfg_dir / "factor_model.yaml")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_forecast_pbsv(parse_args(argv))


if __name__ == "__main__":
    main()
