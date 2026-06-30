"""CLI entry point for the macro-mapping pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from eqcp.config import PROJECT_ROOT
from eqcp.pipelines.macro_mapping import run_macro_mapping


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=run_macro_mapping.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dp = PROJECT_ROOT / "data" / "processed"
    cfg_dir = PROJECT_ROOT / "configs"
    p.add_argument("--factors", type=str, default=None)
    p.add_argument("--macro", type=Path, default=dp / "macro_stationary.csv")
    p.add_argument("--levels", type=Path, default=dp / "macro_levels_aligned.csv")
    p.add_argument("--manifest", type=Path, default=dp / "transform_manifest.csv")
    p.add_argument("--regimes", type=Path, default=dp / "regimes.csv")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path, default=cfg_dir / "macro_mapping.yaml")
    p.add_argument("--factor-config", type=Path, default=cfg_dir / "factor_model.yaml")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run_macro_mapping(parse_args(argv))


if __name__ == "__main__":
    main()
