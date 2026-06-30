#!/usr/bin/env python3
"""CLI entry point for macro panel processing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eqcp.macro_processing.process_macro import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Build macro panel from workbook.")
    ap.add_argument("--input", default="data/macro/NEW_MACRO_COMMODITY_PANEL.xlsx")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--seed", type=int, default=0, help="Reserved for reproducibility logging.")
    args = ap.parse_args()
    logging.info("seed=%s input=%s outdir=%s", args.seed, args.input, args.outdir)
    run(Path(args.input), Path(args.outdir))


if __name__ == "__main__":
    main()
