#!/usr/bin/env python3
"""Clean a raw macro panel into the standard format expected by src/macro_data.py.

INPUT FORMAT
------------
Accepts any wide-format CSV or Excel file where:
  - one column is the date (named "date", "Date", or specified via --date-col)
  - remaining columns are macro series (one series per column)
  - values are raw levels (transforms are specified in the manifest, not here)

OUTPUT (written to data/macro/)
-------------------------------
  macro_panel_clean.csv   wide panel: date index + one column per series
  manifest.csv            name, transform, group for each series

USAGE EXAMPLES
--------------
  # Minimal: just point at the file; all series treated as stationary (level).
  python data/clean_macro_panel.py --input data/my_macro_data.csv

  # Provide a manifest CSV (name, transform, group) to specify transforms.
  python data/clean_macro_panel.py --input data/my_macro_data.xlsx \\
      --manifest data/my_manifest.csv

  # Override the date column name.
  python data/clean_macro_panel.py --input data/my_macro_data.csv \\
      --date-col "observation_date"

MANIFEST FORMAT
---------------
A CSV with (at minimum) these columns:
  name        matches the column name in your input file
  transform   one of: level | diff | log_change
  group       arbitrary label, e.g. FX, Rates, Risk, Inflation, Growth, Supply

If --manifest is omitted, a default manifest is generated with transform="level"
and group="unknown" for every series. Edit data/macro/manifest.csv afterward
to add correct transforms before running the explainability pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "macro"

VALID_TRANSFORMS = {"level", "diff", "log_change"}


def load_input(path: Path, date_col: str) -> pd.DataFrame:
    """Load CSV or Excel; parse dates; set date as index."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Find the date column (case-insensitive fallback).
    if date_col in df.columns:
        col = date_col
    else:
        matches = [c for c in df.columns if c.lower() == date_col.lower()]
        if not matches:
            raise ValueError(
                f"Date column '{date_col}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )
        col = matches[0]

    df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
    df = df.set_index(col).sort_index()
    df.index.name = "date"

    # Drop fully-empty columns and deduplicate dates.
    df = df.dropna(axis=1, how="all")
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_manifest(series_names: list[str], user_manifest: Path | None) -> pd.DataFrame:
    """Return a manifest DataFrame, merging user-provided rows with defaults."""
    if user_manifest is not None and user_manifest.exists():
        mf = pd.read_csv(user_manifest)
        required = {"name", "transform"}
        missing = required - set(mf.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")
        bad = set(mf["transform"]) - VALID_TRANSFORMS
        if bad:
            raise ValueError(f"Unknown transforms in manifest: {bad}")
        if "group" not in mf.columns:
            mf["group"] = "unknown"
        # Fill in any series not listed in the manifest with defaults.
        covered = set(mf["name"])
        extras = [n for n in series_names if n not in covered]
        if extras:
            print(f"  {len(extras)} series not in manifest → defaulting to transform=level")
            mf = pd.concat([
                mf,
                pd.DataFrame({"name": extras, "transform": "level", "group": "unknown"}),
            ], ignore_index=True)
    else:
        print("  No manifest provided → all series default to transform=level, group=unknown")
        print("  Edit data/macro/manifest.csv to set correct transforms before running the pipeline.")
        mf = pd.DataFrame({
            "name": series_names,
            "transform": "level",
            "group": "unknown",
        })

    mf = mf[mf["name"].isin(series_names)].reset_index(drop=True)
    return mf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to raw macro panel (CSV or Excel)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Optional CSV specifying name/transform/group per series")
    parser.add_argument("--date-col", default="date",
                        help="Name of the date column (default: 'date')")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR,
                        help=f"Output directory (default: {OUT_DIR})")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    panel = load_input(args.input, args.date_col)
    print(f"  {panel.shape[0]} dates × {panel.shape[1]} series "
          f"({panel.index.min().date()} → {panel.index.max().date()})")

    manifest = build_manifest(panel.columns.tolist(), args.manifest)

    out_panel = args.output_dir / "macro_panel_clean.csv"
    out_manifest = args.output_dir / "manifest.csv"

    panel.to_csv(out_panel)
    manifest.to_csv(out_manifest, index=False)

    print(f"\nWrote cleaned panel → {out_panel}  {panel.shape}")
    print(f"Wrote manifest      → {out_manifest}  ({len(manifest)} series)")
    print("\nNext step: run scripts/run_macro_mapping.py")


if __name__ == "__main__":
    main()
