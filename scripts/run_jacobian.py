#!/usr/bin/env python3
"""E3: decoder-Jacobian commodity fingerprint and sector -> macro hypothesis.

Consumes ``results/factor_bundle.npz``; aligns decoder loadings (E0) and writes:
  * results/jacobian_loadings.csv    - mean aligned loading of each commodity x factor
  * results/jacobian_fingerprint.md  - top commodities, sector mass, macro hypothesis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factor_alignment import align_by_correlation
from src.factor_extraction import FactorBundle
from src.jacobian import compute_fingerprint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "results" / "factor_bundle.npz")
    p.add_argument("--reference", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = FactorBundle.load(args.bundle)
    alignment = align_by_correlation(bundle, reference=args.reference)
    fp = compute_fingerprint(bundle, alignment)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fp.loadings_frame().to_csv(args.output_dir / "jacobian_loadings.csv")

    lines = ["# E3: Decoder-Jacobian commodity fingerprint\n"]
    lines.append("Mean decoder loadings aligned to the E0 reference identity. "
                 "Sector mass = share of absolute loading per sector.\n")
    lines.append("## Sector mass per factor (share of total |loading|)\n")
    lines.append(fp.sector_mass.round(3).to_markdown())
    lines.append("")
    lines.append("## Sector intensity per factor (mean |loading| per commodity)\n")
    lines.append(fp.sector_intensity.round(3).to_markdown())
    lines.append("")
    for k, f in enumerate(fp.factors):
        lines.append(f"## Factor {f}")
        lines.append(f"- Dominant sector: **{fp.dominant_sector[f]}** "
                     f"(top-2 concentration {fp.concentration[f]:.2f})")
        lines.append(f"- Macro hypothesis: {fp.macro_hypothesis[f]}")
        lines.append("- Top loadings:")
        lines.append(fp.top_loadings(k, n=5).to_markdown(index=False))
        lines.append("")
    (args.output_dir / "jacobian_fingerprint.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"Wrote -> {args.output_dir / 'jacobian_loadings.csv'}")


if __name__ == "__main__":
    main()
