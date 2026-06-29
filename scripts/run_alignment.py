#!/usr/bin/env python3
"""E0: align AE latent factors across rolling windows and audit stability.

Consumes ``results/factor_bundle.npz`` (from extract_factors.py) and writes:
  * results/factors_aligned.csv        - aligned daily factor panel (E1-E4 input)
  * results/alignment_stability.csv    - per-window match quality + perm/sign
  * results/alignment_report.md        - gate verdict + summary metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.factor_alignment import (
    align_by_correlation,
    align_by_decoder_cosine,
    aligned_factor_panel,
    stability_report,
)
from src.factor_extraction import FactorBundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "results" / "factor_bundle.npz")
    p.add_argument("--reference", type=int, default=0, help="Reference window index w0.")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bundle = FactorBundle.load(args.bundle)
    print(f"Loaded bundle: {bundle.n_windows} windows, K={bundle.n_factors}", flush=True)

    corr_align = align_by_correlation(bundle, reference=args.reference)
    cos_align = align_by_decoder_cosine(bundle, reference=args.reference)
    report = stability_report(bundle, corr_align, cos_align)

    panel = aligned_factor_panel(bundle, corr_align)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output_dir / "factors_aligned.csv")

    stab = pd.DataFrame(
        {
            "forecast_date": bundle.forecast_dates,
            "as_of_date": bundle.as_of_dates,
            "corr_match_quality": corr_align.match_quality,
            "cosine_match_quality": cos_align.match_quality,
        }
    )
    for r in range(bundle.n_factors):
        stab[f"perm_f{r+1}"] = corr_align.perm[:, r]
        stab[f"sign_f{r+1}"] = corr_align.sign[:, r]
    stab.to_csv(args.output_dir / "alignment_stability.csv", index=False)

    lines = [
        "# E0: Cross-window factor alignment & stability audit\n",
        f"- Windows: **{report['n_windows']}**, factors K = **{report['n_factors']}**",
        f"- Reference window: {args.reference}",
        f"- **Median adjacent |corr| (anchor 1): {report['median_adjacent_abs_corr']:.3f}**",
        f"- Mean / min adjacent |corr|: {report['mean_adjacent_abs_corr']:.3f} / {report['min_adjacent_abs_corr']:.3f}",
        f"- Median decoder |cosine| (anchor 2): {report.get('median_decoder_cosine', float('nan')):.3f}",
        f"- Anchor agreement rate: {report.get('anchor_agreement_rate', float('nan')):.3f}",
        f"- Permutation-event rate: {report['permutation_event_rate']:.3f}",
        f"- Sign-flip rate per factor: "
        + ", ".join(f"f{r+1}={v:.3f}" for r, v in enumerate(report['sign_flip_rate_per_factor'])),
        "",
        f"## Gate verdict: **{report['verdict'].upper()}**",
        "",
    ]
    verdict_text = {
        "stable": "Median matched |corr| >= 0.8: factors have a stable identity. "
                  "Proceed with cross-window pooled mapping at full strength.",
        "within_window_only": "Median matched |corr| in [0.5, 0.8): identity is "
                  "borderline. Restrict mapping to within-window, or report with caveats.",
        "unstable_finding": "Median matched |corr| < 0.5: factors do NOT have a "
                  "stable identity. The AE coordinate system is not economically "
                  "persistent -- report the instability itself as a finding.",
    }
    lines.append(verdict_text[report["verdict"]])
    (args.output_dir / "alignment_report.md").write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "alignment_report.json").write_text(
        json.dumps({k: v for k, v in report.items()}, indent=2, default=float),
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\nWrote aligned panel -> {args.output_dir / 'factors_aligned.csv'}")


if __name__ == "__main__":
    main()
