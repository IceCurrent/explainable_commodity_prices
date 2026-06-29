#!/usr/bin/env python3
"""E1/E2: map aligned AE factors to the observable macro panel.

Prerequisites
-------------
1. Factor bundle:
     python scripts/extract_factors.py          → results/factor_bundle.npz
2. Cleaned macro panel in data/macro/:
     python data/clean_macro_panel.py --input <your_file>
   Then review/edit data/macro/manifest.csv to set correct transforms.

Experiments run
---------------
  E1   HAC-OLS spanning regression + Bai-Ng canonical-correlation spanning test
  E1.4 Incremental-content test: f vs macro-spanned f_hat vs orthogonal residual u
  E2   Nonlinear (GBT) mapping + SHAP + nonlinearity premium

Outputs (results/)
------------------
  e1_spanning_summary.csv        per-factor R² and dominant macro series
  e1_spanning_coefficients.csv   standardised HAC-OLS betas (macro × factor)
  e1_bai_ng_spanning.json        canonical correlations + spanning verdict
  e1_incremental_rmse.csv        RMSE for original / spanned / residual factors
  e1_incremental_shapley.csv     forecast-Shapley for each variant
  e2_nonlinear_summary.csv       GBT R², linear R², nonlinearity premium, top SHAP
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position,import-error
import pandas as pd

from src.data import load_return_panel
from src.factor_alignment import align_by_correlation, aligned_factor_panel
from src.factor_extraction import FactorBundle
from src.macro_data import load_macro_panel
from src.macro_mapping import (
    bai_ng_spanning_summary,
    incremental_content,
    nonlinear_macro_mapping,
    spanning_regression,
)
# pylint: enable=wrong-import-position,import-error


def parse_args() -> argparse.Namespace:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bundle", type=Path,
        default=PROJECT_ROOT / "results" / "factor_bundle.npz",
        help="Path to factor_bundle.npz (default: results/factor_bundle.npz)",
    )
    parser.add_argument(
        "--reference", type=int, default=0,
        help="Reference window index for factor alignment (default: 0)",
    )
    parser.add_argument(
        "--include-groups", nargs="*", default=None,
        help="Restrict macro panel to these manifest groups (e.g. FX Rates Risk)",
    )
    parser.add_argument(
        "--exclude-series", nargs="*", default=None,
        help="Drop these series from the macro panel",
    )
    parser.add_argument(
        "--skip-incremental", action="store_true",
        help="Skip the (slower) incremental-content re-forecast (E1.4)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results",
    )
    return parser.parse_args()


def _run_e1(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    output_dir: Path,
) -> None:
    """E1: HAC-OLS spanning regression and Bai-Ng CCA test."""
    print("\n[E1] Spanning regression ...")
    reg = spanning_regression(factors, macro)
    span_rows = [
        {
            "factor": fname,
            "r2": result.r2,
            "hac_lags": result.hac_lags,
            "n_obs": result.n_obs,
            "dominant": ", ".join(result.dominant[:3]),
        }
        for fname, result in reg.items()
    ]
    pd.DataFrame(span_rows).to_csv(
        output_dir / "e1_spanning_summary.csv", index=False
    )
    coef = pd.DataFrame({fname: r.coefficients for fname, r in reg.items()})
    coef.to_csv(output_dir / "e1_spanning_coefficients.csv")

    print("[E1] Bai-Ng canonical-correlation spanning test ...")
    bn_result = bai_ng_spanning_summary(factors, macro)
    (output_dir / "e1_bai_ng_spanning.json").write_text(
        json.dumps(bn_result, indent=2)
    )
    ccs = [round(c, 3) for c in bn_result["canonical_correlations"]]
    print(f"  Canonical correlations: {ccs}")
    print(
        f"  Min: {bn_result['min_canonical_corr']:.3f}"
        "  (1.0 = fully spanned by macro)"
    )


def _run_e1_incremental(  # pylint: disable=too-many-arguments
    bundle: FactorBundle,
    alignment,
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    panel,
    output_dir: Path,
) -> None:
    """E1.4: incremental-content decomposition (original / spanned / residual)."""
    print("[E1.4] Incremental-content decomposition ...")
    inc = incremental_content(bundle, alignment, factors, macro, panel=panel)
    inc.rmse.to_csv(output_dir / "e1_incremental_rmse.csv")
    inc.shapley.to_csv(output_dir / "e1_incremental_shapley.csv")
    share = inc.spanned_share_of_value
    if math.isnan(share):
        print("  Factors add ~no OOS forecast value; macro-spanned share undefined.")
    else:
        print(f"  Macro-spanned part retains {share:.1%} of original forecast value.")


def _run_e2(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    output_dir: Path,
) -> None:
    """E2: nonlinear macro mapping (GBT + SHAP + nonlinearity premium)."""
    print("[E2] Nonlinear macro mapping (GBT + SHAP) ...")
    nl_results = nonlinear_macro_mapping(factors, macro)
    nl_rows = [
        {
            "factor": fname,
            "r2_nonlinear": res.r2_nonlinear,
            "r2_linear": res.r2_linear,
            "nonlinearity_premium": res.nonlinearity_premium,
            "top_macro": ", ".join(res.shap_importance.head(3).index),
        }
        for fname, res in nl_results.items()
    ]
    pd.DataFrame(nl_rows).to_csv(
        output_dir / "e2_nonlinear_summary.csv", index=False
    )


def main() -> None:
    """Orchestrate E1/E1.4/E2 macro-mapping experiments."""
    args = parse_args()

    if not args.bundle.exists():
        print(f"Factor bundle not found: {args.bundle}")
        print("Run: python scripts/extract_factors.py")
        sys.exit(1)

    bundle = FactorBundle.load(args.bundle)
    panel = load_return_panel()
    alignment = align_by_correlation(bundle, reference=args.reference)
    factors = aligned_factor_panel(bundle, alignment)

    try:
        macro_panel = load_macro_panel(
            target_index=factors.index,
            include_groups=args.include_groups,
            exclude_series=args.exclude_series,
        )
    except FileNotFoundError as exc:
        print(f"\nMacro panel not found:\n  {exc}")
        print("\nSteps to fix:")
        print("  1. python data/clean_macro_panel.py --input <your_macro_file>")
        print("  2. Review data/macro/manifest.csv and set correct transforms")
        print("  3. Re-run this script")
        sys.exit(1)

    macro = macro_panel.stationary
    factors = factors.reindex(macro.index).dropna()
    macro = macro.loc[factors.index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"E1/E2: {len(factors)} dates | "
        f"{factors.shape[1]} factors | "
        f"{macro.shape[1]} macro series"
    )
    print(f"Macro groups: {sorted(macro_panel.manifest['group'].unique())}")

    _run_e1(factors, macro, args.output_dir)

    if not args.skip_incremental:
        _run_e1_incremental(bundle, alignment, factors, macro, panel, args.output_dir)

    _run_e2(factors, macro, args.output_dir)

    print(f"\nDone. Results written to {args.output_dir}/")


if __name__ == "__main__":
    main()
