"""KCCA nonlinearity verdict from stability sweep."""

from __future__ import annotations

import pandas as pd


def kcca_nonlinearity_verdict(
    stab: pd.DataFrame,
    canon_min: float,
    kcca_min: float,
    iqr_threshold: float = 0.15,
    linear_margin: float = 0.1,
) -> tuple[str, str]:
    """Return (verdict_label, explanation) for the linear vs kernel comparison.

    Nonlinearity is claimed only when ``kcca_min`` is stable across the reg/gamma
    sweep *and* materially above the linear minimum. Otherwise report inconclusive.
    """
    iqr = float(stab["kcca_min"].quantile(0.75) - stab["kcca_min"].quantile(0.25))
    stable = iqr < iqr_threshold
    above_linear = kcca_min > canon_min + linear_margin
    if stable and above_linear:
        return (
            "nonlinear",
            f"kcca_min is stable across the reg/gamma sweep (IQR={iqr:.3f} < {iqr_threshold}) "
            f"and exceeds canon_min by >{linear_margin} ({kcca_min:.3f} vs {canon_min:.3f}).",
        )
    reasons = []
    if not stable:
        reasons.append(f"kcca_min unstable across sweep (IQR={iqr:.3f} >= {iqr_threshold})")
    if not above_linear:
        reasons.append(
            f"kcca_min not materially above canon_min ({kcca_min:.3f} vs {canon_min:.3f})"
        )
    return ("inconclusive / degenerate", "; ".join(reasons) + ".")
