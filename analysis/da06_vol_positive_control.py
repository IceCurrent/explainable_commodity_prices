"""Deep-analysis 06: volatility-target positive control.

If daily commodity MEAN returns are unforecastable (EMH), the same data and
the same engine should still display textbook SECOND-MOMENT predictability
(volatility clustering). We feed |r| into the identical expanding FAAR engine:

    target   y^(h)_{t+h} = sum |r| over (t+1..t+h)   (realized abs-vol proxy)
    AR model y^(h)_{t+h} ~ [1, y^(h)_t]              (vol persistence)
    full     y^(h)_{t+h} ~ [1, y^(h)_t, state_t]     (does the factor state add?)

Expected under 'the pipeline works and the data are real':
  - AR(1)-in-vol crushes the expanding-mean benchmark (positive R2_OOS, tiny p)
  - the factor state may add a little on top (vol commonality / leverage)
Either way the pooled CW machinery must light up on the AR-vs-mean comparison;
if it cannot detect even vol clustering, the machinery or data are broken.

Output: results/deep_analysis/da06_vol_control.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_ctx import load_ctx

from eqcp.forecasting.ar1 import (
    AR_KEY,
    MEAN_KEY,
    clark_west,
    expanding_subset_forecasts,
    r2_oos,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deep_analysis"


def main() -> None:
    ctx = load_ctx(seed=0)
    absr = np.abs(ctx.returns)
    K = ctx.state.shape[1]
    full_key = tuple(range(K))
    var_train = (absr[: ctx.t_split] ** 2).mean(axis=0)
    weights = 1.0 / np.maximum(absr[: ctx.t_split].var(axis=0, ddof=0), 1e-12)
    weights = weights / weights.mean()

    rows = []
    for h in (1, 5, 21):
        fr = expanding_subset_forecasts(
            absr, ctx.state, ctx.t_split, horizon=h,
            subsets=[AR_KEY, full_key], min_train=ctx.cfg.min_train,
        )
        r2_ar_vs_mean = r2_oos(fr, AR_KEY, MEAN_KEY, weights)
        st_ar, p_ar = clark_west(fr, AR_KEY, MEAN_KEY, weights=weights)
        r2_full_vs_ar = r2_oos(fr, full_key, AR_KEY, weights)
        st_f, p_f = clark_west(fr, full_key, AR_KEY, weights=weights)
        rows.append({
            "horizon": h,
            "r2_oos_ar_vs_mean": r2_ar_vs_mean, "cw_p_ar_vs_mean": p_ar,
            "r2_oos_full_vs_ar": r2_full_vs_ar, "cw_p_full_vs_ar": p_f,
            "n_origins": len(fr.origins),
        })
        print(f"h={h}: VOL AR-vs-MEAN R2_OOS={r2_ar_vs_mean:+.4f} (CW p={p_ar:.2e}) | "
              f"state adds: R2={r2_full_vs_ar:+.4f} (CW p={p_f:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "da06_vol_control.csv", index=False)

    # mean-return contrast on identical machinery for the report
    fr_m = expanding_subset_forecasts(
        ctx.returns, ctx.state, ctx.t_split, horizon=1,
        subsets=[AR_KEY, full_key], min_train=ctx.cfg.min_train,
    )
    r2_m = r2_oos(fr_m, full_key, AR_KEY, ctx.weights)
    _, p_m = clark_west(fr_m, full_key, AR_KEY, weights=ctx.weights)
    print(f"\ncontrast, MEAN target h=1: R2_OOS={r2_m:+.5f} CW p={p_m:.3f} (the null)")


if __name__ == "__main__":
    main()
