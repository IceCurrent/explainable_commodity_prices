"""Deep-analysis 08: is the AE the bottleneck? PCA / raw-sector states head-to-head.

If the AE factor step destroyed forecastable content, a transparent linear
state (train-fit PCA) or simple sector means would outperform the AE state in
the identical engine. Under the 'nothing to find' diagnosis, every state
construction lands on the same null.

Output: results/deep_analysis/da08_state_comparison.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_ctx import load_ctx

from eqcp.io.commodities import SECTORS
from eqcp.forecasting.ar1 import AR_KEY, clark_west, expanding_subset_forecasts, r2_oos

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deep_analysis"


def main() -> None:
    ctx = load_ctx(seed=0)
    r, t_split = ctx.returns, ctx.t_split
    T, N = r.shape

    rz = (r - r[:t_split].mean(0)) / r[:t_split].std(0, ddof=0)

    # PCA state (train-fit, frozen loadings)
    cov = np.cov(rz[:t_split], rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    W = evecs[:, ::-1][:, :5]
    pca_state = rz @ W
    evr = evals[::-1][:5].sum() / evals.sum()

    # sector-mean state
    sec_state = np.column_stack([
        rz[:, [ctx.commodities.index(c) for c in members if c in ctx.commodities]].mean(axis=1)
        for members in SECTORS.values()
    ])

    arms = {
        "ae_V_state (pipeline)": ctx.state,
        "pca5_state": pca_state,
        "sector_mean_state": sec_state,
    }
    rows = []
    for name, st in arms.items():
        K = st.shape[1]
        full_key = tuple(range(K))
        fr = expanding_subset_forecasts(
            r, st, t_split, horizon=1, subsets=[AR_KEY, full_key], min_train=ctx.cfg.min_train,
        )
        r2 = r2_oos(fr, full_key, AR_KEY, ctx.weights)
        stat, p = clark_west(fr, full_key, AR_KEY, weights=ctx.weights)
        rows.append({"state": name, "K": K, "r2_oos_vs_ar1": r2, "cw_stat": stat, "cw_p": p})
        print(f"{name:>24}: K={K} R2_OOS={r2:+.5f} CW stat={stat:+.2f} p={p:.3f}")
    print(f"(PCA-5 explains {evr:.1%} of train variance)")
    pd.DataFrame(rows).to_csv(OUT / "da08_state_comparison.csv", index=False)


if __name__ == "__main__":
    main()
