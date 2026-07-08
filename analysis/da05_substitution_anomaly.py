"""Deep-analysis 05: why does the all-dims macro-substitution arm get CW p~0.01?

The substitution arm replaces canonical-variate state dims v_k (functions of
same-day commodity settlements) by rho_k * u_k (functions of same-day MACRO
closes). US macro closes 1.5-5h AFTER the commodity settlements, so u_k
mechanically contains information from inside the day-(t+1) commodity return
window. Hypothesis: the substitution arm's CW significance is that overlap.

Tests:
  1. Replicate the pipeline's substitution arms (none / spanned / all).
  2. EMBARGO arm: shift the macro variates one extra business day. If CW
     significance dies, the anomaly is timing overlap, not slow macro
     transmission (which a 24h embargo would only dampen, not erase).
  3. Fingerprint: per-commodity CW p of the 'all' arm - overlap predicts the
     LME/COMEX metals (earliest settles) reject hardest.
  4. Placebo: jointly circular-shift the macro panel and recompute the 'all'
     arm CW stat 100 times - calibrated p for the real stat.

Output: results/deep_analysis/da05_substitution.csv (+ printed verdict)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_ctx import load_ctx
from _scan_utils import SETTLE_ET

from eqcp.forecasting.ar1 import (
    AR_KEY,
    clark_west,
    clark_west_per_commodity,
    expanding_subset_forecasts,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deep_analysis"


def run_arm(ctx, state_sub, macro_avail, horizon=1):
    K = state_sub.shape[1]
    full_key = tuple(range(K))
    fr = expanding_subset_forecasts(
        ctx.returns, state_sub, ctx.t_split, horizon=horizon,
        subsets=[AR_KEY, full_key], available=macro_avail, min_train=ctx.cfg.min_train,
    )
    v = fr.mse(AR_KEY, ctx.weights) - fr.mse(full_key, ctx.weights)
    stat, p = clark_west(fr, full_key, AR_KEY, weights=ctx.weights)
    return fr, v, stat, p


def main() -> None:
    ctx = load_ctx(seed=0)
    basis, state = ctx.basis, ctx.state
    K = state.shape[1]
    full_key = tuple(range(K))

    # --- macro variates exactly as the pipeline builds them
    M_sub = ctx.M_full.copy()
    for col in ctx.cfg.lagged_macro_series:
        if col in M_sub.columns:
            M_sub[col] = M_sub[col].shift(1)
    M_sub = M_sub.dropna(how="any")
    U = basis.macro_variates(M_sub).reindex(ctx.dates)
    macro_avail = U.notna().all(axis=1).to_numpy()
    U_arr = np.nan_to_num(U.to_numpy(float), nan=0.0)
    rho = basis.rho_train

    def substituted(u_source: np.ndarray, dims) -> np.ndarray:
        s = state.copy()
        for k in dims:
            s[:, k] = rho[k] * u_source[:, k]
        return s

    # arm 'none' and 'all' (pipeline replication)
    _, v_none, st_none, p_none = run_arm(ctx, state, macro_avail)
    fr_all, v_all, st_all, p_all = run_arm(ctx, substituted(U_arr, range(K)), macro_avail)
    print(f"replication: none v={v_none:+.3e} p={p_none:.4f} | all v={v_all:+.3e} "
          f"CW stat={st_all:+.3f} p={p_all:.4f}  (pipeline reported ~0.016)")

    # --- 2. one-extra-day embargo on the macro variates
    U_lag = pd.DataFrame(U_arr, index=ctx.dates).shift(1).fillna(0.0).to_numpy()
    avail_lag = np.roll(macro_avail, 1)
    avail_lag[0] = False
    _, v_l, st_l, p_l = run_arm(ctx, substituted(U_lag, range(K)), avail_lag & macro_avail)
    print(f"embargoed 'all' arm (+1 day): v={v_l:+.3e} CW stat={st_l:+.3f} p={p_l:.4f}")

    # --- 3. per-commodity fingerprint of the same-day 'all' arm
    cw_stat_i, cw_p_i = clark_west_per_commodity(fr_all, full_key, AR_KEY)
    fp = pd.DataFrame({
        "commodity": ctx.commodities,
        "settle_et": [SETTLE_ET[c] for c in ctx.commodities],
        "after_settle_hours": [17.0 - SETTLE_ET[c] for c in ctx.commodities],
        "cw_stat": cw_stat_i,
        "cw_p": cw_p_i,
    }).sort_values("cw_p")
    rho_fp = np.corrcoef(fp["after_settle_hours"], fp["cw_stat"])[0, 1]
    print(f"\nfingerprint: corr(after-settle hours, per-commodity CW stat) = {rho_fp:+.3f}")
    print(fp.head(8).to_string(index=False))

    # --- 4. placebo calibration of the 'all' arm CW stat
    rng = np.random.default_rng(7)
    T = len(ctx.dates)
    stats_plc = []
    for i in range(100):
        shift = int(rng.integers(63, T - 63))
        U_s = np.roll(U_arr, shift, axis=0)
        avail_s = np.roll(macro_avail, shift)
        _, _, st_s, _ = run_arm(ctx, substituted(U_s, range(K)), avail_s & macro_avail)
        stats_plc.append(st_s)
    arr = np.array(stats_plc)
    p_cal = (1 + (arr >= st_all).sum()) / (len(arr) + 1)
    print(f"\nplacebo-calibrated p for same-day 'all' CW stat: {p_cal:.3f} "
          f"(placebo stat mean={arr.mean():+.2f}, p95={np.percentile(arr, 95):+.2f})")

    rows = [
        {"arm": "none", "v": v_none, "cw_stat": st_none, "cw_p": p_none},
        {"arm": "all_same_day", "v": v_all, "cw_stat": st_all, "cw_p": p_all,
         "cw_p_placebo_cal": p_cal},
        {"arm": "all_embargo_1d", "v": v_l, "cw_stat": st_l, "cw_p": p_l},
    ]
    pd.DataFrame(rows).to_csv(OUT / "da05_substitution.csv", index=False)
    fp.to_csv(OUT / "da05_substitution_fingerprint.csv", index=False)


if __name__ == "__main__":
    main()
