"""Deep-analysis 03: is the lag-1 macro signal real diffusion or closing-time overlap?

Three discriminating tests:

1. EMBARGO. Repeat the calibrated h=1 scan predicting r_{t+1} from m_{t-1}
   (a 24h information embargo). Mechanical overlap dies instantly; genuine
   multi-day information diffusion decays smoothly.

2. SETTLE-TIME FINGERPRINT. For after-US-close predictors (FX/rates/vol/equity),
   overlap size = hours between the commodity's settle and the predictor's
   close. Regress each commodity's mean |t| (over 'after' predictors) on its
   after-settle window length. Overlap => strong positive relation
   (LME metals noon settle >> CME energy 14:30).

3. BEFORE-CLOSERS. Predictors determined before US commodity settlements
   (Chinese equities, China 10y, Baltic freight) cannot overlap; their
   rejection excess over placebo measures genuine next-day diffusion.

Outputs: results/deep_analysis/da03_*.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _scan_utils import MACRO_CLOSE_GROUP, SETTLE_ET, h_fwd_returns, tmat
from eqcp.io.commodities import load_return_panel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def calibrated_count(m_arr, y, lags, seed, n_placebo=200, thresh=1.96):
    """Real rejection count and placebo distribution under joint circular shifts."""
    t_real = tmat(m_arr, y, lags)
    real = int((np.abs(t_real) > thresh).sum())
    rng = np.random.default_rng(seed)
    T = m_arr.shape[0]
    counts = []
    for _ in range(n_placebo):
        shift = int(rng.integers(63, T - 63))
        counts.append(int((np.abs(tmat(np.roll(m_arr, shift, axis=0), y, lags)) > thresh).sum()))
    arr = np.array(counts)
    p = (1 + (arr >= real).sum()) / (n_placebo + 1)
    return t_real, real, arr, p


def main() -> None:
    panel = load_return_panel()
    r = pd.DataFrame(panel.raw, index=panel.dates, columns=panel.commodities)
    m = pd.read_csv(ROOT / "data/processed/macro_stationary.csv", parse_dates=["date"], index_col="date")
    m = m.select_dtypes(include=[np.number])
    j = r.join(m, how="inner").dropna()
    rr = j[r.columns].to_numpy(float)
    mm = j[m.columns].to_numpy(float)
    y1 = h_fwd_returns(rr, 1)

    # ---------- 1. embargo scan: m_{t-1} -> r_{t+1}
    mm_lag = np.vstack([np.full((1, mm.shape[1]), np.nan), mm[:-1]])
    valid = ~np.isnan(mm_lag).any(axis=1)
    t_emb, real_emb, plc_emb, p_emb = calibrated_count(mm_lag[valid], y1[valid], lags=5, seed=1)
    t_lag0, real_lag0, plc_lag0, p_lag0 = calibrated_count(mm[valid], y1[valid], lags=5, seed=2)
    print("h=1 scan, same-day predictor m_t:    "
          f"real={real_lag0} placebo_mean={plc_lag0.mean():.1f} p={p_lag0:.3f}")
    print("h=1 scan, embargoed predictor m_t-1: "
          f"real={real_emb} placebo_mean={plc_emb.mean():.1f} p={p_emb:.3f}")
    pd.DataFrame({
        "arm": ["same_day", "embargo_1d"],
        "real_count": [real_lag0, real_emb],
        "placebo_mean": [plc_lag0.mean(), plc_emb.mean()],
        "placebo_p95": [np.percentile(plc_lag0, 95), np.percentile(plc_emb, 95)],
        "p_value": [p_lag0, p_emb],
    }).to_csv(OUT / "da03_embargo.csv", index=False)

    # ---------- 2. settle-time fingerprint (same-day scan, 'after' predictors)
    after_idx = [k for k, c in enumerate(m.columns) if MACRO_CLOSE_GROUP.get(c) == "after"]
    mean_abs_t = np.abs(t_lag0[after_idx]).mean(axis=0)  # (N,) per commodity
    hours_open = np.array([17.0 - SETTLE_ET[c] for c in r.columns])  # window to 5pm FX close
    fp = pd.DataFrame({
        "commodity": r.columns,
        "settle_et": [SETTLE_ET[c] for c in r.columns],
        "after_settle_hours": hours_open,
        "mean_abs_t_after_predictors": mean_abs_t,
    }).sort_values("after_settle_hours", ascending=False)
    rho = np.corrcoef(hours_open, mean_abs_t)[0, 1]
    fp.to_csv(OUT / "da03_settle_fingerprint.csv", index=False)
    print("\nsettle-time fingerprint: corr(after-settle hours, mean |t|) = "
          f"{rho:+.3f} (overlap predicts strongly positive)")
    print(fp.to_string(index=False))

    # ---------- 3. before-closers only
    before_idx = [k for k, c in enumerate(m.columns) if MACRO_CLOSE_GROUP.get(c) == "before"]
    names_b = [m.columns[k] for k in before_idx]
    t_b = t_lag0[before_idx]
    real_b = int((np.abs(t_b) > 1.96).sum())
    rng = np.random.default_rng(3)
    counts = []
    mm_b = mm[valid][:, before_idx]
    for _ in range(200):
        shift = int(rng.integers(63, len(mm_b) - 63))
        counts.append(int((np.abs(tmat(np.roll(mm_b, shift, axis=0), y1[valid], 5)) > 1.96).sum()))
    arr = np.array(counts)
    p_b = (1 + (arr >= real_b).sum()) / 201
    print(f"\nbefore-closers ({names_b}): real={real_b} of {len(before_idx) * rr.shape[1]}, "
          f"placebo mean={arr.mean():.1f} p95={np.percentile(arr, 95):.0f} p={p_b:.3f}")

    # per-group real counts for the record
    rows = []
    for grp in ("after", "before", "ambiguous"):
        idx = [k for k, c in enumerate(m.columns) if MACRO_CLOSE_GROUP.get(c) == grp]
        rows.append({
            "group": grp, "n_series": len(idx),
            "n_reject_h1": int((np.abs(t_lag0[idx]) > 1.96).sum()),
            "n_pairs": len(idx) * rr.shape[1],
            "reject_rate": float((np.abs(t_lag0[idx]) > 1.96).mean()),
            "n_reject_embargo": int((np.abs(t_emb[idx]) > 1.96).sum()),
            "reject_rate_embargo": float((np.abs(t_emb[idx]) > 1.96).mean()),
        })
    grp_df = pd.DataFrame(rows)
    grp_df.to_csv(OUT / "da03_group_counts.csv", index=False)
    print("\nrejection rates by predictor close-time group (same-day vs embargoed):")
    print(grp_df.to_string(index=False))


if __name__ == "__main__":
    main()
