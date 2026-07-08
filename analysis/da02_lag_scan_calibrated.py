"""Deep-analysis 02: is there ANY lagged macro -> commodity signal, honestly tested?

Two parts:

A. Date-shift audit. A stale or late-recorded predictor manufactures fake lag-1
   "predictability". For each suspicious macro series, profile its correlation
   with SPX returns at offsets -3..+3. A series whose |corr| with SPX peaks at
   offset +1 (today's recorded value co-moves with TOMORROW's SPX) is recorded a
   day late in the workbook; any lag-1 forecast built on it is mechanical.

B. Calibrated predictive scan. All 37 macro series x 21 commodities x
   h in {1,5,21}: predictive regression r_{i,t+1..t+h} ~ m_{j,t} with
   Newey-West t-stats (bandwidth 2h). The null count of |t|>1.96 rejections is
   calibrated by jointly circular-shifting the ENTIRE macro panel (preserving
   every cross- and auto-correlation within each panel) 200 times. The excess
   of real rejections over the placebo distribution is the honest measure of
   lagged macro signal, independent of the AE/CCA machinery.

Outputs: results/deep_analysis/da02_*.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqcp.io.commodities import load_return_panel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)

SUSPECTS = ["ig_oas", "hy_oas", "bdiy", "bdti", "gpr", "epu", "vix", "move", "csi300", "shcomp"]


def nw_tstat(x: np.ndarray, y: np.ndarray, lags: int) -> float:
    """Newey-West t-stat of slope in y = a + b x (both demeaned inside)."""
    x = x - x.mean()
    y = y - y.mean()
    sxx = float(x @ x)
    if sxx <= 0:
        return 0.0
    b = float(x @ y) / sxx
    u = y - b * x
    z = x * u
    T = len(z)
    v = float(z @ z) / T
    for lag in range(1, min(lags, T - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        v += 2.0 * w * float(z[:-lag] @ z[lag:]) / T
    se = np.sqrt(v / T) / (sxx / T)
    return b / se if se > 0 else 0.0


def h_fwd_returns(r: np.ndarray, h: int) -> np.ndarray:
    """Forward h-day cumulative return starting at t+1: rows t -> sum(t+1..t+h)."""
    T, N = r.shape
    out = np.full((T, N), np.nan)
    c = np.cumsum(r, axis=0)
    out[: T - h] = c[h:] - np.where(np.arange(1, T - h + 1)[:, None] > 0, np.vstack([np.zeros((1, N)), c[: T - h - 1]]), 0)
    # simpler: out[t] = c[t+h] - c[t]
    out = np.full((T, N), np.nan)
    out[: T - h] = c[h:] - c[:-h]
    return out


def scan_counts(m_arr: np.ndarray, fwd: dict[int, np.ndarray], thresh: float = 1.96) -> dict[int, np.ndarray]:
    """|NW-t| rejection matrix (J x N) for each horizon."""
    res = {}
    for h, y in fwd.items():
        J = m_arr.shape[1]
        N = y.shape[1]
        tmat = np.zeros((J, N))
        valid = ~np.isnan(y[:, 0])
        for j in range(J):
            xj = m_arr[valid, j]
            for i in range(N):
                tmat[j, i] = nw_tstat(xj, y[valid, i], lags=max(2 * h, 5))
        res[h] = np.abs(tmat) > thresh
        res[f"t_{h}"] = tmat
    return res


def main() -> None:
    panel = load_return_panel()
    r = pd.DataFrame(panel.raw, index=panel.dates, columns=panel.commodities)
    m = pd.read_csv(ROOT / "data/processed/macro_stationary.csv", parse_dates=["date"], index_col="date")
    m = m.select_dtypes(include=[np.number])
    j = r.join(m, how="inner").dropna()
    rr = j[r.columns].to_numpy(float)
    mm = j[m.columns].to_numpy(float)
    T = len(j)
    print(f"aligned T={T}, N={rr.shape[1]}, J={mm.shape[1]}")

    # ---------- A. date-shift audit vs SPX
    spx = j["spx"].to_numpy(float)
    rows = []
    for s in SUSPECTS:
        x = j[s].to_numpy(float)
        prof = {}
        for k in range(-3, 4):
            if k >= 0:
                c = np.corrcoef(x[: T - k], spx[k:])[0, 1]  # x_t vs spx_{t+k}
            else:
                c = np.corrcoef(x[-k:], spx[: T + k])[0, 1]
            prof[k] = c
        peak = max(prof, key=lambda k: abs(prof[k]))
        rows.append({"series": s, **{f"corr_spx_t{k:+d}": prof[k] for k in range(-3, 4)},
                     "peak_offset": peak, "peak_corr": prof[peak]})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "da02_dateshift_audit.csv", index=False)
    print("\ndate-shift audit (peak offset of corr with SPX; +1 = recorded a day LATE):")
    print(audit[["series", "corr_spx_t-1", "corr_spx_t+0", "corr_spx_t+1", "peak_offset", "peak_corr"]]
          .to_string(index=False))

    # ---------- B. calibrated scan
    horizons = [1, 5, 21]
    fwd = {h: h_fwd_returns(rr, h) for h in horizons}
    real = scan_counts(mm, fwd)
    real_counts = {h: int(real[h].sum()) for h in horizons}
    print("\nreal rejection counts (|NW t|>1.96) out of", mm.shape[1] * rr.shape[1], "pairs:")
    print("  ", real_counts)

    rng = np.random.default_rng(0)
    n_placebo = 200
    plc_counts = {h: [] for h in horizons}
    for _ in range(n_placebo):
        shift = int(rng.integers(63, T - 63))
        m_s = np.roll(mm, shift, axis=0)
        plc = scan_counts(m_s, fwd)
        for h in horizons:
            plc_counts[h].append(int(plc[h].sum()))
    out_rows = []
    for h in horizons:
        arr = np.array(plc_counts[h])
        p = (1 + (arr >= real_counts[h]).sum()) / (n_placebo + 1)
        out_rows.append({"horizon": h, "real_count": real_counts[h], "placebo_mean": arr.mean(),
                         "placebo_p95": np.percentile(arr, 95), "placebo_max": arr.max(), "p_value": p})
        print(f"h={h}: real={real_counts[h]}  placebo mean={arr.mean():.1f} p95={np.percentile(arr, 95):.0f} "
              f"max={arr.max()}  P(count>=real)={p:.3f}")
    pd.DataFrame(out_rows).to_csv(OUT / "da02_scan_calibration.csv", index=False)

    # per-macro-series localization at h=1
    per_series = pd.DataFrame({
        "series": m.columns,
        "n_reject_h1": real[1].sum(axis=1),
        "mean_abs_t_h1": np.abs(real["t_1"]).mean(axis=1),
    }).sort_values("n_reject_h1", ascending=False)
    per_series.to_csv(OUT / "da02_per_series_h1.csv", index=False)
    print("\ntop predictor series by h=1 rejection count:")
    print(per_series.head(8).to_string(index=False))

    # save full t-matrix for h=1
    pd.DataFrame(real["t_1"], index=m.columns, columns=r.columns).to_csv(OUT / "da02_tmat_h1.csv")


if __name__ == "__main__":
    main()
