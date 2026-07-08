"""Deep-analysis 01: data forensics on the commodity panel and macro panel.

Checks performed (all read-only):
  A. Commodity panel integrity: calendar coverage, staleness of the retained 21,
     return moments, extreme days, cross-sectional sanity correlations.
  B. Macro panel integrity: forward-fill detection (repeated-value fraction),
     calendar alignment with the commodity panel, missingness.
  C. Known-economics sanity anchors: WTI~Brent, gold~silver, gold~DXY(neg),
     copper~AUD, energy factor ~ xle. If these fail the data is wrong; if they
     hold the data is economically real and the forecast null is not a data
     mirage.

Outputs: results/deep_analysis/da01_*.csv and a printed summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqcp.io.commodities import SECTORS, load_return_panel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    panel = load_return_panel()
    r = pd.DataFrame(panel.raw, index=panel.dates, columns=panel.commodities)
    T, N = r.shape
    print(f"panel: T={T} N={N}  {r.index[0].date()} -> {r.index[-1].date()}")

    # ---- A. calendar
    dow = pd.Series(r.index.dayofweek).value_counts().sort_index()
    print("day-of-week counts (0=Mon):", dow.to_dict())
    gaps = pd.Series(r.index).diff().dt.days.dropna()
    big_gaps = (gaps > 4).sum()
    print(f"calendar gaps >4 days: {big_gaps} (max gap {gaps.max()} days)")

    # ---- A. staleness + moments
    rows = []
    for c in r.columns:
        x = r[c].to_numpy()
        zero = np.isclose(x, 0.0)
        run = best = 0
        for z in zero:
            run = run + 1 if z else 0
            best = max(best, run)
        rows.append(
            {
                "commodity": c,
                "sector": next((s for s, m in SECTORS.items() if c in m), "?"),
                "zero_frac": zero.mean(),
                "longest_zero_run": best,
                "ann_mean_pct": x.mean() * 252 * 100,
                "ann_vol_pct": x.std() * np.sqrt(252) * 100,
                "skew": pd.Series(x).skew(),
                "kurtosis": pd.Series(x).kurt(),
                "min_daily_pct": x.min() * 100,
                "max_daily_pct": x.max() * 100,
                "worst_day": str(r.index[np.argmin(x)].date()),
                "best_day": str(r.index[np.argmax(x)].date()),
                "ac1": pd.Series(x).autocorr(1),
            }
        )
    stats = pd.DataFrame(rows).sort_values("zero_frac", ascending=False)
    stats.to_csv(OUT / "da01_commodity_stats.csv", index=False)
    print("\nstalest 5 retained series:")
    print(stats.head(5)[["commodity", "zero_frac", "longest_zero_run", "ac1"]].to_string(index=False))
    print("\n|AC(1)| > 0.10:", stats.loc[stats.ac1.abs() > 0.10, ["commodity", "ac1"]].to_dict("records"))

    # ---- C. sanity anchors (contemporaneous, full sample)
    anchors = [
        ("WTI", "Brent", "+high"),
        ("Gold", "Silver", "+high"),
        ("Gasoline", "WTI", "+high"),
        ("Corn", "HRWWheat", "+mid"),
        ("Copper", "Aluminium", "+mid"),
    ]
    print("\ncommodity-commodity sanity correlations:")
    for a, b, expect in anchors:
        print(f"  corr({a},{b}) = {r[a].corr(r[b]):+.3f}   expected {expect}")

    # ---- B. macro panel
    m = pd.read_csv(ROOT / "data/processed/macro_stationary.csv", parse_dates=["date"], index_col="date")
    m = m.select_dtypes(include=[np.number])  # drop the regime label column
    print(f"\nmacro stationary: T={len(m)} J={m.shape[1]}  {m.index[0].date()} -> {m.index[-1].date()}")
    mrows = []
    for c in m.columns:
        x = m[c].dropna()
        rep = (x.diff() == 0).mean()  # fraction of days with unchanged transformed value
        zero = np.isclose(x, 0.0).mean()  # zero daily change => source was ffilled/stale
        mrows.append({"series": c, "n": len(x), "zero_change_frac": zero, "repeat_frac": rep, "ac1": x.autocorr(1)})
    mstats = pd.DataFrame(mrows).sort_values("zero_change_frac", ascending=False)
    mstats.to_csv(OUT / "da01_macro_stats.csv", index=False)
    print("macro series with >15% zero-change days (stale/ffilled):")
    print(mstats[mstats.zero_change_frac > 0.15].to_string(index=False))
    print("\nmacro series with |AC(1)|>0.15 (transformed should be ~white):")
    print(mstats[mstats.ac1.abs() > 0.15][["series", "ac1"]].to_string(index=False))

    # macro vs commodity calendar
    common = r.index.intersection(m.index)
    print(f"\ncommon calendar: {len(common)} of T_commodity={T}, T_macro={len(m)}")

    # cross-asset sanity: gold vs dollar, copper vs AUD (contemporaneous)
    j = r.join(m, how="inner")
    for a, b, expect in [
        ("Gold", "fx_dxy", "negative"),
        ("Copper", "fx_audusd", "positive"),
        ("WTI", "xle", "positive"),
        ("Gold", "tips_10y", "negative"),
    ]:
        if b in j.columns:
            print(f"  corr({a},{b}) = {j[a].corr(j[b]):+.3f}   expected {expect}")

    # ---- lag-1 cross-asset teaser (the actual forecasting question, univariate)
    print("\nlag-1 macro -> next-day commodity correlations (|rho| max over 37 macro x 21 comm):")
    lag = m.shift(1).reindex(j.index)
    best = ("", "", 0.0)
    vals = []
    for mc in m.columns:
        for cc in r.columns:
            v = j[cc].corr(lag[mc])
            vals.append(v)
            if abs(v) > abs(best[2]):
                best = (mc, cc, v)
    vals = np.array([v for v in vals if np.isfinite(v)])
    n_eff = len(j)
    print(f"  max |corr| = {best[2]:+.4f} ({best[0]} -> {best[1]}); 2/sqrt(T)={2 / np.sqrt(n_eff):.4f}")
    print(f"  count |corr|>2/sqrt(T): {(np.abs(vals) > 2 / np.sqrt(n_eff)).sum()} of {len(vals)} "
          f"(binomial expectation under null ~ {0.046 * len(vals):.0f})")


if __name__ == "__main__":
    main()
