"""Deep-analysis 07: the forecastability ceiling, decomposed link by link.

The chain the project needs for factor-state forecasting to work:

  (1) factors capture commodity return variance          [AE recon share]
  (2) some factor directions are macro-spanned           [rho^2 per CV dim]
  (3) the state carries over to t+1                      [AC(1) of v_k, u_k]
  (4) => tomorrow's return has a predictable component   [in-sample R^2 of
        r_{t+1} on state_t -- the with-look-ahead upper bound]
  (5) OOS estimation cost of K extra regressors          [~0.3-0.5% R^2]

If (4) << (5) the observed negative R2_OOS is arithmetic, not a bug. Also
computes the h=21/63 minimal detectable effects implied by effective sample
sizes, to bound what the long-horizon ladder could ever have shown.

Output: results/deep_analysis/da07_ceiling.csv (+ printed summary)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_ctx import load_ctx
from _scan_utils import h_fwd_returns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "deep_analysis"


def ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    """In-sample R^2 of y on [1, x] (full look-ahead upper bound)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1.0 - resid.var() / y.var()


def main() -> None:
    ctx = load_ctx(seed=0)
    r, state, t_split = ctx.returns, ctx.state, ctx.t_split
    T, N = r.shape
    K = state.shape[1]

    # ---- (1) factor content of returns: linear projection R^2 (train-fit)
    rz = (r - r[:t_split].mean(0)) / r[:t_split].std(0, ddof=0)
    F = ctx.F.to_numpy(float)
    X = np.column_stack([np.ones(T), F])
    beta, *_ = np.linalg.lstsq(X[:t_split], rz[:t_split], rcond=None)
    fit = X @ beta
    sh_train = 1 - ((rz - fit)[:t_split].var(0) / rz[:t_split].var(0)).mean()
    sh_oos = 1 - ((rz - fit)[t_split:].var(0) / rz[t_split:].var(0)).mean()
    print(f"(1) factor content: mean variance share captured by 5 latents "
          f"train={sh_train:.1%} OOS={sh_oos:.1%}")

    # ---- (2) macro-spanned share of each CV dim (train + frozen OOS)
    rho_tr = ctx.basis.rho_cv_train
    print(f"(2) macro-spanned share rho_cv^2 (train): {np.round(rho_tr**2, 3)}")

    # ---- (3) persistence of the state
    V = state
    M_u = ctx.basis.macro_variates(ctx.M_full).reindex(ctx.dates).to_numpy(float)
    rows = []
    for k in range(K):
        v = pd.Series(V[:, k])
        u = pd.Series(M_u[:, k])
        rows.append({
            "dim": f"dim{k + 1}", "rho_cv_train": rho_tr[k],
            "ac1_v": v.autocorr(1), "ac5_v": v.autocorr(5), "ac21_v": v.autocorr(21),
            "ac1_u": u.autocorr(1), "ac5_u": u.autocorr(5), "ac21_u": u.autocorr(21),
        })
    pers = pd.DataFrame(rows)
    print("(3) state persistence (autocorrelations):")
    print(pers.round(3).to_string(index=False))

    # ---- (4) in-sample ceilings with FULL look-ahead (pooled, standardized)
    ceil_rows = []
    for h in (1, 5, 21, 63):
        y = h_fwd_returns(rz, h)  # forward h-day sums of standardized returns
        valid = ~np.isnan(y).any(axis=1)
        yv, sv = y[valid], state[valid]
        r2_pool = float(np.mean([ols_r2(yv[:, i], sv) for i in range(N)])) / 1.0
        # also: state + own-lag (the actual model class)
        y_lag = np.vstack([np.full((h, N), np.nan), y[:-h]])[valid]
        ok = ~np.isnan(y_lag).any(axis=1)
        r2_full = float(np.mean([
            ols_r2(yv[ok][:, i], np.column_stack([y_lag[ok][:, i], sv[ok]])) for i in range(N)
        ]))
        ceil_rows.append({"horizon": h, "r2_insample_state": r2_pool,
                          "r2_insample_state_ownlag": r2_full})
        print(f"(4) h={h}: IN-SAMPLE pooled R^2 of fwd returns on state = {r2_pool:.4%} "
              f"(state+own-lag: {r2_full:.4%})")

    # ---- (5) MDE at long horizons: CW t needs ~2.8/sqrt(n_eff) one-sided at 0.5 power
    print("\n(5) power arithmetic (one-sided alpha=.05, power=.80, iid approx):")
    for h, n_eff in [(1, 1148), (5, 229), (21, 54), (63, 18)]:
        # detectable standardized mean shift delta = (z_a + z_b)/sqrt(n) of the
        # CW loss differential; translate to R^2 via delta ~ R2 * sd ratio ~ R2
        mde = (1.645 + 0.842) / np.sqrt(n_eff)
        print(f"    h={h:>2}: effective n={n_eff:>4} -> MDE ~ {mde:.1%} of loss sd "
              f"(daily-equivalent predictive R^2)")

    pd.DataFrame(ceil_rows).to_csv(OUT / "da07_ceiling.csv", index=False)
    pers.to_csv(OUT / "da07_state_persistence.csv", index=False)


if __name__ == "__main__":
    main()
