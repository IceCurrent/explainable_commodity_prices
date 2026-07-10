"""Rolling-window walk-forward forecasting with per-window re-explanation.

The old pipeline froze one AE + one CCA basis on the first 60% of the sample
and only refit linear coefficients. This engine instead re-fits *everything* on
each rolling window and asks the harder question: does explainability survive?

For each window ``w`` with a ``train_window``-day train block followed by a
``test_block``-day out-of-sample block:

1. z-score returns on train-window statistics, retrain the AE on the train
   block, and encode train+test days through the just-trained encoder;
2. refit the linear CCA basis on ``(F_train, M_train)``, ordering canonical
   dims by train canonical correlation (descending) and sign-anchoring each on
   its dominant macro structure correlation so ranks are comparable across
   windows;
3. project all days onto the window's canonical-variate state and fit a direct
   h-step factor-augmented AR for every factor subset on the train block;
4. forecast the ``test_block`` origins and stash the losses.

The AE latent coordinates rotate window to window (gauge + optimizer
multiplicity + genuine regime drift), so the *factor side* is not comparable
across windows. The *macro side* is: every window's canonical vectors live in
the same fixed macro-series space. We therefore judge rank identity by the
cross-window cosine of each dim's 37-dim macro structure-correlation vector —
if rank ``k`` keeps pointing at the same macro combination, rank-pooled Shapley
attribution over all windows is legitimate; if it shatters, that is the
disclosed limitation of marrying rolling windows to explainability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from eqcp.cca.inference import purged_cv_canon
from eqcp.cca.linear import align_xy, linear_cca_full
from eqcp.config import FactorModelConfig, RollingForecastConfig
from eqcp.factors.extract import FactorModelType, train_factors
from eqcp.forecasting.ar1 import (
    AR_KEY,
    MEAN_KEY,
    ZERO_KEY,
    SubsetForecasts,
    all_subsets,
    h_period_returns,
)

SIGN_ANCHOR_FLOOR = 0.10


@dataclass
class WindowBasis:
    """Per-window CCA basis (train-only), with ranks sorted by canonical corr."""

    A: np.ndarray  # (K, K) factor-side canonical vectors (columns)
    B: np.ndarray  # (J, K) macro-side canonical vectors
    mu_f: np.ndarray  # (K,) train factor mean
    mu_m: np.ndarray  # (J,) train macro mean
    rho_train: np.ndarray  # (K,) train canonical correlations (descending)
    struct_macro: np.ndarray  # (J, K) macro structure correlations (fixed macro space)
    macro_cols: list[str]
    n_spanned: int
    ridge_m: float


@dataclass
class RollingResult:
    """Everything the rolling roll produces, ready for attribution + reporting."""

    forecasts: dict[int, SubsetForecasts]  # horizon -> pooled OOS forecasts
    windows: pd.DataFrame  # per-window canonical diagnostics
    struct_macro: np.ndarray  # (n_windows, J, K) macro structure-corr per window
    macro_cols: list[str]
    n_dims: int
    n_windows: int
    dates: pd.DatetimeIndex
    commodities: list[str]
    spanned_by_window: np.ndarray  # (n_windows,) leading-block size per window
    meta: dict = field(default_factory=dict)


def _zscore_train(train: np.ndarray, full: np.ndarray) -> np.ndarray:
    """z-score ``full`` using the mean/std of ``train`` only (no lookahead)."""
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, ddof=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (full - mu) / sd


def _spanned_boundary(rho: np.ndarray, rho_min: float) -> int:
    """Leading block = largest adjacent gap in the rho spectrum, else threshold."""
    gaps = rho[:-1] - rho[1:]
    if len(gaps) and float(np.max(gaps)) >= 0.1:
        return int(np.argmax(gaps)) + 1
    return max(int(np.sum(rho > rho_min)), 1)


def _fit_window_basis(
    F_train: pd.DataFrame,
    M_train: pd.DataFrame,
    ridge_grid: tuple[float, ...],
    n_folds: int,
    embargo: int,
    rho_min: float,
) -> WindowBasis | None:
    """Refit the CCA basis on one train window; rank-sort and sign-anchor.

    Returns ``None`` if the window's factor block is degenerate (dead/duplicated
    latents), which is treated as a skipped window rather than a hard failure.
    """
    Fa, Ma = align_xy(F_train, M_train)
    if len(Fa) < 10 * Fa.shape[1]:
        return None
    Fz = (Fa - Fa.mean()) / Fa.std(ddof=0).replace(0.0, np.nan)
    if Fz.isna().any().any():
        return None
    eig = np.linalg.eigvalsh(np.corrcoef(Fz.to_numpy(float), rowvar=False))
    if eig.min() < 1e-8 * eig.mean():
        return None

    scores: dict[float, float] = {}
    for rg in ridge_grid:
        oos = purged_cv_canon(Fa, Ma, n_folds=n_folds, embargo=embargo, ridge=rg, ridge_f=0.0)
        scores[rg] = float(np.nanmean(oos))
    ridge_m = max(scores, key=lambda k: scores[k]) if scores else 0.0

    rho, A, B, _, _, _, struct_m = linear_cca_full(Fa, Ma, ridge=ridge_m, ridge_f=0.0)
    K = A.shape[1]
    for k in range(K):
        j = int(np.argmax(np.abs(struct_m[:, k])))
        anchor = struct_m[j, k] if abs(struct_m[j, k]) >= SIGN_ANCHOR_FLOOR else None
        if anchor is None:
            anchor = A[int(np.argmax(np.abs(A[:, k]))), k]
        if anchor < 0:
            A[:, k] *= -1.0
            B[:, k] *= -1.0
            struct_m[:, k] *= -1.0
    return WindowBasis(
        A=A,
        B=B,
        mu_f=Fa.to_numpy(float).mean(axis=0),
        mu_m=Ma.to_numpy(float).mean(axis=0),
        rho_train=rho,
        struct_macro=struct_m,
        macro_cols=list(Ma.columns),
        n_spanned=_spanned_boundary(rho, rho_min),
        ridge_m=float(ridge_m),
    )


def _encode(model: torch.nn.Module, xz: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        enc = model.encode(torch.tensor(xz, dtype=torch.float32))  # type: ignore[operator]
    return enc.cpu().numpy().astype(np.float64)


def _window_forecast_one_h(
    returns: np.ndarray,
    state: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    h: int,
    keys: list[tuple[int, ...]],
    min_train_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit direct h-step FAAR on the train block, predict the test origins.

    Design per commodity i: ``[1, y_lag_i, state...]``. Training pairs are
    ``(tau, tau + h)`` fully inside the train block (target realized before any
    test day, so parameters are leakage-free). Origins are test-block days whose
    h-step target is realized within the sample.
    """
    T, N = returns.shape
    K = state.shape[1]
    y = h_period_returns(returns, h)  # (T, N)

    train_end = int(train_idx[-1])
    pair_taus = [
        tau
        for tau in range(int(train_idx[0]) + h, train_end + 1)
        if tau + h <= train_end and np.isfinite(y[tau]).all() and np.isfinite(y[tau + h]).all()
    ]
    if len(pair_taus) < min_train_pairs:
        return None
    origins = np.array(
        [
            t
            for t in test_idx
            if t + h < T and np.isfinite(y[t]).all() and np.isfinite(y[t + h]).all()
        ],
        dtype=int,
    )
    if len(origins) == 0:
        return None

    taus = np.asarray(pair_taus, dtype=int)
    P = 2 + K

    def design(idx: np.ndarray) -> np.ndarray:
        d = np.empty((len(idx), N, P))
        d[:, :, 0] = 1.0
        d[:, :, 1] = y[idx]
        d[:, :, 2:] = state[idx][:, None, :]
        return d

    Xtr = design(taus)  # (n_pairs, N, P)
    Ytr = y[taus + h]  # (n_pairs, N)
    Xte = design(origins)  # (n_origins, N, P)

    preds = np.full((len(keys), len(origins), N), np.nan)
    actual = y[origins + h]
    for m_i, key in enumerate(keys):
        if key == ZERO_KEY:
            preds[m_i] = 0.0
            continue
        if key == MEAN_KEY:
            preds[m_i] = Ytr.mean(axis=0)[None, :]
            continue
        cols = [0, 1] + [2 + j for j in key]
        for i in range(N):
            Xi = Xtr[:, i, :][:, cols]
            beta, *_ = np.linalg.lstsq(Xi, Ytr[:, i], rcond=None)
            preds[m_i, :, i] = Xte[:, i, :][:, cols] @ beta
    return origins, origins + h, preds, actual


def rolling_forecasts(
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    macro: pd.DataFrame,
    cfg: RollingForecastConfig,
    factor_cfg: FactorModelConfig,
    model_type: FactorModelType = "vanilla",
    seed: int = 0,
    log=None,
) -> RollingResult:
    """Run the full rolling roll and pool OOS forecasts across windows by rank."""
    T, N = returns.shape
    K = factor_cfg.n_factors
    subsets = all_subsets(K)
    keys: list[tuple[int, ...]] = [ZERO_KEY, MEAN_KEY, AR_KEY]
    keys += [s for s in subsets if s != AR_KEY]

    macro_lag = macro.copy()
    for col in cfg.lagged_macro_series:
        if col in macro_lag.columns:
            macro_lag[col] = macro_lag[col].shift(1)
    macro_lag = macro_lag.dropna(how="any")

    win_cfg = FactorModelConfig(
        n_factors=factor_cfg.n_factors,
        epochs=cfg.ae_epochs,
        batch_size=factor_cfg.batch_size,
        learning_rate=factor_cfg.learning_rate,
        patience=factor_cfg.patience,
        seed=seed,
        activation=factor_cfg.activation,
        beta=factor_cfg.beta,
    )

    starts = list(range(0, T - cfg.train_window - 1, cfg.step))
    per_h_origins: dict[int, list] = {h: [] for h in cfg.horizons}
    per_h_targets: dict[int, list] = {h: [] for h in cfg.horizons}
    per_h_preds: dict[int, list] = {h: [] for h in cfg.horizons}
    per_h_actual: dict[int, list] = {h: [] for h in cfg.horizons}

    win_rows: list[dict] = []
    struct_list: list[np.ndarray] = []
    spanned_list: list[int] = []
    macro_cols: list[str] = []
    n_used = 0
    for w, s0 in enumerate(starts):
        tr = np.arange(s0, s0 + cfg.train_window)
        te = np.arange(s0 + cfg.train_window, min(s0 + cfg.train_window + cfg.test_block, T))
        if len(te) == 0:
            break

        xz = _zscore_train(returns[tr], returns[np.concatenate([tr, te])])
        model, _ = train_factors(xz[: len(tr)], win_cfg, model_type=model_type, seed=seed)
        F_all = _encode(model, xz)  # (len(tr)+len(te), K)
        idx_all = np.concatenate([tr, te])
        F_df = pd.DataFrame(
            F_all, index=dates[idx_all], columns=[f"f{i + 1}" for i in range(K)]
        )
        F_train_df = F_df.iloc[: len(tr)]
        M_train_df = macro_lag.loc[macro_lag.index.isin(dates[tr])]

        basis = _fit_window_basis(
            F_train_df, M_train_df, cfg.ridge_grid, cfg.n_folds, cfg.embargo, cfg.spanned_rho_min
        )
        if basis is None:
            if log:
                log(f"  window {w} ({dates[tr[0]].date()}): degenerate basis, skipped")
            continue
        macro_cols = basis.macro_cols

        # Canonical-variate state on the full (train+test) window calendar.
        state_win = (F_all - basis.mu_f) @ basis.A  # (len(tr)+len(te), K)
        state_full = np.full((T, K), np.nan)
        state_full[idx_all] = state_win

        got_any = False
        for h in cfg.horizons:
            res = _window_forecast_one_h(
                returns, state_full, tr, te, h, keys, cfg.min_train_pairs
            )
            if res is None:
                continue
            o, tg, pr, ac = res
            per_h_origins[h].append(o)
            per_h_targets[h].append(tg)
            per_h_preds[h].append(pr)
            per_h_actual[h].append(ac)
            got_any = True
        if not got_any:
            continue

        # OOS canonical correlation on the test block (frozen-in-window basis).
        M_test_df = macro_lag.loc[macro_lag.index.isin(dates[te])]
        rho_oos = np.full(K, np.nan)
        if len(M_test_df) >= 5:
            common = dates[te].isin(M_test_df.index)
            Vte = state_win[len(tr):][common]
            Ute = (M_test_df.reindex(dates[te][common]).to_numpy(float) - basis.mu_m) @ basis.B
            for k in range(K):
                v = Vte[:, k] - Vte[:, k].mean()
                u = Ute[:, k] - Ute[:, k].mean()
                den = np.sqrt((v @ v) * (u @ u))
                rho_oos[k] = float(v @ u / den) if den > 0 else np.nan

        struct_list.append(basis.struct_macro)
        spanned_list.append(basis.n_spanned)
        top = [
            "; ".join(
                f"{basis.macro_cols[j]}({basis.struct_macro[j, k]:+.2f})"
                for j in np.argsort(-np.abs(basis.struct_macro[:, k]))[:3]
            )
            for k in range(K)
        ]
        win_rows.append(
            {
                "window": w,
                "train_start": str(dates[tr[0]].date()),
                "test_start": str(dates[te[0]].date()),
                "test_end": str(dates[te[-1]].date()),
                "n_train": len(tr),
                "n_test": len(te),
                "ridge_m": basis.ridge_m,
                "n_spanned": basis.n_spanned,
                **{f"rho{k + 1}": float(basis.rho_train[k]) for k in range(K)},
                **{f"rho_oos{k + 1}": float(rho_oos[k]) for k in range(K)},
                **{f"top_macro_V{k + 1}": top[k] for k in range(K)},
            }
        )
        n_used += 1
        if log and (w % 20 == 0):
            log(
                f"  window {w} train {dates[tr[0]].date()}->{dates[tr[-1]].date()} "
                f"test {dates[te[0]].date()}->{dates[te[-1]].date()} "
                f"rho={np.round(basis.rho_train, 2)} spanned=V1..V{basis.n_spanned}"
            )

    if n_used == 0:
        raise RuntimeError("no usable rolling windows produced forecasts")

    forecasts: dict[int, SubsetForecasts] = {}
    for h in cfg.horizons:
        if not per_h_origins[h]:
            continue
        forecasts[h] = SubsetForecasts(
            keys=keys,
            origins=np.concatenate(per_h_origins[h]),
            targets=np.concatenate(per_h_targets[h]),
            preds=np.concatenate(per_h_preds[h], axis=1),
            actual=np.concatenate(per_h_actual[h], axis=0),
            horizon=h,
        )

    return RollingResult(
        forecasts=forecasts,
        windows=pd.DataFrame(win_rows),
        struct_macro=np.stack(struct_list),
        macro_cols=macro_cols,
        n_dims=K,
        n_windows=n_used,
        dates=dates,
        commodities=[],
        spanned_by_window=np.asarray(spanned_list),
        meta={"n_windows_total": len(starts), "n_windows_used": n_used},
    )


def _classify_stability(obs: float, p_value: float, cos_stable: float) -> str:
    """Graded ladder: distinguish genuine stability from above-chance-but-weak."""
    if p_value >= 0.05:
        return "rotating"  # indistinguishable from the shuffled null
    if obs >= cos_stable:
        return "stable"
    if obs >= 0.4:
        return "partial"
    return "weak"


def rank_stability(
    result: RollingResult, cos_stable: float, n_perm: int = 200, seed: int = 0
) -> pd.DataFrame:
    """Cross-window stability of each canonical rank in the fixed macro space.

    For rank ``k`` we take its ``J``-dim macro structure-correlation vector in
    every window (a coordinate that is comparable across windows because the
    macro series never change) and report the median absolute cosine similarity
    between all window pairs. A high median cosine means rank ``k`` keeps
    pointing at the same macro combination even though the AE latent axes rotate
    freely — the condition under which rank-pooled attribution is meaningful.

    Rigor add-ons:

    * **Permutation null** — independently shuffling the macro-series labels of
      each window's fingerprint destroys any cross-window macro identity while
      preserving the loading magnitude distribution; the median |cosine| under
      this shuffle is the empirical chance level, and the one-sided p-value is
      the fraction of ``n_perm`` shuffles that match/beat the observed cosine.
      The theoretical random-unit-vector level ``sqrt(2 / (pi J))`` is reported
      alongside as a sanity anchor.
    * **OOS validation** — the median frozen-in-window canonical correlation of
      the rank on its held-out test block (``rho_oos`` in ``result.windows``),
      i.e. does the direction still track macro out-of-sample, not just in fit.
    * **Graded ladder** — stable / partial / weak / rotating instead of a single
      binary bar (``macro_stable`` is retained for downstream licensing).
    """
    S = result.struct_macro  # (W, J, K)
    W, J, K = S.shape
    rng = np.random.default_rng(seed)
    chance = float(np.sqrt(2.0 / (np.pi * J)))
    win = result.windows
    rows = []
    for k in range(K):
        V = S[:, :, k]  # (W, J)
        norm = np.linalg.norm(V, axis=1)
        ok = norm > 0
        Vn = V[ok] / norm[ok][:, None]
        iu = np.triu_indices(len(Vn), k=1)
        cos = Vn @ Vn.T
        pair_cos = np.abs(cos[iu]) if len(iu[0]) else np.array([np.nan])
        obs = float(np.nanmedian(pair_cos))

        null = np.empty(n_perm)
        for b in range(n_perm):
            Vp = np.stack([rng.permutation(row) for row in Vn])
            cb = Vp @ Vp.T
            null[b] = float(np.nanmedian(np.abs(cb[iu]))) if len(iu[0]) else np.nan
        p_value = float((1 + np.sum(null >= obs)) / (n_perm + 1))

        roos_col = f"rho_oos{k + 1}"
        roos = win[roos_col].to_numpy(float) if roos_col in win else np.array([np.nan])
        rows.append(
            {
                "rank": f"V{k + 1}",
                "median_abs_cos": obs,
                "q10_abs_cos": float(np.nanpercentile(pair_cos, 10)),
                "q90_abs_cos": float(np.nanpercentile(pair_cos, 90)),
                "null_med_abs_cos": float(np.nanmedian(null)),
                "null_p95": float(np.nanpercentile(null, 95)),
                "chance_abs_cos": chance,
                "p_value": p_value,
                "median_rho_oos": float(np.nanmedian(np.abs(roos))),
                "frac_windows_rho_oos_ge_0.3": float(np.nanmean(np.abs(roos) >= 0.3)),
                "stability_class": _classify_stability(obs, p_value, cos_stable),
                "macro_stable": bool(obs >= cos_stable and p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)
