"""Spanning regressions and nonlinear macro mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eqcp.cca.inference import purged_time_series_folds
from eqcp.cca.linear import align_xy


@dataclass
class SpanningRegressionResult:
    factor: str
    r2: float
    coefficients: pd.Series
    tstats: pd.Series
    pvalues: pd.Series
    dominant: list[str]
    n_obs: int
    hac_lags: int


def spanning_regression(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    hac_lags: int | None = None,
    dominant_threshold: float = 0.1,
) -> dict[str, SpanningRegressionResult]:
    import statsmodels.api as sm

    f, m = align_xy(factors, macro)
    n = len(f)
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    m_std = (m - m.mean()) / m.std(ddof=0)
    X = sm.add_constant(m_std)
    results: dict[str, SpanningRegressionResult] = {}
    for col in f.columns:
        y = (f[col] - f[col].mean()) / f[col].std(ddof=0)
        model = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        betas = model.params.drop("const")
        dominant = list(
            betas[betas.abs() >= dominant_threshold].sort_values(key=np.abs, ascending=False).index
        )
        results[col] = SpanningRegressionResult(
            factor=col,
            r2=float(model.rsquared),
            coefficients=betas,
            tstats=model.tvalues.drop("const"),
            pvalues=model.pvalues.drop("const"),
            dominant=dominant,
            n_obs=int(n),
            hac_lags=hac_lags,
        )
    return results


@dataclass
class NonlinearMappingResult:
    factor: str
    r2_nonlinear: float
    r2_linear: float
    nonlinearity_premium: float
    shap_importance: pd.Series


def nonlinear_macro_mapping(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    n_splits: int = 5,
    embargo: int = 21,
    random_state: int = 42,
) -> dict[str, NonlinearMappingResult]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    f, m = align_xy(factors, macro)
    X = m.to_numpy()
    folds = purged_time_series_folds(len(f), n_splits=n_splits, embargo=embargo)
    out: dict[str, NonlinearMappingResult] = {}

    for col in f.columns:
        y = f[col].to_numpy()
        nl_preds, lin_preds, truths = [], [], []
        for train_idx, test_idx in folds:
            gbt = HistGradientBoostingRegressor(random_state=random_state)
            gbt.fit(X[train_idx], y[train_idx])
            lin = LinearRegression().fit(X[train_idx], y[train_idx])
            nl_preds.append(gbt.predict(X[test_idx]))
            lin_preds.append(lin.predict(X[test_idx]))
            truths.append(y[test_idx])
        yt = np.concatenate(truths)
        r2_nl = float(r2_score(yt, np.concatenate(nl_preds)))
        r2_lin = float(r2_score(yt, np.concatenate(lin_preds)))

        gbt_full = HistGradientBoostingRegressor(random_state=random_state).fit(X, y)
        shap_imp = _shap_importance(gbt_full, X, m.columns)
        out[col] = NonlinearMappingResult(
            factor=col,
            r2_nonlinear=r2_nl,
            r2_linear=r2_lin,
            nonlinearity_premium=r2_nl - r2_lin,
            shap_importance=shap_imp,
        )
    return out


def _shap_importance(model, X: np.ndarray, columns) -> pd.Series:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X, check_additivity=False)
        imp = np.abs(values).mean(axis=0)
    except Exception:
        rng = np.random.default_rng(0)
        imp = np.zeros(X.shape[1])
        pred0 = model.predict(X)
        for j in range(X.shape[1]):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            imp[j] = np.mean(np.abs(model.predict(Xp) - pred0))
    return pd.Series(imp, index=columns).sort_values(ascending=False)
