"""E1/E2: map aligned latent factors to the observable macro panel.

E1 (linear spanning + incremental content)
    * ``spanning_regression`` -- per-factor OLS on M_t with Newey-West (HAC) SE,
      standardized coefficients, R^2, dominant regressors.
    * ``canonical_correlations`` / ``bai_ng_spanning_summary`` -- the
      canonical-correlation core of the Bai & Ng (2006) spanning test: are the
      latent factors spanned by the observed macro panel?
    * ``decompose_factors`` -- split f_k = f_hat^M_k + u_k (macro-spanned part +
      orthogonal residual).
    * ``incremental_content`` -- re-run the factor-augmented forecaster with f_k,
      with only f_hat^M_k, and with only u_k; compare OOS RMSE and forecast-Shapley.

E2 (nonlinear macro mapping)
    * ``nonlinear_macro_mapping`` -- gradient-boosted f_k = h_k(M_t) with
      purged/embargoed time-series CV, SHAP attribution of M, and the
      nonlinearity premium R^2_nonlin - R^2_lin.

Everything is a pure function of (aligned factor panel, macro panel) plus, for
the incremental test, the factor bundle. Nothing here runs until the macro panel
is supplied (see ``src/macro_data.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import ReturnPanel, load_return_panel
from src.factor_alignment import Alignment
from src.factor_extraction import FactorBundle


# --------------------------------------------------------------------------- #
# E1: linear spanning regression
# --------------------------------------------------------------------------- #
@dataclass
class SpanningRegressionResult:
    factor: str
    r2: float
    coefficients: pd.Series      # standardized betas (excl. const)
    tstats: pd.Series
    pvalues: pd.Series
    dominant: list[str]
    n_obs: int
    hac_lags: int


def _align_xy(factors: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = factors.index.intersection(macro.index)
    if len(idx) == 0:
        raise ValueError("factor and macro panels share no dates after alignment")
    return factors.loc[idx], macro.loc[idx]


def spanning_regression(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    hac_lags: int | None = None,
    dominant_threshold: float = 0.1,
) -> dict[str, SpanningRegressionResult]:
    """Per-factor HAC-OLS f_k = a + theta' M + u with standardized regressors."""
    import statsmodels.api as sm

    f, m = _align_xy(factors, macro)
    n = len(f)
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))  # Newey-West rule of thumb

    m_std = (m - m.mean()) / m.std(ddof=0)
    X = sm.add_constant(m_std)
    results: dict[str, SpanningRegressionResult] = {}
    for col in f.columns:
        y = (f[col] - f[col].mean()) / f[col].std(ddof=0)
        model = sm.OLS(y, X, missing="drop").fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_lags}
        )
        betas = model.params.drop("const")
        dominant = list(betas[betas.abs() >= dominant_threshold].sort_values(key=np.abs, ascending=False).index)
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


# --------------------------------------------------------------------------- #
# E1: Bai-Ng (2006) canonical-correlation spanning core
# --------------------------------------------------------------------------- #
def canonical_correlations(factors: pd.DataFrame, macro: pd.DataFrame) -> np.ndarray:
    """Canonical correlations between the factor space and the macro space.

    Computed via the SVD of the whitened cross-covariance. Returns the vector of
    canonical correlations in [0, 1], length min(K, J). All correlations close to
    1 => the latent factor space is (linearly) spanned by the macro panel.
    """
    f, m = _align_xy(factors, macro)
    F = (f - f.mean()).to_numpy()
    M = (m - m.mean()).to_numpy()
    # Whitening via thin QR.
    Qf, _ = np.linalg.qr(F)
    Qm, _ = np.linalg.qr(M)
    s = np.linalg.svd(Qf.T @ Qm, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def bai_ng_spanning_summary(factors: pd.DataFrame, macro: pd.DataFrame) -> dict:
    """Spanning diagnostic in the spirit of Bai & Ng (2006).

    Reports the canonical correlations and the implied number of latent factors
    NOT spanned by the macro panel (canonical correlation below ``threshold``).
    A min canonical correlation near 1 means the latent factors are spanned.
    """
    cc = canonical_correlations(factors, macro)
    return {
        "canonical_correlations": cc.tolist(),
        "min_canonical_corr": float(cc.min()),
        "mean_canonical_corr": float(cc.mean()),
        "n_factors": int(factors.shape[1]),
        "n_macro": int(macro.shape[1]),
    }


# --------------------------------------------------------------------------- #
# E1: incremental-content decomposition
# --------------------------------------------------------------------------- #
@dataclass
class FactorDecomposition:
    spanned: pd.DataFrame      # f_hat^M, date-indexed
    residual: pd.DataFrame     # u = f - f_hat^M
    intercepts: pd.Series      # per factor
    coefficients: pd.DataFrame  # (macro x factor) theta on STANDARDIZED macro
    macro_mean: pd.Series      # standardizer used to fit theta
    macro_std: pd.Series


def decompose_factors(
    factors: pd.DataFrame, macro: pd.DataFrame, ridge_alpha: float = 10.0
) -> FactorDecomposition:
    """Split each factor into its macro-spanned part and orthogonal residual.

    The projection standardizes the macro regressors and applies ridge shrinkage
    (intercept unpenalized). This is essential here: the macro panel is highly
    collinear (FX block, IG/HY credit) and on wildly different scales, so a plain
    OLS projection produces coefficient blow-up that is stable on the fit sample
    but explodes when the map is extrapolated to in-window macro for the
    incremental forecast. Ridge keeps the spanned reconstruction well-behaved.
    """
    f, m = _align_xy(factors, macro)
    mu = m.mean()
    sd = m.std(ddof=0).replace(0.0, 1.0)
    Z = ((m - mu) / sd).to_numpy()
    Zc = np.column_stack([np.ones(len(Z)), Z])            # (n, 1+J)
    J = Z.shape[1]
    A = Zc.T @ Zc + ridge_alpha * np.eye(J + 1)
    A[0, 0] -= ridge_alpha                                 # do not penalize intercept
    coef = np.linalg.solve(A, Zc.T @ f.to_numpy())         # (1+J, K)
    fitted = Zc @ coef
    spanned = pd.DataFrame(fitted, index=f.index, columns=f.columns)
    residual = f - spanned
    return FactorDecomposition(
        spanned=spanned,
        residual=residual,
        intercepts=pd.Series(coef[0], index=f.columns),
        coefficients=pd.DataFrame(coef[1:], index=m.columns, columns=f.columns),
        macro_mean=mu,
        macro_std=sd,
    )


def _macro_on_panel(panel: ReturnPanel, macro: pd.DataFrame) -> np.ndarray:
    """Reindex the stationary macro panel onto the full return-panel dates.

    Returns (T_panel, J) forward-filled array so per-window slices are available.
    """
    aligned = macro.reindex(pd.DatetimeIndex(panel.dates)).ffill().bfill()
    return aligned.to_numpy()


@dataclass
class IncrementalContentResult:
    rmse: pd.DataFrame      # variant x {ar1, model}
    shapley: pd.DataFrame   # variant x factor
    spanned_share_of_value: float


def incremental_content(
    bundle: FactorBundle,
    alignment: Alignment,
    factors_aligned: pd.DataFrame,
    macro: pd.DataFrame,
    panel: ReturnPanel | None = None,
) -> IncrementalContentResult:
    """E1.4: compare forecast value of f vs macro-spanned f_hat vs residual u.

    Re-runs the factor-augmented forecaster (via the Shapley engine, which also
    yields per-variant phi_k) three ways and reports OOS RMSE and forecast-Shapley.
    """
    from src.forecast_shapley import compute_forecast_shapley

    panel = panel or load_return_panel()
    decomposition = decompose_factors(factors_aligned, macro)
    cols = list(decomposition.coefficients.index)
    macro_on_panel = _macro_on_panel(panel, macro[cols])
    # Standardize in-window macro with the SAME scaler used to fit theta.
    z_on_panel = (macro_on_panel - decomposition.macro_mean[cols].to_numpy()) / \
        decomposition.macro_std[cols].to_numpy()
    theta = decomposition.coefficients.to_numpy()
    intercept = decomposition.intercepts.to_numpy()

    def aligned_provider(w: int) -> np.ndarray:
        return alignment.sign[w] * bundle.factors[w][:, alignment.perm[w]]

    def spanned_provider(w: int) -> np.ndarray:
        start, end = int(bundle.window_start_idx[w]), int(bundle.window_end_idx[w])
        return intercept + z_on_panel[start:end] @ theta

    def residual_provider(w: int) -> np.ndarray:
        return aligned_provider(w) - spanned_provider(w)

    variants = {
        "original": aligned_provider,
        "spanned": spanned_provider,
        "residual": residual_provider,
    }
    rmse_rows, shap_rows = {}, {}
    for name, prov in variants.items():
        res = compute_forecast_shapley(
            bundle, alignment, panel=panel, window_factor_provider=prov, verbose=False
        )
        rmse_rows[name] = {
            "rmse_ar1": float(np.sqrt(res.mse_ar1_aggregate)),
            "rmse_model": float(np.sqrt(res.mse_full_aggregate)),
        }
        shap_rows[name] = dict(zip(res.factors, res.phi_aggregate))

    rmse = pd.DataFrame(rmse_rows).T
    shapley = pd.DataFrame(shap_rows).T
    orig_value = rmse.loc["original", "rmse_ar1"] - rmse.loc["original", "rmse_model"]
    span_value = rmse.loc["spanned", "rmse_ar1"] - rmse.loc["spanned", "rmse_model"]
    # The share is only meaningful when the factors actually add forecast value.
    # If the original RMSE improvement is ~0 or negative (factors do not help),
    # the ratio is undefined -- report NaN rather than an exploding percentage.
    share = float(span_value / orig_value) if orig_value > 1e-9 else float("nan")
    return IncrementalContentResult(rmse=rmse, shapley=shapley, spanned_share_of_value=share)


# --------------------------------------------------------------------------- #
# E2: nonlinear macro mapping + macro-SHAP
# --------------------------------------------------------------------------- #
def purged_time_series_folds(n: int, n_splits: int = 5, embargo: int = 21) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining CV folds with an embargo gap to avoid leakage."""
    folds = []
    size = max(n // (n_splits + 1), 1)
    for i in range(n_splits):
        t0 = (i + 1) * size
        t1 = min((i + 2) * size, n)
        if t0 >= n:
            break
        train_end = max(t0 - embargo, 0)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(t0, t1)
        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))
    return folds


@dataclass
class NonlinearMappingResult:
    factor: str
    r2_nonlinear: float
    r2_linear: float
    nonlinearity_premium: float
    shap_importance: pd.Series  # mean |SHAP| per macro


def nonlinear_macro_mapping(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    n_splits: int = 5,
    embargo: int = 21,
    random_state: int = 42,
) -> dict[str, NonlinearMappingResult]:
    """E2: GBT f_k = h_k(M) with purged CV, SHAP attribution, nonlinearity premium."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    f, m = _align_xy(factors, macro)
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

        # SHAP on a full-sample fit for attribution.
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
    """Mean |SHAP| per feature; falls back to permutation importance if shap absent."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X, check_additivity=False)
        imp = np.abs(values).mean(axis=0)
    except Exception:
        # Cheap permutation-style fallback when shap is unavailable: mean change
        # in prediction when each feature is shuffled.
        rng = np.random.default_rng(0)
        imp = np.zeros(X.shape[1])
        pred0 = model.predict(X)
        for j in range(X.shape[1]):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            imp[j] = np.mean(np.abs(model.predict(Xp) - pred0))
    return pd.Series(imp, index=columns).sort_values(ascending=False)
