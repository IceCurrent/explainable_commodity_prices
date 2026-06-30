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


def _rbf_gram(X: np.ndarray, gamma: float) -> np.ndarray:
    """RBF (Gaussian) Gram matrix K_ij = exp(-gamma * ||x_i - x_j||^2)."""
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _median_gamma(X: np.ndarray) -> float:
    """Median heuristic for the RBF bandwidth: gamma = 1 / median(pairwise sq dist)."""
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    iu = np.triu_indices_from(d2, k=1)
    med = float(np.median(np.maximum(d2[iu], 0.0)))
    return 1.0 / med if med > 0 else 1.0


def _center_gram(K: np.ndarray) -> np.ndarray:
    """Double-centering H K H with H = I - 11'/n, done cheaply and symmetrized."""
    rm = K.mean(axis=0, keepdims=True)
    Kc = K - rm - rm.T + K.mean()
    return 0.5 * (Kc + Kc.T)


def kernel_canonical_correlations(
    factors: pd.DataFrame,
    macro: pd.DataFrame,
    reg: float = 1.0,
    top_k: int | None = None,
    gamma_f: float | None = None,
    gamma_m: float | None = None,
    eig_tol: float = 1e-10,
) -> np.ndarray:
    """Regularized Kernel CCA between the factor space and the macro space.

    The nonlinear analogue of :func:`canonical_correlations`. Each view is lifted
    into an RBF reproducing-kernel Hilbert space and *linear* CCA is solved there
    (Lai & Fyfe 2000; Bach & Jordan 2002; Hardoon et al. 2004), so the resulting
    canonical correlations measure dependence achievable by arbitrary smooth
    nonlinear transforms of f and of the macro panel -- a finite-sample,
    capacity-controlled estimate of the Hirschfeld-Gebelein-Renyi maximal
    correlation between the two spaces.

    Derivation (so the n-by-n eigenproblem is the only heavy step). With centered
    feature maps Phi_a, Phi_b and Gram matrices K_a = Phi_a Phi_a', K_b likewise,
    write the eigendecomposition K = U diag(lambda) U'. Regularized linear CCA in
    feature space (covariance ridged by kappa) has canonical correlations equal to
    the singular values of

        D_a (U_a' U_b) D_b,   D = diag( sqrt(lambda / (lambda + kappa)) ),

    with kappa = ``reg`` * mean(lambda) set per view. ``reg`` is the crux: without
    it the high-frequency kernel directions drive every canonical correlation to 1
    (the well-known KCCA degeneracy); the ridge shrinks those tail directions to 0
    so only genuinely shared structure survives.

    Returns the canonical correlations in [0, 1], descending. ``top_k`` keeps the
    leading ``top_k`` (default: number of macro columns) so the output is directly
    comparable to the linear :func:`canonical_correlations`, which has exactly
    min(K, J) values; the kernel version otherwise returns up to n mostly-spurious
    tail directions whose minimum is uninformative.
    """
    f, m = _align_xy(factors, macro)
    F = f.to_numpy(dtype=float)
    M = m.to_numpy(dtype=float)
    # Standardize each view so the RBF bandwidth sees comparable per-feature scale.
    F = (F - F.mean(0)) / np.where(F.std(0, ddof=0) == 0, 1.0, F.std(0, ddof=0))
    M = (M - M.mean(0)) / np.where(M.std(0, ddof=0) == 0, 1.0, M.std(0, ddof=0))

    gamma_f = _median_gamma(F) if gamma_f is None else gamma_f
    gamma_m = _median_gamma(M) if gamma_m is None else gamma_m
    Kf = _center_gram(_rbf_gram(F, gamma_f))
    Km = _center_gram(_rbf_gram(M, gamma_m))

    def _whiten(K: np.ndarray):
        lam, U = np.linalg.eigh(K)
        keep = lam > eig_tol * float(lam.max())
        lam, U = lam[keep], U[:, keep]
        kappa = reg * float(lam.mean())
        d = np.sqrt(lam / (lam + kappa))           # in (0, 1); ~1 if lam>>kappa, ~0 if lam<<kappa
        return U, d

    Uf, df = _whiten(Kf)
    Um, dm = _whiten(Km)
    inner = (df[:, None] * (Uf.T @ Um)) * dm[None, :]
    cc = np.linalg.svd(inner, compute_uv=False)
    cc = np.clip(cc, 0.0, 1.0)
    k = m.shape[1] if top_k is None else top_k
    return cc[:k]


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


# =========================================================================== #
# REAL-DATA MACRO MAPPING (CCA_methods.md S6)
#
# Everything below extends the in-sample QR-SVD core above with the rigor the
# J=37 collinear macro panel demands: ridge, out-of-sample (purged CV) canonical
# correlations, a circular-shift permutation null, block-bootstrap CIs, an
# interpretable bloc-PC reduction, a Nystrom KCCA for the null, and a lead/lag
# scan. The existing public functions (canonical_correlations,
# kernel_canonical_correlations, bai_ng_spanning_summary, _align_xy, _rbf_gram,
# _median_gamma, _center_gram) are reused unchanged -- linear_cca_full(ridge=0)
# is the explicit equivalence anchor for canonical_correlations.
# =========================================================================== #
def _as_array(X) -> np.ndarray:
    return X.to_numpy(dtype=float) if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X, float)


def _ridged_inv_sqrt_from_data(Xc: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    """(Sigma + r I)^{-1/2} with Sigma = Xc'Xc/T, r = ridge*mean(diag(Sigma)).

    Computed from the SVD of the *centered data* ``Xc`` rather than from the Gram
    ``Xc'Xc`` -- forming the Gram squares the condition number, and the macro
    panel's FX/credit collinearity makes that the dominant rounding source. The
    eigenvalues of Sigma are ``s**2 / T`` (s = singular values of Xc) and
    ``mean(eigenvalue) == mean(diag(Sigma))``, so the ridge ``r`` matches the
    spec exactly while the whitening stays numerically faithful to the QR-SVD
    core (linear_cca_full(ridge=0) then reproduces canonical_correlations to ~1e-13).
    """
    T = Xc.shape[0]
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (s ** 2) / T                       # eigenvalues of Sigma
    r = ridge * float(lam.mean())
    inv = 1.0 / np.sqrt(lam + r)
    return (Vt.T * inv) @ Vt, r


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0


def _structure_corr(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Correlation of each column of X (T x p) with each canonical variate V (T x r)."""
    Xs = X - X.mean(0)
    Vs = V - V.mean(0)
    Xn = Xs / np.where(Xs.std(0) == 0, 1.0, Xs.std(0))
    Vn = Vs / np.where(Vs.std(0) == 0, 1.0, Vs.std(0))
    return (Xn.T @ Vn) / X.shape[0]


# --------------------------------------------------------------------------- #
# 4.1 ridged linear CCA with vectors, variates, structure correlations
# --------------------------------------------------------------------------- #
def linear_cca_full(F, M, ridge: float = 0.0):
    """Full ridged linear CCA between factor view ``F`` (T x K) and macro ``M`` (T x J).

    Generalizes :func:`canonical_correlations` (the ``ridge=0`` special case) to
    return everything the interpretation needs: the canonical correlations, the
    raw weight vectors ``A`` (factor side) / ``B`` (macro side), the canonical
    variates ``V = F_c A`` / ``U = M_c B``, and -- the interpretable objects --
    the *structure correlations* of each original variable with each canonical
    dimension. Under the collinear macro panel the raw weights ``B`` are unstable
    while the structure correlations are not, so the latter carry the reading.

    The ridge shrinks the per-view covariances by ``ridge * mean(diag(Sigma))``
    (i.e. ``ridge`` times the mean per-feature variance), stabilizing the
    inverse-sqrt whitening on the collinear FX/credit/rates blocks.

    Returns ``(corrs, A, B, U, V, struct_factor, struct_macro)``.
    """
    Fa, Ma = _as_array(F), _as_array(M)
    if Fa.shape[0] != Ma.shape[0]:
        raise ValueError("F and M must be row-aligned before linear_cca_full")
    T, K = Fa.shape
    J = Ma.shape[1]
    Fc = Fa - Fa.mean(0)
    Mc = Ma - Ma.mean(0)
    Sfm = Fc.T @ Mc / T
    # (Sigma + r I)^{-1/2} from the data SVD: r = ridge * mean(diag(Sigma)),
    # matching the spec while avoiding the condition-number squaring of forming
    # the Gram (critical on the collinear macro panel).
    Sff_isq, _ = _ridged_inv_sqrt_from_data(Fc, ridge)
    Smm_isq, _ = _ridged_inv_sqrt_from_data(Mc, ridge)
    Omega = Sff_isq @ Sfm @ Smm_isq                      # K x J
    P, s, Qt = np.linalg.svd(Omega, full_matrices=False)  # P:KxK, s:K, Qt:KxJ
    r = min(K, J)
    corrs = np.clip(s[:r], 0.0, 1.0)
    A = Sff_isq @ P[:, :r]                                # K x r
    B = Smm_isq @ Qt.T[:, :r]                             # J x r
    V = Fc @ A                                            # T x r
    U = Mc @ B                                            # T x r
    struct_factor = _structure_corr(Fa, V)               # K x r
    struct_macro = _structure_corr(Ma, U)                # J x r
    return corrs, A, B, U, V, struct_factor, struct_macro


# --------------------------------------------------------------------------- #
# 4.2 circular-shift permutation null (preserves each view's autocorrelation)
# --------------------------------------------------------------------------- #
def circular_perm_null(score_fn, F, M, n_perm: int, seed: int) -> dict:
    """Permutation null of "no cross-view alignment" via circular shifts of ``M``.

    For each draw, ``M`` is rolled by a random offset tau ~ U(1, T-1) along time;
    the roll leaves ``M``'s own autocovariance intact and only breaks the F-M
    phase. ``score_fn(F, M) -> vector of canonical correlations`` is recomputed
    on each rolled panel. Returns the observed and null distributions of both the
    ``min`` (the spanning statistic) and the ``mean``, with one-sided upper
    p-values ``p = (1 + #{null >= obs}) / (n_perm + 1)``.

    Because J is large and collinear, the null band itself sits high -- a high
    in-sample min proves nothing; significance means the observed value clears
    this null.
    """
    Fa, Ma = _as_array(F), _as_array(M)
    T = Ma.shape[0]
    obs = np.asarray(score_fn(Fa, Ma), float)
    obs_min, obs_mean = float(obs.min()), float(obs.mean())
    rng = np.random.default_rng(seed)
    null_min = np.empty(n_perm)
    null_mean = np.empty(n_perm)
    for b in range(n_perm):
        tau = int(rng.integers(1, T))
        s = np.asarray(score_fn(Fa, np.roll(Ma, tau, axis=0)), float)
        null_min[b] = s.min()
        null_mean[b] = s.mean()
    p_min = (1 + int(np.sum(null_min >= obs_min))) / (n_perm + 1)
    p_mean = (1 + int(np.sum(null_mean >= obs_mean))) / (n_perm + 1)
    return {
        "obs": obs,
        "obs_min": obs_min,
        "obs_mean": obs_mean,
        "null_min": null_min,
        "null_mean": null_mean,
        "null_min_p95": float(np.percentile(null_min, 95)),
        "null_mean_p95": float(np.percentile(null_mean, 95)),
        "null_min_mean": float(null_min.mean()),
        "null_mean_mean": float(null_mean.mean()),
        "p_min": p_min,
        "p_mean": p_mean,
    }


# --------------------------------------------------------------------------- #
# 4.3 purged/embargoed cross-validated (out-of-sample) canonical correlations
# --------------------------------------------------------------------------- #
def purged_cv_canon(F, M, n_folds: int = 5, embargo: int = 10, ridge: float = 0.0) -> np.ndarray:
    """Out-of-sample canonical correlations via contiguous purged/embargoed CV.

    Each test fold is a contiguous time block; the training set is everything
    else minus an ``embargo``-day band on both sides of the block (purging
    autocorrelation leakage). Directions ``A, B`` are fit on the (ridged,
    train-mean-centered) training data and projected onto the test block; the OOS
    canonical correlation per dimension is ``corr(V_test[:,k], U_test[:,k])``,
    averaged over folds. In-sample high + OOS low is the J=37 overfitting
    signature; this is the overfitting detector and the ridge selector.
    """
    Fa, Ma = _as_array(F), _as_array(M)
    T, K = Fa.shape
    r = min(K, Ma.shape[1])
    bounds = np.linspace(0, T, n_folds + 1).astype(int)
    per_fold = []
    for i in range(n_folds):
        ts, te = int(bounds[i]), int(bounds[i + 1])
        lo, hi = max(ts - embargo, 0), min(te + embargo, T)
        train = np.concatenate([np.arange(0, lo), np.arange(hi, T)])
        test = np.arange(ts, te)
        if len(train) <= K + 1 or len(test) < 3:
            continue
        muF, muM = Fa[train].mean(0), Ma[train].mean(0)
        _, A, B, _, _, _, _ = linear_cca_full(Fa[train], Ma[train], ridge=ridge)
        V_te = (Fa[test] - muF) @ A
        U_te = (Ma[test] - muM) @ B
        per_fold.append([_pearson(V_te[:, k], U_te[:, k]) for k in range(r)])
    if not per_fold:
        return np.full(r, np.nan)
    return np.nanmean(np.asarray(per_fold), axis=0)


def select_ridge_oos(F, M, ridges=(0.0, 1e-3, 1e-2, 1e-1, 1.0),
                     n_folds: int = 5, embargo: int = 10) -> tuple[float, dict]:
    """Pick the ridge maximizing OOS mean canonical correlation (4.3)."""
    scores = {}
    for rg in ridges:
        oos = purged_cv_canon(F, M, n_folds=n_folds, embargo=embargo, ridge=rg)
        scores[rg] = float(np.nanmean(oos))
    best = max(scores, key=scores.get)
    return best, scores


# --------------------------------------------------------------------------- #
# 4.4 stationary block-bootstrap CIs on rho_min / rho_mean
# --------------------------------------------------------------------------- #
def _stationary_bootstrap_idx(T: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index (geometric block lengths, wrap)."""
    p = 1.0 / max(mean_block, 1)
    idx = np.empty(T, dtype=int)
    t = 0
    cur = int(rng.integers(0, T))
    while t < T:
        idx[t] = cur
        t += 1
        if rng.random() < p:
            cur = int(rng.integers(0, T))
        else:
            cur = (cur + 1) % T
    return idx


def block_bootstrap_ci(F, M, ridge: float, mean_block: int = 21,
                       n_boot: int = 1000, seed: int = 0) -> dict:
    """Stationary-bootstrap 95% CIs on ``rho_min`` and ``rho_mean``.

    Resamples ``(F, M)`` rows *jointly* in geometric-length blocks (preserving
    alignment and short-range autocorrelation), recomputes the in-sample
    canonical correlations, and returns the 2.5/97.5 percentiles. Reports the
    estimation uncertainty of the observed dependence.
    """
    Fa, Ma = _as_array(F), _as_array(M)
    T = Fa.shape[0]
    rng = np.random.default_rng(seed)
    mins = np.empty(n_boot)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_idx(T, mean_block, rng)
        rho = linear_cca_full(Fa[idx], Ma[idx], ridge=ridge)[0]
        mins[b] = rho.min()
        means[b] = rho.mean()
    return {
        "min_lo": float(np.percentile(mins, 2.5)),
        "min_hi": float(np.percentile(mins, 97.5)),
        "mean_lo": float(np.percentile(means, 2.5)),
        "mean_hi": float(np.percentile(means, 97.5)),
    }


# --------------------------------------------------------------------------- #
# 4.5 bloc reduction -> one sign-fixed first PC per macro bloc
# --------------------------------------------------------------------------- #
def bloc_reduce(M_df: pd.DataFrame, bloc_map: dict) -> pd.DataFrame:
    """Reduce the macro panel to one sign-fixed first PC per bloc.

    For each bloc, the member columns are standardized and the first principal
    component is taken (sign-fixed to load positively on the bloc's mean series).
    Returns a T x n_bloc panel -- the interpretable, low-overfit CCA input
    (avoids the J ~ T rank saturation of the full 37-var panel in small regimes).
    """
    cols_out = {}
    for bloc, cols in bloc_map.items():
        present = [c for c in cols if c in M_df.columns]
        if not present:
            continue
        X = M_df[present].to_numpy(float)
        sd = X.std(0)
        Xs = (X - X.mean(0)) / np.where(sd == 0, 1.0, sd)
        U, S, _ = np.linalg.svd(Xs, full_matrices=False)
        pc = U[:, 0] * S[0]
        if _pearson(pc, Xs.mean(1)) < 0:
            pc = -pc
        cols_out[bloc] = pc
    return pd.DataFrame(cols_out, index=M_df.index)


# --------------------------------------------------------------------------- #
# 4.6 Nystrom-approximated KCCA (for the permutation null / bootstrap)
# --------------------------------------------------------------------------- #
def _rbf_cross(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """RBF kernel between rows of A and rows of B: exp(-gamma ||a-b||^2)."""
    aa = np.einsum("ij,ij->i", A, A)
    bb = np.einsum("ij,ij->i", B, B)
    d2 = aa[:, None] + bb[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _nystrom_features(X: np.ndarray, gamma: float, landmarks: np.ndarray,
                      eig_tol: float = 1e-10) -> np.ndarray:
    """Centered Nystrom feature map Z (T x L) with Z Z' ~ centered RBF Gram."""
    Xl = X[landmarks]
    W = _rbf_cross(Xl, Xl, gamma)
    E = _rbf_cross(X, Xl, gamma)
    lam, Q = np.linalg.eigh(0.5 * (W + W.T))
    keep = lam > eig_tol * float(lam.max())
    lam, Q = lam[keep], Q[:, keep]
    W_isq = (Q * (1.0 / np.sqrt(lam))) @ Q.T
    Z = E @ W_isq
    return Z - Z.mean(0)


def kcca_nystrom(F, M, reg: float = 1.0, gamma_f: float | None = None,
                 gamma_m: float | None = None, n_landmarks: int = 400,
                 top_k: int | None = None, seed: int = 0) -> np.ndarray:
    """Nystrom-approximated regularized kernel CCA.

    Lightweight stand-in for :func:`kernel_canonical_correlations` so the KCCA
    permutation null / bootstrap is affordable (the exact T x T Gram eigendecomp
    is too heavy to permute hundreds of times -- CCA_methods.md S6). Each view is
    standardized, mapped through a centered Nystrom RBF feature approximation,
    then fed to the ridged linear CCA core (the feature-space covariance ridge
    ``reg * mean(eigenvalue)`` matches the exact KCCA's ``kappa = reg*mean(lam)``).
    Use the exact routine for the point estimate; this for the null.
    """
    Fa, Ma = _as_array(F), _as_array(M)
    Fs = (Fa - Fa.mean(0)) / np.where(Fa.std(0) == 0, 1.0, Fa.std(0))
    Ms = (Ma - Ma.mean(0)) / np.where(Ma.std(0) == 0, 1.0, Ma.std(0))
    gf = _median_gamma(Fs) if gamma_f is None else gamma_f
    gm = _median_gamma(Ms) if gamma_m is None else gamma_m
    T = Fs.shape[0]
    rng = np.random.default_rng(seed)
    L = min(n_landmarks, T)
    landmarks = rng.choice(T, size=L, replace=False)
    Zf = _nystrom_features(Fs, gf, landmarks)
    Zm = _nystrom_features(Ms, gm, landmarks)
    cc = linear_cca_full(Zf, Zm, ridge=reg)[0]
    cc = np.clip(cc, 0.0, 1.0)
    k = Ma.shape[1] if top_k is None else top_k
    return cc[:k]


# --------------------------------------------------------------------------- #
# 4.7 lead/lag scan
# --------------------------------------------------------------------------- #
def leadlag_scan(F, M, lags, ridge: float = 0.0) -> dict:
    """Mean canonical correlation as a function of lag.

    For each ``l``, align ``F_t`` with ``M_{t-l}`` (positive ``l`` lags the
    macro panel: macro leads commodities) and report ``mean`` canonical
    correlation. Peak at ``l=0`` => contemporaneous. Returns ``{lag: rho_mean}``.
    """
    Fa, Ma = _as_array(F), _as_array(M)
    T = Fa.shape[0]
    out = {}
    for l in lags:
        if l == 0:
            fi, mi = Fa, Ma
        elif l > 0:
            fi, mi = Fa[l:], Ma[:T - l]
        else:
            fi, mi = Fa[:T + l], Ma[-l:]
        if fi.shape[0] <= Fa.shape[1] + 1:
            out[int(l)] = np.nan
            continue
        out[int(l)] = float(linear_cca_full(fi, mi, ridge=ridge)[0].mean())
    return out
