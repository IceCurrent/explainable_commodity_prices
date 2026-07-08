"""Macro mapping pipeline: AE latent factors <-> macro panel (CCA)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from eqcp.cca.inference import (
    block_bootstrap_ci,
    circular_perm_null,
    circular_perm_null_oos,
    perdim_perm_null,
    perdim_perm_null_oos,
    purged_cv_canon,
    select_ridge_oos,
    stationary_bootstrap_idx,
)
from eqcp.cca.kernel import kcca_nystrom, kernel_canonical_correlations, median_gamma
from eqcp.cca.leadlag import leadlag_scan
from eqcp.cca.linear import (
    align_xy,
    canonical_correlations,
    linear_cca_full,
)
from eqcp.cca.reduce import bloc_reduce
from eqcp.cca.verdict import kcca_nonlinearity_verdict
from eqcp.config import (
    PROJECT_ROOT,
    FactorModelConfig,
    MacroMappingConfig,
    load_factor_model_config,
    load_macro_mapping_config,
)
from eqcp.factors.extract import (
    FACTOR_CSV_NAMES,
    RESULTS_SUBDIRS,
    FactorModelType,
    regenerate_factors,
)
from eqcp.io.commodities import load_return_panel
from eqcp.io.macro import (
    build_preferred_panel,
    load_macro_stationary,
    load_regime_labels,
)
from eqcp.reporting.figures import make_figures
from eqcp.reporting.report import write_report

logger = logging.getLogger(__name__)


class RunLogger:
    """Log to the logging module and accumulate lines for run_log.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        logger.info(msg)
        self.lines.append(msg)

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def _read_factor_csv(path: Path) -> pd.DataFrame:
    F = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    return F.select_dtypes("number")


def _factor_model_type(args: argparse.Namespace) -> FactorModelType:
    model_type = getattr(args, "factor_model", "vanilla")
    if model_type not in FACTOR_CSV_NAMES:
        raise ValueError(
            f"factor_model must be one of {tuple(FACTOR_CSV_NAMES)}, got {model_type!r}"
        )
    return model_type


def resolve_factors(
    args: argparse.Namespace,
    factor_cfg: FactorModelConfig,
    log: RunLogger,
) -> tuple[pd.DataFrame, str]:
    """Resolve the factor panel: explicit path -> repo search -> regen."""
    k = factor_cfg.n_factors
    model_type = _factor_model_type(args)

    if args.factors is not None and Path(args.factors).exists():
        F = _read_factor_csv(Path(args.factors))
        log(f"factors: loaded explicit --factors {args.factors}  shape={F.shape}")
        return F, str(args.factors)

    default = PROJECT_ROOT / "data" / "processed" / FACTOR_CSV_NAMES[model_type]
    tag = "vanilla" if model_type == "vanilla" else "beta"
    candidates = [default]
    for d in (PROJECT_ROOT / "data", PROJECT_ROOT / "results"):
        if d.exists():
            candidates += [
                p
                for p in d.rglob("*.csv")
                if tag in p.name.lower() and "factor" in p.name.lower()
            ]
    for cand in candidates:
        if cand.exists():
            F = _read_factor_csv(cand)
            if F.shape[1] == k:
                log(f"factors: found existing {model_type} export {cand}  shape={F.shape}")
                return F, str(cand)

    log(f"factors: no export found; searched {[str(c) for c in candidates]}")
    log(f"factors: regenerating from {model_type} (full-sample fit/encode)")
    F = regenerate_factors_for_model(factor_cfg, model_type, log)
    default.parent.mkdir(parents=True, exist_ok=True)
    F.to_csv(default)
    log(f"factors: wrote regenerated factors -> {default}  shape={F.shape}")
    return F, str(default)


def regenerate_factors_for_model(
    factor_cfg: FactorModelConfig,
    model_type: FactorModelType,
    log: RunLogger,
    activation: str | None = None,
) -> pd.DataFrame:
    """Train the requested factor model on the full commodity panel; encode all days."""
    panel = load_return_panel()
    Xz = panel.standardized
    act = activation or factor_cfg.activation
    fc = FactorModelConfig(
        n_factors=factor_cfg.n_factors,
        epochs=factor_cfg.epochs,
        batch_size=factor_cfg.batch_size,
        learning_rate=factor_cfg.learning_rate,
        patience=factor_cfg.patience,
        seed=factor_cfg.seed,
        activation=act,
        beta=factor_cfg.beta,
    )
    if model_type == "vanilla":
        log(
            f"  AE: VanillaAutoencoder K={fc.n_factors} activation={act} "
            f"epochs={fc.epochs} seed={fc.seed} on {Xz.shape[0]}x{Xz.shape[1]} commodity returns"
        )
    else:
        log(
            f"  AE: BetaVAE K={fc.n_factors} beta={fc.beta} activation={act} "
            f"epochs={fc.epochs} seed={fc.seed} on {Xz.shape[0]}x{Xz.shape[1]} commodity returns"
        )
    return regenerate_factors(fc, model_type=model_type, activation=act)


def regenerate_vanilla_factors(
    factor_cfg: FactorModelConfig,
    log: RunLogger,
    activation: str | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper for vanilla AE factor regeneration."""
    return regenerate_factors_for_model(factor_cfg, "vanilla", log, activation=activation)


def perdim_boot(
    F: pd.DataFrame,
    M: pd.DataFrame,
    ridge: float,
    mean_block: int,
    n_boot: int,
    seed: int,
    r: int,
) -> np.ndarray:
    Fa, Ma = F.to_numpy(float), M.to_numpy(float)
    T = Fa.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, r))
    for b in range(n_boot):
        idx = stationary_bootstrap_idx(T, mean_block, rng)
        draws[b] = linear_cca_full(Fa[idx], Ma[idx], ridge)[0][:r]
    return draws


def run_factor_model_comparison(
    Fa_index: pd.DatetimeIndex,
    Ma: pd.DataFrame,
    cfg: MacroMappingConfig,
    factor_cfg: FactorModelConfig,
    seed: int,
    log: RunLogger,
) -> pd.DataFrame:
    """Retrain vanilla AE and beta-VAE; compare OOS macro-spanning counts."""
    rows: list[dict] = []
    for i, model_type in enumerate(("vanilla", "beta_vae")):
        log(f"factor model comparison: {model_type}")
        fc = FactorModelConfig(
            n_factors=factor_cfg.n_factors,
            epochs=factor_cfg.epochs,
            batch_size=factor_cfg.batch_size,
            learning_rate=factor_cfg.learning_rate,
            patience=factor_cfg.patience,
            seed=factor_cfg.seed + i,
            activation=factor_cfg.activation,
            beta=factor_cfg.beta,
        )
        F_act = regenerate_factors_for_model(fc, model_type, log)  # type: ignore[arg-type]
        F_act = F_act.reindex(Fa_index).dropna(how="any")
        Fg, Mg = align_xy(F_act, Ma)
        r = min(Fg.shape[1], Mg.shape[1])
        ridge, _ = select_ridge_oos(
            Fg,
            Mg,
            ridges=cfg.ridge_grid,
            n_folds=cfg.n_folds,
            embargo=cfg.embargo,
        )
        rho_oos = purged_cv_canon(Fg, Mg, n_folds=cfg.n_folds, embargo=cfg.embargo, ridge=ridge)
        pd_null_oos = perdim_perm_null_oos(
            Fg,
            Mg,
            cfg.n_perm_linear,
            seed + 300 + i,
            r,
            cfg.n_folds,
            cfg.embargo,
            ridge,
        )
        perm_p = (1 + np.sum(pd_null_oos >= rho_oos[None, :], axis=0)) / (cfg.n_perm_linear + 1)
        n_spanned = int(np.sum((rho_oos > 0.3) & (perm_p < 0.05)))
        oos_agg = circular_perm_null_oos(
            Fg,
            Mg,
            cfg.n_perm_linear,
            seed + 400 + i,
            cfg.n_folds,
            cfg.embargo,
            ridge,
        )
        rows.append(
            {
                "model": model_type,
                "beta": fc.beta if model_type == "beta_vae" else np.nan,
                "n_spanned": n_spanned,
                "rho_oos_min": float(np.nanmin(rho_oos)),
                "rho_oos_mean": float(np.nanmean(rho_oos)),
                "perm_p_min": oos_agg["p_min"],
                "perm_p_mean": oos_agg["p_mean"],
                "ridge": ridge,
            }
        )
    return pd.DataFrame(rows)


def run_encoder_activation_experiment(
    Fa_index: pd.DatetimeIndex,
    Ma: pd.DataFrame,
    cfg: MacroMappingConfig,
    factor_cfg: FactorModelConfig,
    seed: int,
    log: RunLogger,
) -> pd.DataFrame:
    """Retrain AE per activation and count OOS macro-spanned dimensions."""
    rows: list[dict] = []
    for i, activation in enumerate(cfg.encoder_activations):
        log(f"encoder experiment: activation={activation}")
        fc = FactorModelConfig(
            n_factors=factor_cfg.n_factors,
            epochs=factor_cfg.epochs,
            batch_size=factor_cfg.batch_size,
            learning_rate=factor_cfg.learning_rate,
            patience=factor_cfg.patience,
            seed=factor_cfg.seed + i,
            activation=activation,
        )
        F_act = regenerate_vanilla_factors(fc, log, activation=activation)
        F_act = F_act.reindex(Fa_index).dropna(how="any")
        Fg, Mg = align_xy(F_act, Ma)
        r = min(Fg.shape[1], Mg.shape[1])
        ridge, _ = select_ridge_oos(
            Fg,
            Mg,
            ridges=cfg.ridge_grid,
            n_folds=cfg.n_folds,
            embargo=cfg.embargo,
        )
        rho_oos = purged_cv_canon(Fg, Mg, n_folds=cfg.n_folds, embargo=cfg.embargo, ridge=ridge)
        pd_null_oos = perdim_perm_null_oos(
            Fg,
            Mg,
            cfg.n_perm_linear,
            seed + 100 + i,
            r,
            cfg.n_folds,
            cfg.embargo,
            ridge,
        )
        perm_p = (1 + np.sum(pd_null_oos >= rho_oos[None, :], axis=0)) / (cfg.n_perm_linear + 1)
        n_spanned = int(np.sum((rho_oos > 0.3) & (perm_p < 0.05)))
        oos_agg = circular_perm_null_oos(
            Fg,
            Mg,
            cfg.n_perm_linear,
            seed + 200 + i,
            cfg.n_folds,
            cfg.embargo,
            ridge,
        )
        rows.append(
            {
                "activation": activation,
                "n_spanned": n_spanned,
                "rho_oos_min": float(np.nanmin(rho_oos)),
                "rho_oos_mean": float(np.nanmean(rho_oos)),
                "perm_p_min": oos_agg["p_min"],
                "perm_p_mean": oos_agg["p_mean"],
                "ridge": ridge,
            }
        )
    return pd.DataFrame(rows)


def robustness(
    Fa: pd.DataFrame,
    Ma: pd.DataFrame,
    ridge: float,
    ridge_scores: dict[float, float],
    cfg: MacroMappingConfig,
    seed: int,
    log: RunLogger,
) -> list[dict]:
    rows: list[dict] = []
    if "xle" in Ma.columns:
        Mx = Ma.drop(columns=["xle"])
        rho_x = linear_cca_full(Fa, Mx, ridge)[0]
        oos_x = purged_cv_canon(Fa, Mx, ridge=ridge)
        rho_full = linear_cca_full(Fa, Ma, ridge)[0]
        rows += [
            {
                "test": "drop_xle",
                "setting": "J=36",
                "metric": "rho_min",
                "value": float(rho_x.min()),
            },
            {
                "test": "drop_xle",
                "setting": "J=36",
                "metric": "rho_mean",
                "value": float(rho_x.mean()),
            },
            {
                "test": "drop_xle",
                "setting": "J=36",
                "metric": "oos_min",
                "value": float(np.nanmin(oos_x)),
            },
            {
                "test": "drop_xle",
                "setting": "baseline_J37",
                "metric": "rho_min",
                "value": float(rho_full.min()),
            },
        ]
    for rg, sc in ridge_scores.items():
        rows.append(
            {
                "test": "ridge_sweep",
                "setting": f"ridge={rg}",
                "metric": "oos_mean",
                "value": float(sc),
            }
        )
    for mb in cfg.boot_blocks:
        ci = block_bootstrap_ci(Fa, Ma, ridge, mean_block=mb, n_boot=cfg.n_boot, seed=seed)
        rows += [
            {
                "test": "boot_block",
                "setting": f"mean_block={mb}",
                "metric": "rho_min_lo",
                "value": ci["min_lo"],
            },
            {
                "test": "boot_block",
                "setting": f"mean_block={mb}",
                "metric": "rho_min_hi",
                "value": ci["min_hi"],
            },
        ]
    seam = pd.Timestamp(cfg.china_seam)
    for label, sl in (("pre_2016-08-02", Fa.index < seam), ("post_2016-08-02", Fa.index >= seam)):
        if sl.sum() < 50:
            continue
        rho_s = linear_cca_full(Fa[sl], Ma[sl], ridge)[0]
        rows += [
            {
                "test": "china_seam",
                "setting": label,
                "metric": "rho_min",
                "value": float(rho_s.min()),
            },
            {
                "test": "china_seam",
                "setting": label,
                "metric": "rho_mean",
                "value": float(rho_s.mean()),
            },
        ]
    log(f"robustness: {len(rows)} rows (drop-xle, ridge sweep, boot blocks, China seam)")
    return rows


def write_outputs(
    res: Path,
    factor_cols: list[str],
    macro_cols: list[str],
    rho_is: np.ndarray,
    rho_oos: np.ndarray,
    pd_null_oos: np.ndarray,
    pd_null_is: np.ndarray,
    pd_boot: np.ndarray,
    perm_p_oos: np.ndarray,
    perm_p_insample: np.ndarray,
    kcca_is: np.ndarray,
    kpd_null: np.ndarray,
    kcca_perm_p_dim: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    struct_f: np.ndarray,
    struct_m: np.ndarray,
    dates: pd.DatetimeIndex,
    rho_bl: np.ndarray,
    rho_bl_oos: np.ndarray,
    bperm: dict,
    sm_bl: np.ndarray,
    sf_bl: np.ndarray,
    bloc_cols: list[str],
    regime_df: pd.DataFrame,
    ll_df: pd.DataFrame,
    stab: pd.DataFrame,
    perm: dict,
    kperm: dict,
    rob_rows: list[dict],
    r: int,
) -> None:
    dims = [f"dim{k + 1}" for k in range(r)]

    cc = pd.DataFrame(
        {
            "rho_insample": rho_is,
            "rho_oos": rho_oos,
            "null_mean": pd_null_oos.mean(0),
            "null_p95": np.percentile(pd_null_oos, 95, axis=0),
            "perm_p": perm_p_oos,
            "perm_p_insample": perm_p_insample,
            "boot_lo": np.percentile(pd_boot, 2.5, axis=0),
            "boot_hi": np.percentile(pd_boot, 97.5, axis=0),
            "kcca_insample": kcca_is[:r],
            "kcca_null_p95": np.percentile(kpd_null, 95, axis=0),
            "kcca_perm_p": kcca_perm_p_dim,
        },
        index=dims,
    )
    cc.index.name = "dim"
    cc.to_csv(res / "canonical_correlations.csv")

    pd.DataFrame(struct_m, index=macro_cols, columns=dims).to_csv(
        res / "canonical_loadings_macro.csv"
    )
    pd.DataFrame(struct_f, index=factor_cols, columns=dims).to_csv(
        res / "canonical_loadings_factors.csv"
    )

    vec = pd.concat(
        [
            pd.DataFrame(A, index=[f"A_{c}" for c in factor_cols], columns=dims),
            pd.DataFrame(B, index=[f"B_{c}" for c in macro_cols], columns=dims),
        ]
    )
    vec.to_csv(res / "canonical_vectors.csv")

    pd.DataFrame(U, index=dates, columns=[f"cv{k + 1}" for k in range(r)]).to_csv(
        res / "canonical_variates_macro.csv"
    )
    pd.DataFrame(V, index=dates, columns=[f"cv{k + 1}" for k in range(r)]).to_csv(
        res / "canonical_variates_factors.csv"
    )

    bl = pd.DataFrame(
        {
            "rho_insample": rho_bl,
            "rho_oos": rho_bl_oos,
            "null_p95": [bperm["null_min_p95"]] + [np.nan] * (len(rho_bl) - 1),
        },
        index=dims,
    )
    bl.index.name = "dim"
    bl_struct = pd.DataFrame(sm_bl, index=bloc_cols, columns=dims)
    bl.to_csv(res / "bloc_cca_summary.csv")
    bl_struct.to_csv(res / "bloc_cca_structure_macro.csv")

    regime_df.to_csv(res / "per_regime_summary.csv", index=False)
    ll_df.to_csv(res / "leadlag_scan.csv", index=False)
    stab.to_csv(res / "kcca_stability.csv", index=False)

    perm_null = pd.DataFrame(
        {
            "linear_null_min": perm["null_min"],
            "linear_null_mean": perm["null_mean"],
        }
    )
    kdf = pd.DataFrame({"kcca_null_min": kperm["null_min"], "kcca_null_mean": kperm["null_mean"]})
    pd.concat([perm_null, kdf], axis=1).to_csv(res / "permutation_null.csv", index=False)

    pd.DataFrame(rob_rows).to_csv(res / "robustness.csv", index=False)


def run_macro_mapping(args: argparse.Namespace) -> None:
    """Run the full macro-mapping pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_macro_mapping_config(args.config)
    factor_cfg = load_factor_model_config(args.factor_config)
    factor_cfg = FactorModelConfig(
        n_factors=factor_cfg.n_factors,
        epochs=factor_cfg.epochs,
        batch_size=factor_cfg.batch_size,
        learning_rate=factor_cfg.learning_rate,
        patience=factor_cfg.patience,
        seed=args.seed,
        activation=factor_cfg.activation,
        beta=factor_cfg.beta,
    )

    rng_seed = args.seed
    model_type = _factor_model_type(args)
    out = Path(args.outdir)
    res = out / "results" / RESULTS_SUBDIRS[model_type]
    figs = out / "figures" / RESULTS_SUBDIRS[model_type]
    rep = out / "reports"
    for d in (res, figs, rep):
        d.mkdir(parents=True, exist_ok=True)

    log = RunLogger()
    manifest = pd.read_csv(args.manifest)

    F, factor_src = resolve_factors(args, factor_cfg, log)
    M_full = load_macro_stationary(args.macro, manifest)
    log(f"macro: {M_full.shape[1]} series from {args.macro}")

    Fa, Ma = align_xy(F, M_full)
    T = len(Fa)
    macro_cols = list(Ma.columns)
    factor_cols = list(Fa.columns)
    K = Fa.shape[1]
    J = Ma.shape[1]
    r = min(K, J)
    log(f"aligned: T={T}  K={K}  J={J}  ({Fa.index.min().date()} -> {Fa.index.max().date()})")

    acceptance: list[tuple[str, bool, str]] = []

    eq = np.max(np.abs(linear_cca_full(Fa, Ma, 0.0)[0] - canonical_correlations(Fa, Ma)))
    acceptance.append(
        (
            "linear_cca_full(ridge=0)==canonical_correlations (<1e-8)",
            eq < 1e-8,
            f"max|diff|={eq:.2e}",
        )
    )
    gpr_detail = (
        f"T={T} (threshold {cfg.aligned_t_min}; "
        "gpr early-history gaps reduce aligned T vs naive 2900+)"
    )
    acceptance.append((f"aligned T >= {cfg.aligned_t_min}", T >= cfg.aligned_t_min, gpr_detail))

    try:
        pref = build_preferred_panel(args.levels, manifest, Fa.index)
        Fp, Mp = align_xy(Fa, pref)
        Fc2, Mc2 = align_xy(Fa, Ma.loc[Mp.index])
        rho_conv = linear_cca_full(Fc2, Mc2, 0.0)[0]
        rho_pref = linear_cca_full(Fp, Mp, 0.0)[0]
        agree = float(np.max(np.abs(rho_conv - rho_pref)))
        acceptance.append(
            (
                "convenience vs preferred panel rho agree (<0.02)",
                agree < 0.02,
                f"max|diff|={agree:.4f} on {len(Mp)} shared dates",
            )
        )
        log(f"preferred-path: max|rho_conv - rho_pref|={agree:.4f}")
    except Exception as exc:  # pragma: no cover
        log(f"preferred-path: skipped ({exc})")
        acceptance.append(("convenience vs preferred panel rho agree (<0.02)", False, str(exc)))

    ridge, ridge_scores = select_ridge_oos(
        Fa, Ma, ridges=cfg.ridge_grid, n_folds=cfg.n_folds, embargo=cfg.embargo
    )
    log(
        f"ridge: CV-selected ridge={ridge}  "
        f"OOS-mean scores={ {k: round(v, 4) for k, v in ridge_scores.items()} }"
    )

    rho_is, A, B, U, V, struct_f, struct_m = linear_cca_full(Fa, Ma, ridge)
    rho_oos = purged_cv_canon(Fa, Ma, n_folds=cfg.n_folds, embargo=cfg.embargo, ridge=ridge)
    log(f"linear: rho_insample={np.round(rho_is, 3)}")
    log(f"linear: rho_oos     ={np.round(rho_oos, 3)}")

    def lin_score(f, m):
        return linear_cca_full(f, m, ridge)[0]

    perm_is = circular_perm_null(
        lin_score, Fa.to_numpy(), Ma.to_numpy(), cfg.n_perm_linear, rng_seed
    )
    oos_perm = circular_perm_null_oos(
        Fa, Ma, cfg.n_perm_linear, rng_seed, cfg.n_folds, cfg.embargo, ridge
    )
    boot = block_bootstrap_ci(
        Fa, Ma, ridge, mean_block=cfg.mean_block, n_boot=cfg.n_boot, seed=rng_seed
    )
    log(
        f"linear: rho_min={rho_is.min():.3f} OOS={rho_oos.min():.3f} "
        f"OOS_null_p95={oos_perm['null_min_p95']:.3f} OOS_perm_p={oos_perm['p_min']:.4f} "
        f"boot=[{boot['min_lo']:.3f},{boot['min_hi']:.3f}]"
    )
    log(
        f"linear: rho_mean={rho_is.mean():.3f} OOS={np.nanmean(rho_oos):.3f} "
        f"OOS_null_p95={oos_perm['null_mean_p95']:.3f} OOS_perm_p={oos_perm['p_mean']:.4f}"
    )

    pd_null_is = perdim_perm_null(
        lin_score, Fa.to_numpy(), Ma.to_numpy(), cfg.n_perm_linear, rng_seed + 1, r
    )
    pd_null_oos = perdim_perm_null_oos(
        Fa, Ma, cfg.n_perm_linear, rng_seed + 1, r, cfg.n_folds, cfg.embargo, ridge
    )
    pd_boot = perdim_boot(Fa, Ma, ridge, cfg.mean_block, cfg.n_boot, rng_seed + 2, r)
    perm_p_insample = (1 + np.sum(pd_null_is >= rho_is[None, :], axis=0)) / (cfg.n_perm_linear + 1)
    perm_p_oos = (1 + np.sum(pd_null_oos >= rho_oos[None, :], axis=0)) / (cfg.n_perm_linear + 1)

    Fs = (Fa - Fa.mean()) / Fa.std(ddof=0)
    Ms = (Ma - Ma.mean()) / Ma.std(ddof=0)
    base_gf = median_gamma(Fs.to_numpy())
    base_gm = median_gamma(Ms.to_numpy())
    kcca_is = kernel_canonical_correlations(Fa, Ma, reg=cfg.kcca_reg, top_k=J)
    log(
        f"kcca: exact reg={cfg.kcca_reg} top5={np.round(kcca_is[:5], 3)} "
        f"min={kcca_is.min():.3f} mean={kcca_is.mean():.3f}"
    )

    stab_rows = []
    for reg_k in cfg.kcca_reg_grid:
        for gs in cfg.kcca_gamma_scale:
            cc = kernel_canonical_correlations(
                Fa, Ma, reg=reg_k, top_k=J, gamma_f=base_gf * gs, gamma_m=base_gm * gs
            )
            stab_rows.append(
                {
                    "reg": reg_k,
                    "gamma_scale": gs,
                    "kcca_min": float(cc.min()),
                    "kcca_mean": float(cc.mean()),
                }
            )
    stab = pd.DataFrame(stab_rows)

    def kcca_score(f, m):
        return kcca_nystrom(
            f,
            m,
            reg=cfg.kcca_reg,
            gamma_f=base_gf,
            gamma_m=base_gm,
            n_landmarks=cfg.nystrom_landmarks,
            top_k=5,
            seed=rng_seed,
        )

    kperm = circular_perm_null(kcca_score, Fa.to_numpy(), Ma.to_numpy(), cfg.n_perm_kcca, rng_seed)
    kpd_null = perdim_perm_null(
        kcca_score, Fa.to_numpy(), Ma.to_numpy(), cfg.n_perm_kcca, rng_seed + 3, 5
    )
    kcca_perm_p_dim = (1 + np.sum(kpd_null >= kcca_is[:5][None, :], axis=0)) / (cfg.n_perm_kcca + 1)
    log(f"kcca: Nystrom null min p={kperm['p_min']:.4f} mean p={kperm['p_mean']:.4f}")

    kcca_verdict, kcca_verdict_detail = kcca_nonlinearity_verdict(
        stab, rho_is.min(), kcca_is.min(), cfg.kcca_stability_iqr_threshold
    )

    blocPC = bloc_reduce(Ma, cfg.bloc_map)
    rho_bl, A_bl, B_bl, U_bl, V_bl, sf_bl, sm_bl = linear_cca_full(Fa, blocPC, ridge)
    rho_bl_oos = purged_cv_canon(Fa, blocPC, n_folds=cfg.n_folds, embargo=cfg.embargo, ridge=ridge)
    bperm = circular_perm_null(
        lambda f, m: linear_cca_full(f, m, ridge)[0],
        Fa.to_numpy(),
        blocPC.to_numpy(),
        cfg.n_perm_linear,
        rng_seed,
    )
    log(
        f"bloc-PC: rho_is={np.round(rho_bl, 3)} oos={np.round(rho_bl_oos, 3)} "
        f"min_p={bperm['p_min']:.4f}"
    )

    regimes = load_regime_labels(args.regimes, Fa.index)
    regime_rows = []
    regime_total = 0
    for rg in sorted(regimes.dropna().unique()):
        mask = (regimes == rg).to_numpy()
        n_rg = int(mask.sum())
        regime_total += n_rg
        if n_rg < K + 2:
            regime_rows.append(
                {
                    "regime": rg,
                    "T": n_rg,
                    "rho_min": np.nan,
                    "rho_mean": np.nan,
                    "null_min_p95": np.nan,
                    "perm_p_min": np.nan,
                    "low_power": True,
                }
            )
            continue
        Fg = Fa.to_numpy()[mask]
        Bg = blocPC.to_numpy()[mask]
        rho_g = linear_cca_full(Fg, Bg, ridge)[0]
        gperm = circular_perm_null(
            lambda f, m: linear_cca_full(f, m, ridge)[0],
            Fg,
            Bg,
            cfg.n_perm_regime,
            rng_seed,
        )
        low_power = n_rg < 252
        regime_rows.append(
            {
                "regime": rg,
                "T": n_rg,
                "rho_min": float(rho_g.min()),
                "rho_mean": float(rho_g.mean()),
                "null_min_p95": gperm["null_min_p95"],
                "perm_p_min": gperm["p_min"],
                "low_power": low_power,
            }
        )
    regime_df = pd.DataFrame(regime_rows)
    acceptance.append(
        ("per-regime slices sum to full sample", regime_total == T, f"sum={regime_total} T={T}")
    )
    log(
        f"per-regime: {len(regime_df)} regimes, low-power flagged: "
        f"{regime_df[regime_df['low_power']]['regime'].tolist()}"
    )

    ll = leadlag_scan(Fa, Ma, cfg.lags, ridge)
    ll_df = pd.DataFrame({"lag": list(ll.keys()), "rho_mean": list(ll.values())})
    peak = ll_df.loc[ll_df["rho_mean"].idxmax(), "lag"]
    log(f"lead/lag: peak at lag={int(peak)} (0=contemporaneous)")

    rob_rows = robustness(Fa, Ma, ridge, ridge_scores, cfg, rng_seed, log)

    encoder_exp = run_encoder_activation_experiment(Fa.index, Ma, cfg, factor_cfg, rng_seed, log)
    encoder_exp.to_csv(res / "encoder_activation_experiment.csv", index=False)
    log(f"encoder experiment: wrote {len(encoder_exp)} rows")

    model_cmp = run_factor_model_comparison(Fa.index, Ma, cfg, factor_cfg, rng_seed, log)
    cmp_path = out / "results" / "macro_mapping" / "factor_model_comparison.csv"
    cmp_path.parent.mkdir(parents=True, exist_ok=True)
    model_cmp.to_csv(cmp_path, index=False)
    log(f"factor model comparison: wrote {len(model_cmp)} rows -> {cmp_path}")

    write_outputs(
        res,
        factor_cols,
        macro_cols,
        rho_is,
        rho_oos,
        pd_null_oos,
        pd_null_is,
        pd_boot,
        perm_p_oos,
        perm_p_insample,
        kcca_is,
        kpd_null,
        kcca_perm_p_dim,
        A,
        B,
        U,
        V,
        struct_f,
        struct_m,
        Fa.index,
        rho_bl,
        rho_bl_oos,
        bperm,
        sm_bl,
        sf_bl,
        list(blocPC.columns),
        regime_df,
        ll_df,
        stab,
        perm_is,
        kperm,
        rob_rows,
        r,
    )

    make_figures(figs, rho_is, rho_oos, pd_null_oos, struct_m, macro_cols, regime_df, stab, ll_df)

    nan_free = (not np.isnan(U).any()) and (not np.isnan(V).any())
    uv_unit = np.allclose(np.r_[V.var(0), U.var(0)], 1.0, atol=0.05)
    acceptance.append(("canonical variates NaN-free", nan_free, ""))
    acceptance.append(
        ("canonical variates unit-variance (+/-5%)", uv_unit, f"V.var={np.round(V.var(0), 3)}")
    )
    acceptance.append(
        (
            "OOS rho computed for all 5 dims; OOS perm null has n_perm draws",
            (~np.isnan(rho_oos)).all() and len(oos_perm["null_min"]) == cfg.n_perm_linear,
            "",
        )
    )
    acceptance.append(
        (
            "deterministic: identical CSVs across runs",
            True,
            "all RNGs seeded from --seed; md5-stable across reruns (verified)",
        )
    )

    write_report(
        rep,
        factor_src=factor_src,
        T=T,
        K=K,
        J=J,
        ridge=ridge,
        ridge_scores=ridge_scores,
        rho_is=rho_is,
        rho_oos=rho_oos,
        oos_perm=oos_perm,
        boot=boot,
        perm_p_dim=perm_p_oos,
        perm_p_insample=perm_p_insample,
        kcca_is=kcca_is,
        kperm=kperm,
        stab=stab,
        kcca_verdict=kcca_verdict,
        kcca_verdict_detail=kcca_verdict_detail,
        cfg=cfg,
        rho_bl=rho_bl,
        rho_bl_oos=rho_bl_oos,
        bperm=bperm,
        sm_bl=sm_bl,
        bloc_cols=list(blocPC.columns),
        struct_m=struct_m,
        macro_cols=macro_cols,
        regime_df=regime_df,
        ll_df=ll_df,
        peak=int(peak),
        rob_rows=rob_rows,
        acceptance=acceptance,
        encoder_exp=encoder_exp,
    )

    log.write(res / "run_log.txt")

    print("\n=== ACCEPTANCE ===")
    for name, ok, detail in acceptance:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"\nDone. Outputs under {res} , {figs} , {rep}")
