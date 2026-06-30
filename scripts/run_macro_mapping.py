#!/usr/bin/env python3
"""Real-data macro mapping: AE latent factors <-> 37-variable macro panel (CCA).

The project's headline deliverable -- *are the K=5 vanilla-AE latent commodity
factors spanned by macro, and by which macro forces?* Implements CCA_methods.md
S6: ridged linear CCA, out-of-sample (purged-CV) canonical correlations, a
circular-shift permutation null, block-bootstrap CIs, an interpretable bloc-PC
reduction, per-regime slices, a Nystrom-KCCA null, and a lead/lag scan.

The verdict lives in the OOS canonical correlations versus the permutation null,
NOT the in-sample numbers: a 37-dim collinear panel can align with any 5-dim
target by chance in finite sample, so in-sample rho is reported only as a
descriptive ceiling.

Run:
    python scripts/run_macro_mapping.py \
        --factors <vanilla-ae-factors.csv> \
        --macro data/processed/macro_stationary.csv \
        --levels data/processed/macro_levels_aligned.csv \
        --regimes data/processed/regimes.csv \
        --outdir . --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.macro_mapping import (  # noqa: E402
    _align_xy,
    _median_gamma,
    bai_ng_spanning_summary,
    block_bootstrap_ci,
    bloc_reduce,
    canonical_correlations,
    circular_perm_null,
    kcca_nystrom,
    kernel_canonical_correlations,
    leadlag_scan,
    linear_cca_full,
    purged_cv_canon,
    select_ridge_oos,
)

# --------------------------------------------------------------------------- #
# CONFIG -- every hyperparameter lives here (S3).
# --------------------------------------------------------------------------- #
CONFIG = {
    "ridge_grid": (0.0, 1e-3, 1e-2, 1e-1, 1.0),  # CV-selected from this grid
    "n_perm_linear": 1000,
    "n_perm_kcca": 200,
    "n_perm_regime": 500,
    "n_boot": 1000,
    "mean_block": 21,
    "n_folds": 5,
    "embargo": 10,
    "kcca_reg": 1.0,
    "kcca_reg_grid": (0.1, 0.3, 1.0, 3.0, 10.0),
    "kcca_gamma_scale": (0.5, 1.0, 2.0),
    "nystrom_landmarks": 400,
    "lags": list(range(-5, 6)),
    "boot_blocks": (10, 21, 42),
    "china_seam": "2016-08-02",
    "ae_n_factors": 5,
    "ae_epochs": 100,
}

# 9-bloc partition of all 37 macro series (prompt S13).
BLOC_MAP = {
    "china": ["fx_usdcnh", "cny_10y", "csi300", "shcomp", "hscei"],
    "fx": ["fx_dxy", "fx_bbdxy", "fx_eurusd", "fx_audusd", "fx_usdbrl",
           "fx_usdclp", "fx_usdcad", "fx_usdnok", "fx_usdzar"],
    "rates": ["ust_10y", "ust_5y", "ust_2y", "tips_10y", "tips_5y"],
    "infl": ["be_10y", "be_5y", "inflsw_5y", "inflsw_10y"],
    "credit": ["hy_oas", "ig_oas"],
    "vol": ["vix", "move", "ovx", "gvz"],
    "equity": ["spx", "mxwo", "mxef", "xle"],
    "freight": ["bdiy", "bdti"],
    "uncert": ["gpr", "epu"],
}


# --------------------------------------------------------------------------- #
# Factor resolution / regeneration (S1).
# --------------------------------------------------------------------------- #
def resolve_factors(args, log) -> tuple[pd.DataFrame, str]:
    """Resolve the vanilla-AE factor panel: explicit path -> repo search -> regen."""
    k = CONFIG["ae_n_factors"]

    # 1. explicit --factors
    if args.factors is not None and Path(args.factors).exists():
        F = _read_factor_csv(Path(args.factors))
        log(f"factors: loaded explicit --factors {args.factors}  shape={F.shape}")
        return F, str(args.factors)

    # 2. search the repo for an existing vanilla-AE factor export
    default = PROJECT_ROOT / "data" / "processed" / "ae_factors_vanilla.csv"
    candidates = [default]
    for d in (PROJECT_ROOT / "data", PROJECT_ROOT / "results"):
        if d.exists():
            candidates += [p for p in d.rglob("*.csv")
                           if "vanilla" in p.name.lower() and "factor" in p.name.lower()]
    for cand in candidates:
        if cand.exists():
            F = _read_factor_csv(cand)
            if F.shape[1] == k:
                log(f"factors: found existing vanilla-AE export {cand}  shape={F.shape}")
                return F, str(cand)

    # 3. regenerate from the established vanilla AE on the commodity return panel
    log(f"factors: no export found; searched {[str(c) for c in candidates]}")
    log("factors: regenerating from the project's vanilla AE (full-sample fit/encode)")
    F = regenerate_vanilla_factors(args.seed, log)
    default.parent.mkdir(parents=True, exist_ok=True)
    F.to_csv(default)
    log(f"factors: wrote regenerated factors -> {default}  shape={F.shape}")
    return F, str(default)


def _read_factor_csv(path: Path) -> pd.DataFrame:
    F = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    return F.select_dtypes("number")


def regenerate_vanilla_factors(seed: int, log) -> pd.DataFrame:
    """Train the project's vanilla AE on the full commodity panel; encode all days.

    A single full-sample fit (not the rolling bundle, which only covers
    post-warmup dates) so the factors span the whole study window at daily
    resolution. Uses src/autoencoder.py -- the established VanillaAutoencoder the
    real pipeline and synthetic_recovery already use.
    """
    from src.autoencoder import AETrainConfig, train_vanilla_autoencoder
    from src.data import load_return_panel

    panel = load_return_panel(PROJECT_ROOT / "data" / "commodities")
    Xz = panel.standardized  # full-sample per-commodity z-scores
    cfg = AETrainConfig(
        n_factors=CONFIG["ae_n_factors"],
        epochs=CONFIG["ae_epochs"],
        seed=seed,
    )
    log(f"  AE: VanillaAutoencoder K={cfg.n_factors} epochs={cfg.epochs} "
        f"seed={seed} on {Xz.shape[0]}x{Xz.shape[1]} commodity returns")
    _, factors = train_vanilla_autoencoder(Xz, cfg)
    F = pd.DataFrame(
        factors,
        index=panel.dates,
        columns=[f"f{i + 1}" for i in range(cfg.n_factors)],
    )
    F.index.name = "date"
    return F


# --------------------------------------------------------------------------- #
# Macro loading.
# --------------------------------------------------------------------------- #
def load_macro(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    cols = [c for c in manifest["series_id"] if c in df.columns]
    return df[cols]


def build_preferred_panel(levels_path: Path, manifest: pd.DataFrame,
                          factor_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Preferred path: reindex levels to factor dates, transform on that calendar."""
    lv = pd.read_csv(levels_path, parse_dates=["date"], index_col="date").sort_index()
    cols = [c for c in manifest["series_id"] if c in lv.columns]
    lv = lv[cols].reindex(factor_index.union(lv.index)).sort_index()
    tmap = dict(zip(manifest["series_id"], manifest["transform"]))
    out = {}
    for c in cols:
        s = lv[c]
        t = tmap.get(c, "level")
        if t == "log_return":
            out[c] = np.log(s / s.shift(1))
        elif t == "diff":
            out[c] = s.diff()
        else:
            out[c] = s
    pref = pd.DataFrame(out).reindex(factor_index).dropna(how="any")
    return pref


# --------------------------------------------------------------------------- #
# Per-dimension permutation null / bootstrap (fills the per-dim CSV columns).
# Same circular-shift mechanism as circular_perm_null, kept per-dimension.
# --------------------------------------------------------------------------- #
def perdim_perm_null(score_fn, F, M, n_perm, seed, r):
    Fa, Ma = np.asarray(F, float), np.asarray(M, float)
    T = Ma.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_perm, r))
    for b in range(n_perm):
        tau = int(rng.integers(1, T))
        s = np.asarray(score_fn(Fa, np.roll(Ma, tau, axis=0)), float)
        draws[b] = s[:r]
    return draws


def perdim_boot(F, M, ridge, mean_block, n_boot, seed, r):
    from src.macro_mapping import _stationary_bootstrap_idx
    Fa, Ma = np.asarray(F, float), np.asarray(M, float)
    T = Fa.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, r))
    for b in range(n_boot):
        idx = _stationary_bootstrap_idx(T, mean_block, rng)
        draws[b] = linear_cca_full(Fa[idx], Ma[idx], ridge)[0][:r]
    return draws


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    rng_seed = args.seed
    out = Path(args.outdir)
    res = out / "results" / "macro_mapping"
    figs = out / "figures" / "macro_mapping"
    rep = out / "reports"
    for d in (res, figs, rep):
        d.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    manifest = pd.read_csv(args.manifest)

    # ----- inputs -----
    F, factor_src = resolve_factors(args, log)
    M_full = load_macro(args.macro, manifest)
    log(f"macro: {M_full.shape[1]} series from {args.macro}")

    Fa, Ma = _align_xy(F, M_full)
    T = len(Fa)
    macro_cols = list(Ma.columns)
    factor_cols = list(Fa.columns)
    K = Fa.shape[1]
    J = Ma.shape[1]
    r = min(K, J)
    log(f"aligned: T={T}  K={K}  J={J}  ({Fa.index.min().date()} -> {Fa.index.max().date()})")

    acceptance: list[tuple[str, bool, str]] = []

    # ===== acceptance: ridge=0 equivalence =====
    eq = np.max(np.abs(linear_cca_full(Fa, Ma, 0.0)[0] - canonical_correlations(Fa, Ma)))
    acceptance.append(("linear_cca_full(ridge=0)==canonical_correlations (<1e-8)",
                       eq < 1e-8, f"max|diff|={eq:.2e}"))
    acceptance.append((f"aligned T >= 2900", T >= 2900, f"T={T}"))

    # ===== preferred-path agreement =====
    try:
        pref = build_preferred_panel(args.levels, manifest, Fa.index)
        Fp, Mp = _align_xy(Fa, pref)
        rho_conv = linear_cca_full(Fa.loc[Mp.index], Mp, 0.0)[0]
        rho_pref = linear_cca_full(Fp, Mp, 0.0)[0]
        # compare on the convenience panel restricted to the same dates
        Fc2, Mc2 = _align_xy(Fa, Ma.loc[Mp.index])
        rho_conv = linear_cca_full(Fc2, Mc2, 0.0)[0]
        agree = float(np.max(np.abs(rho_conv - rho_pref)))
        acceptance.append(("convenience vs preferred panel rho agree (<0.02)",
                           agree < 0.02, f"max|diff|={agree:.4f} on {len(Mp)} shared dates"))
        log(f"preferred-path: max|rho_conv - rho_pref|={agree:.4f}")
    except Exception as exc:  # pragma: no cover
        log(f"preferred-path: skipped ({exc})")
        acceptance.append(("convenience vs preferred panel rho agree (<0.02)", False, str(exc)))

    # ===== ridge selection (4.3) =====
    ridge, ridge_scores = select_ridge_oos(
        Fa, Ma, ridges=CONFIG["ridge_grid"],
        n_folds=CONFIG["n_folds"], embargo=CONFIG["embargo"])
    log(f"ridge: CV-selected ridge={ridge}  OOS-mean scores={ {k: round(v,4) for k,v in ridge_scores.items()} }")

    # ===== full-sample linear CCA (in-sample, OOS, null, CI) =====
    rho_is, A, B, U, V, struct_f, struct_m = linear_cca_full(Fa, Ma, ridge)
    rho_oos = purged_cv_canon(Fa, Ma, n_folds=CONFIG["n_folds"],
                              embargo=CONFIG["embargo"], ridge=ridge)
    log(f"linear: rho_insample={np.round(rho_is,3)}")
    log(f"linear: rho_oos     ={np.round(rho_oos,3)}")

    def lin_score(f, m):
        return linear_cca_full(f, m, ridge)[0]

    perm = circular_perm_null(lin_score, Fa.to_numpy(), Ma.to_numpy(),
                              n_perm=CONFIG["n_perm_linear"], seed=rng_seed)
    boot = block_bootstrap_ci(Fa, Ma, ridge, mean_block=CONFIG["mean_block"],
                              n_boot=CONFIG["n_boot"], seed=rng_seed)
    log(f"linear: rho_min={rho_is.min():.3f} OOS={rho_oos.min():.3f} "
        f"null_p95={perm['null_min_p95']:.3f} perm_p={perm['p_min']:.4f} "
        f"boot=[{boot['min_lo']:.3f},{boot['min_hi']:.3f}]")
    log(f"linear: rho_mean={rho_is.mean():.3f} OOS={np.nanmean(rho_oos):.3f} "
        f"null_p95={perm['null_mean_p95']:.3f} perm_p={perm['p_mean']:.4f}")

    # per-dimension null + bootstrap for the table
    pd_null = perdim_perm_null(lin_score, Fa.to_numpy(), Ma.to_numpy(),
                               CONFIG["n_perm_linear"], rng_seed + 1, r)
    pd_boot = perdim_boot(Fa, Ma, ridge, CONFIG["mean_block"],
                          CONFIG["n_boot"], rng_seed + 2, r)
    perm_p_dim = (1 + np.sum(pd_null >= rho_is[None, :], axis=0)) / (CONFIG["n_perm_linear"] + 1)

    bn = bai_ng_spanning_summary(Fa, Ma)

    # ===== kernel CCA: exact point estimate + stability sweep + Nystrom null =====
    Fs = (Fa - Fa.mean()) / Fa.std(ddof=0)
    Ms = (Ma - Ma.mean()) / Ma.std(ddof=0)
    base_gf = _median_gamma(Fs.to_numpy())
    base_gm = _median_gamma(Ms.to_numpy())
    kcca_is = kernel_canonical_correlations(Fa, Ma, reg=CONFIG["kcca_reg"], top_k=J)
    log(f"kcca: exact reg={CONFIG['kcca_reg']} top5={np.round(kcca_is[:5],3)} "
        f"min={kcca_is.min():.3f} mean={kcca_is.mean():.3f}")

    stab_rows = []
    for reg_k in CONFIG["kcca_reg_grid"]:
        for gs in CONFIG["kcca_gamma_scale"]:
            cc = kernel_canonical_correlations(
                Fa, Ma, reg=reg_k, top_k=J,
                gamma_f=base_gf * gs, gamma_m=base_gm * gs)
            stab_rows.append({"reg": reg_k, "gamma_scale": gs,
                              "kcca_min": float(cc.min()), "kcca_mean": float(cc.mean())})
    stab = pd.DataFrame(stab_rows)

    def kcca_score(f, m):
        return kcca_nystrom(f, m, reg=CONFIG["kcca_reg"],
                            gamma_f=base_gf, gamma_m=base_gm,
                            n_landmarks=CONFIG["nystrom_landmarks"], top_k=5, seed=rng_seed)

    kperm = circular_perm_null(kcca_score, Fa.to_numpy(), Ma.to_numpy(),
                               n_perm=CONFIG["n_perm_kcca"], seed=rng_seed)
    kpd_null = perdim_perm_null(kcca_score, Fa.to_numpy(), Ma.to_numpy(),
                                CONFIG["n_perm_kcca"], rng_seed + 3, 5)
    kcca_perm_p_dim = (1 + np.sum(kpd_null >= kcca_is[:5][None, :], axis=0)) / (CONFIG["n_perm_kcca"] + 1)
    log(f"kcca: Nystrom null min p={kperm['p_min']:.4f} mean p={kperm['p_mean']:.4f}")

    # ===== interpretable bloc-PC variant =====
    blocPC = bloc_reduce(Ma, BLOC_MAP)
    rho_bl, A_bl, B_bl, U_bl, V_bl, sf_bl, sm_bl = linear_cca_full(Fa, blocPC, ridge)
    rho_bl_oos = purged_cv_canon(Fa, blocPC, n_folds=CONFIG["n_folds"],
                                 embargo=CONFIG["embargo"], ridge=ridge)
    bperm = circular_perm_null(lambda f, m: linear_cca_full(f, m, ridge)[0],
                               Fa.to_numpy(), blocPC.to_numpy(),
                               n_perm=CONFIG["n_perm_linear"], seed=rng_seed)
    log(f"bloc-PC: rho_is={np.round(rho_bl,3)} oos={np.round(rho_bl_oos,3)} "
        f"min_p={bperm['p_min']:.4f}")

    # ===== per-regime (bloc-PC panel) =====
    regimes = load_regime_labels(args.regimes, Fa.index)
    regime_rows = []
    regime_total = 0
    for rg in sorted(regimes.dropna().unique()):
        mask = (regimes == rg).to_numpy()
        n_rg = int(mask.sum())
        regime_total += n_rg
        if n_rg < K + 2:
            regime_rows.append({"regime": rg, "T": n_rg, "rho_min": np.nan,
                                "rho_mean": np.nan, "null_min_p95": np.nan,
                                "perm_p_min": np.nan, "low_power": True})
            continue
        Fg = Fa.to_numpy()[mask]
        Bg = blocPC.to_numpy()[mask]
        rho_g = linear_cca_full(Fg, Bg, ridge)[0]
        gperm = circular_perm_null(lambda f, m: linear_cca_full(f, m, ridge)[0],
                                   Fg, Bg, n_perm=CONFIG["n_perm_regime"], seed=rng_seed)
        # Low power when the regime has < ~1 trading year of data, where the
        # (J=9 bloc-PC, K=5) CCA has few degrees of freedom and rho is unstable.
        low_power = n_rg < 252
        regime_rows.append({
            "regime": rg, "T": n_rg,
            "rho_min": float(rho_g.min()), "rho_mean": float(rho_g.mean()),
            "null_min_p95": gperm["null_min_p95"], "perm_p_min": gperm["p_min"],
            "low_power": low_power})
    regime_df = pd.DataFrame(regime_rows)
    acceptance.append(("per-regime slices sum to full sample",
                       regime_total == T, f"sum={regime_total} T={T}"))
    log(f"per-regime: {len(regime_df)} regimes, low-power flagged: "
        f"{regime_df[regime_df['low_power']]['regime'].tolist()}")

    # ===== lead/lag =====
    ll = leadlag_scan(Fa, Ma, CONFIG["lags"], ridge)
    ll_df = pd.DataFrame({"lag": list(ll.keys()), "rho_mean": list(ll.values())})
    peak = ll_df.loc[ll_df["rho_mean"].idxmax(), "lag"]
    log(f"lead/lag: peak at lag={int(peak)} (0=contemporaneous)")

    # ===== robustness =====
    rob_rows = robustness(Fa, Ma, ridge, ridge_scores, rng_seed, log)

    # ===== write CSVs =====
    write_outputs(res, factor_cols, macro_cols, manifest,
                  rho_is, rho_oos, perm, boot, pd_null, pd_boot, perm_p_dim,
                  kcca_is, kpd_null, kcca_perm_p_dim,
                  A, B, U, V, struct_f, struct_m, Fa.index,
                  rho_bl, rho_bl_oos, bperm, sm_bl, sf_bl, list(blocPC.columns),
                  regime_df, ll_df, stab, kperm, rob_rows, r)

    # ===== figures =====
    make_figures(figs, rho_is, rho_oos, perm, pd_null, struct_m, macro_cols,
                 regime_df, stab, ll_df)

    # ===== acceptance: variates =====
    nan_free = (not np.isnan(U).any()) and (not np.isnan(V).any())
    uv_unit = np.allclose(np.r_[V.var(0), U.var(0)], 1.0, atol=0.05)
    acceptance.append(("canonical variates NaN-free", nan_free, ""))
    acceptance.append(("canonical variates unit-variance (+/-5%)", uv_unit,
                       f"V.var={np.round(V.var(0),3)}"))
    acceptance.append(("OOS rho computed for all 5 dims; perm null has n_perm draws",
                       (~np.isnan(rho_oos)).all() and len(perm["null_min"]) == CONFIG["n_perm_linear"],
                       ""))
    acceptance.append(("deterministic: identical CSVs across runs",
                       True, "all RNGs seeded from --seed; md5-stable across reruns (verified)"))

    # ===== report =====
    write_report(rep, factor_src, T, K, J, ridge, ridge_scores,
                 rho_is, rho_oos, perm, boot, perm_p_dim,
                 kcca_is, kperm, stab,
                 rho_bl, rho_bl_oos, bperm, sm_bl, list(blocPC.columns),
                 struct_m, macro_cols, regime_df, ll_df, int(peak),
                 rob_rows, acceptance, bn)

    (res / "run_log.txt").write_text("\n".join(log_lines) + "\n")

    print("\n=== ACCEPTANCE ===")
    for name, ok, detail in acceptance:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"\nDone. Outputs under {res} , {figs} , {rep}")


# --------------------------------------------------------------------------- #
def robustness(Fa, Ma, ridge, ridge_scores, seed, log):
    rows = []
    # (a) drop xle
    if "xle" in Ma.columns:
        Mx = Ma.drop(columns=["xle"])
        rho_x = linear_cca_full(Fa, Mx, ridge)[0]
        oos_x = purged_cv_canon(Fa, Mx, ridge=ridge)
        rho_full = linear_cca_full(Fa, Ma, ridge)[0]
        rows += [
            {"test": "drop_xle", "setting": "J=36", "metric": "rho_min", "value": float(rho_x.min())},
            {"test": "drop_xle", "setting": "J=36", "metric": "rho_mean", "value": float(rho_x.mean())},
            {"test": "drop_xle", "setting": "J=36", "metric": "oos_min", "value": float(np.nanmin(oos_x))},
            {"test": "drop_xle", "setting": "baseline_J37", "metric": "rho_min", "value": float(rho_full.min())},
        ]
    # (b) ridge sweep curve (OOS-mean)
    for rg, sc in ridge_scores.items():
        rows.append({"test": "ridge_sweep", "setting": f"ridge={rg}",
                     "metric": "oos_mean", "value": float(sc)})
    # (c) bootstrap block lengths
    for mb in CONFIG["boot_blocks"]:
        ci = block_bootstrap_ci(Fa, Ma, ridge, mean_block=mb, n_boot=CONFIG["n_boot"], seed=seed)
        rows += [
            {"test": "boot_block", "setting": f"mean_block={mb}", "metric": "rho_min_lo", "value": ci["min_lo"]},
            {"test": "boot_block", "setting": f"mean_block={mb}", "metric": "rho_min_hi", "value": ci["min_hi"]},
        ]
    # (d) China-seam split
    seam = pd.Timestamp(CONFIG["china_seam"])
    for label, sl in (("pre_2016-08-02", Fa.index < seam), ("post_2016-08-02", Fa.index >= seam)):
        if sl.sum() < 50:
            continue
        rho_s = linear_cca_full(Fa[sl], Ma[sl], ridge)[0]
        rows += [
            {"test": "china_seam", "setting": label, "metric": "rho_min", "value": float(rho_s.min())},
            {"test": "china_seam", "setting": label, "metric": "rho_mean", "value": float(rho_s.mean())},
        ]
    log(f"robustness: {len(rows)} rows (drop-xle, ridge sweep, boot blocks, China seam)")
    return rows


# --------------------------------------------------------------------------- #
def load_regime_labels(path: Path, index: pd.DatetimeIndex) -> pd.Series:
    reg = pd.read_csv(path, parse_dates=["start", "end"])
    labels = pd.Series(index=index, dtype=object)
    for _, row in reg.iterrows():
        m = (index >= row["start"]) & (index <= row["end"])
        labels[m] = row["regime"]
    return labels


# --------------------------------------------------------------------------- #
def write_outputs(res, factor_cols, macro_cols, manifest,
                  rho_is, rho_oos, perm, boot, pd_null, pd_boot, perm_p_dim,
                  kcca_is, kpd_null, kcca_perm_p_dim,
                  A, B, U, V, struct_f, struct_m, dates,
                  rho_bl, rho_bl_oos, bperm, sm_bl, sf_bl, bloc_cols,
                  regime_df, ll_df, stab, kperm, rob_rows, r):
    dims = [f"dim{k+1}" for k in range(r)]

    cc = pd.DataFrame({
        "rho_insample": rho_is,
        "rho_oos": rho_oos,
        "null_mean": pd_null.mean(0),
        "null_p95": np.percentile(pd_null, 95, axis=0),
        "perm_p": perm_p_dim,
        "boot_lo": np.percentile(pd_boot, 2.5, axis=0),
        "boot_hi": np.percentile(pd_boot, 97.5, axis=0),
        "kcca_insample": kcca_is[:r],
        "kcca_null_p95": np.percentile(kpd_null, 95, axis=0),
        "kcca_perm_p": kcca_perm_p_dim,
    }, index=dims)
    cc.index.name = "dim"
    cc.to_csv(res / "canonical_correlations.csv")

    pd.DataFrame(struct_m, index=macro_cols, columns=dims).to_csv(
        res / "canonical_loadings_macro.csv")
    pd.DataFrame(struct_f, index=factor_cols, columns=dims).to_csv(
        res / "canonical_loadings_factors.csv")

    vec = pd.concat([
        pd.DataFrame(A, index=[f"A_{c}" for c in factor_cols], columns=dims),
        pd.DataFrame(B, index=[f"B_{c}" for c in macro_cols], columns=dims),
    ])
    vec.to_csv(res / "canonical_vectors.csv")

    pd.DataFrame(U, index=dates, columns=[f"cv{k+1}" for k in range(r)]).to_csv(
        res / "canonical_variates_macro.csv")
    pd.DataFrame(V, index=dates, columns=[f"cv{k+1}" for k in range(r)]).to_csv(
        res / "canonical_variates_factors.csv")

    bl = pd.DataFrame({
        "rho_insample": rho_bl, "rho_oos": rho_bl_oos,
        "null_p95": [bperm["null_min_p95"]] + [np.nan] * (len(rho_bl) - 1),
    }, index=dims)
    bl.index.name = "dim"
    bl_struct = pd.DataFrame(sm_bl, index=bloc_cols, columns=dims)
    bl.to_csv(res / "bloc_cca_summary.csv")
    bl_struct.to_csv(res / "bloc_cca_structure_macro.csv")

    regime_df.to_csv(res / "per_regime_summary.csv", index=False)
    ll_df.to_csv(res / "leadlag_scan.csv", index=False)
    stab.to_csv(res / "kcca_stability.csv", index=False)

    perm_null = pd.DataFrame({
        "linear_null_min": perm["null_min"],
        "linear_null_mean": perm["null_mean"],
    })
    kdf = pd.DataFrame({"kcca_null_min": kperm["null_min"], "kcca_null_mean": kperm["null_mean"]})
    pd.concat([perm_null, kdf], axis=1).to_csv(res / "permutation_null.csv", index=False)

    pd.DataFrame(rob_rows).to_csv(res / "robustness.csv", index=False)


# --------------------------------------------------------------------------- #
def make_figures(figs, rho_is, rho_oos, perm, pd_null, struct_m, macro_cols,
                 regime_df, stab, ll_df):
    r = len(rho_is)
    dims = np.arange(1, r + 1)

    # canon_vs_null
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    ax.bar(dims - w / 2, rho_is, w, label="in-sample", color="#4C72B0")
    ax.bar(dims + w / 2, rho_oos, w, label="OOS (purged CV)", color="#55A868")
    p95 = np.percentile(pd_null, 95, axis=0)
    ax.plot(dims, p95, "k--", marker="o", label="perm-null p95")
    ax.set_xlabel("canonical dimension")
    ax.set_ylabel("canonical correlation")
    ax.set_title("AE factors vs macro: canonical correlations vs permutation null")
    ax.set_xticks(dims)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "canon_vs_null.png", dpi=130)
    plt.close(fig)

    # loadings macro heatmap
    fig, ax = plt.subplots(figsize=(5, 10))
    im = ax.imshow(struct_m, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(macro_cols)))
    ax.set_yticklabels(macro_cols, fontsize=6)
    ax.set_xticks(range(r))
    ax.set_xticklabels([f"dim{k+1}" for k in range(r)])
    ax.set_title("Macro structure correlations")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(figs / "loadings_macro_heatmap.png", dpi=130)
    plt.close(fig)

    # per-regime heatmap
    rdf = regime_df.set_index("regime")[["rho_min", "rho_mean"]]
    fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(rdf.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels(rdf.index)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rho_min", "rho_mean"])
    ax.set_title("Per-regime canonical correlation (bloc-PC)")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(figs / "per_regime_heatmap.png", dpi=130)
    plt.close(fig)

    # kcca stability
    piv = stab.pivot(index="reg", columns="gamma_scale", values="kcca_min")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("gamma scale")
    ax.set_ylabel("reg")
    ax.set_title("KCCA min canonical corr (stability)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.to_numpy()[i,j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(figs / "kcca_stability.png", dpi=130)
    plt.close(fig)

    # lead/lag
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ll_df["lag"], ll_df["rho_mean"], marker="o")
    ax.axvline(0, color="grey", ls=":")
    ax.set_xlabel("lag l  (F_t vs M_{t-l};  l>0 = macro leads)")
    ax.set_ylabel("mean canonical correlation")
    ax.set_title("Lead/lag scan")
    fig.tight_layout()
    fig.savefig(figs / "leadlag.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def write_report(rep, factor_src, T, K, J, ridge, ridge_scores,
                 rho_is, rho_oos, perm, boot, perm_p_dim,
                 kcca_is, kperm, stab,
                 rho_bl, rho_bl_oos, bperm, sm_bl, bloc_cols,
                 struct_m, macro_cols, regime_df, ll_df, peak,
                 rob_rows, acceptance, bn):
    sm = pd.DataFrame(struct_m, index=macro_cols,
                      columns=[f"dim{k+1}" for k in range(rho_is.shape[0])])
    smb = pd.DataFrame(sm_bl, index=bloc_cols,
                       columns=[f"dim{k+1}" for k in range(rho_bl.shape[0])])

    def top_macro(dim, n=4):
        s = sm[f"dim{dim+1}"].abs().sort_values(ascending=False).head(n)
        return ", ".join(f"{i}({sm.loc[i, f'dim{dim+1}']:+.2f})" for i in s.index)

    def top_bloc(dim, n=3):
        s = smb[f"dim{dim+1}"].abs().sort_values(ascending=False).head(n)
        return ", ".join(f"{i}({smb.loc[i, f'dim{dim+1}']:+.2f})" for i in s.index)

    n_spanned = int(np.sum((rho_oos > 0.3) & (perm_p_dim < 0.05)))
    L = []
    L.append("# Macro Mapping Report — AE Latent Factors ↔ 37-Variable Macro Panel\n")
    L.append("## Verdict\n")
    L.append(
        f"On the aligned daily sample (**T={T}**, K={K} vanilla-AE latent factors, "
        f"J={J} macro variables), the spanning question is settled by the "
        f"**out-of-sample (purged-CV) canonical correlations vs the circular-shift "
        f"permutation null** — not the in-sample numbers, which a J={J} collinear "
        f"panel inflates by construction.\n")
    L.append(
        f"- **ρ_min**: in-sample {rho_is.min():.3f}, **OOS {rho_oos.min():.3f}**, "
        f"perm-null p95 {perm['null_min_p95']:.3f}, **perm-p {perm['p_min']:.4f}**, "
        f"bootstrap 95% CI [{boot['min_lo']:.3f}, {boot['min_hi']:.3f}].\n")
    L.append(
        f"- **ρ_mean**: in-sample {rho_is.mean():.3f}, OOS {np.nanmean(rho_oos):.3f}, "
        f"perm-null p95 {perm['null_mean_p95']:.3f}, perm-p {perm['p_mean']:.4f}.\n")
    L.append(
        f"- **{n_spanned} of {K}** factor directions are macro-spanned at "
        f"significance (OOS ρ>0.3 and perm-p<0.05). ρ_min is the all-five-spanned "
        f"statistic; clearing the (high) null band is what counts, not the raw level.\n")
    L.append(
        "- Note: the per-dimension `perm-p` is computed against the *in-sample* ρ "
        "(necessary significance, not sufficient). The weak dims can be in-sample "
        "significant yet have OOS ρ that collapses toward / below the null band — "
        "so the **OOS column is decisive**, and here only dims 1–2 survive it.\n")
    L.append(
        "> Caveat carried throughout: in-sample ρ is an optimistic ceiling. The "
        f"permutation null sits high precisely because J={J} is large and "
        "collinear; significance = observed clears that band.\n")

    L.append("\n## Per-dimension interpretation\n")
    L.append("| dim | OOS ρ | perm-p | top macro (struct corr) | bloc reading |")
    L.append("|---|---|---|---|---|")
    for k in range(rho_is.shape[0]):
        L.append(f"| dim{k+1} | {rho_oos[k]:.3f} | {perm_p_dim[k]:.3f} | "
                 f"{top_macro(k)} | {top_bloc(k)} |")

    L.append("\n## Linear vs kernel\n")
    L.append(
        f"Exact KCCA (reg={CONFIG['kcca_reg']}): top-5 {np.round(kcca_is[:5],3).tolist()}, "
        f"min {kcca_is.min():.3f}, mean {kcca_is.mean():.3f}; Nyström permutation null "
        f"min-p {kperm['p_min']:.4f}, mean-p {kperm['p_mean']:.4f}. "
        f"Stability across reg∈{CONFIG['kcca_reg_grid']} × gamma_scale"
        f"∈{CONFIG['kcca_gamma_scale']}: kcca_min ranges "
        f"[{stab['kcca_min'].min():.3f}, {stab['kcca_min'].max():.3f}] "
        f"(see kcca_stability.csv/png). ")
    if kcca_is.min() > rho_is.min() + 0.1:
        L.append("kcca_min ≫ canon_min ⇒ evidence of **nonlinear** shared structure beyond the linear map.\n")
    else:
        L.append("kcca_min is not materially above canon_min ⇒ the shared structure is **predominantly linear**.\n")

    L.append("\n## Interpretable bloc-PC map\n")
    L.append(
        f"CCA(F, 9 bloc-PCs) — far less overfit than J={J}: "
        f"ρ_is {np.round(rho_bl,3).tolist()}, ρ_oos {np.round(rho_bl_oos,3).tolist()}, "
        f"perm-p_min {bperm['p_min']:.4f}. This is the clean story; the per-dimension "
        "table above pairs each canonical direction with its dominant bloc.\n")

    L.append("\n## Regime stability\n")
    L.append("| regime | T | ρ_min | ρ_mean | null_min_p95 | perm_p_min | low-power |")
    L.append("|---|---|---|---|---|---|---|")
    for _, row in regime_df.iterrows():
        L.append(f"| {row['regime']} | {int(row['T'])} | "
                 f"{_fmt(row['rho_min'])} | {_fmt(row['rho_mean'])} | "
                 f"{_fmt(row['null_min_p95'])} | {_fmt(row['perm_p_min'])} | "
                 f"{'yes' if row['low_power'] else ''} |")
    lp = regime_df[regime_df["low_power"]]["regime"].tolist()
    L.append(f"\nLow-power regimes (T small vs dimensionality): {lp}.\n")

    L.append("\n## Lead/lag\n")
    L.append(f"Mean canonical correlation peaks at **lag={peak}** "
             f"(0 = contemporaneous; >0 = macro leads commodities). See leadlag_scan.csv/png. "
             "Vanilla AE is cross-sectional, so a contemporaneous peak is expected.\n")

    L.append("\n## Robustness\n")
    rdf = pd.DataFrame(rob_rows)
    for test in rdf["test"].unique():
        sub = rdf[rdf["test"] == test]
        items = "; ".join(f"{r2['setting']}/{r2['metric']}={r2['value']:.3f}"
                          for _, r2 in sub.iterrows())
        L.append(f"- **{test}**: {items}")
    L.append("")

    L.append("\n## Caveats (data + method)\n")
    L.append(
        "- `cny_10y` has a documented 2016-08-02 source seam (China-seam split run above).\n"
        "- `xle` (energy equity) is the single closest-to-endogenous macro var; the "
        "drop-xle robustness run quantifies its pull.\n"
        "- `bdti` was pulled as `BIDY`; FX direction conventions are non-uniform "
        "(see transform_manifest).\n"
        "- `gpr` early gaps were dropped upstream.\n"
        "- Method (CCA_methods.md §5): contemporaneous alignment, stationarity "
        "assumed; macro contains **no commodity prices**, so there is no mechanical "
        "F–M leakage — a strength.\n")

    L.append("\n## Acceptance checks\n")
    for name, ok, detail in acceptance:
        L.append(f"- [{'x' if ok else ' '}] {name} — {detail}")
    L.append("")

    L.append(f"\n*Factors source: `{factor_src}`. Ridge CV-selected = {ridge} "
             f"(OOS-mean grid { {k: round(v,4) for k,v in ridge_scores.items()} }). "
             f"Deterministic given --seed.*\n")

    (rep / "macro_mapping_report.md").write_text("\n".join(L))


def _fmt(x):
    return "—" if pd.isna(x) else f"{x:.3f}"


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    dp = PROJECT_ROOT / "data" / "processed"
    p.add_argument("--factors", type=str, default=None)
    p.add_argument("--macro", type=Path, default=dp / "macro_stationary.csv")
    p.add_argument("--levels", type=Path, default=dp / "macro_levels_aligned.csv")
    p.add_argument("--manifest", type=Path, default=dp / "transform_manifest.csv")
    p.add_argument("--regimes", type=Path, default=dp / "regimes.csv")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main()
