"""Macro-mapping narrative report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eqcp.config import MacroMappingConfig


def _fmt(x) -> str:
    return "—" if pd.isna(x) else f"{x:.3f}"


def write_report(
    rep: Path,
    *,
    factor_src: str,
    T: int,
    K: int,
    J: int,
    ridge: float,
    ridge_scores: dict[float, float],
    rho_is: np.ndarray,
    rho_oos: np.ndarray,
    oos_perm: dict,
    boot: dict,
    perm_p_dim: np.ndarray,
    perm_p_insample: np.ndarray,
    kcca_is: np.ndarray,
    kperm: dict,
    stab: pd.DataFrame,
    kcca_verdict: str,
    kcca_verdict_detail: str,
    cfg: MacroMappingConfig,
    rho_bl: np.ndarray,
    rho_bl_oos: np.ndarray,
    bperm: dict,
    sm_bl: np.ndarray,
    bloc_cols: list[str],
    struct_m: np.ndarray,
    macro_cols: list[str],
    regime_df: pd.DataFrame,
    ll_df: pd.DataFrame,
    peak: int,
    rob_rows: list[dict],
    acceptance: list[tuple[str, bool, str]],
    encoder_exp: pd.DataFrame,
) -> None:
    """Write ``reports/macro_mapping_report.md``."""
    r = rho_is.shape[0]
    sm = pd.DataFrame(struct_m, index=macro_cols, columns=[f"dim{k + 1}" for k in range(r)])
    smb = pd.DataFrame(
        sm_bl, index=bloc_cols, columns=[f"dim{k + 1}" for k in range(rho_bl.shape[0])]
    )

    def top_macro(dim: int, n: int = 4) -> str:
        s = sm[f"dim{dim + 1}"].abs().sort_values(ascending=False).head(n)
        return ", ".join(f"{i}({sm.loc[i, f'dim{dim + 1}']:+.2f})" for i in s.index)

    def top_bloc(dim: int, n: int = 3) -> str:
        s = smb[f"dim{dim + 1}"].abs().sort_values(ascending=False).head(n)
        return ", ".join(f"{i}({smb.loc[i, f'dim{dim + 1}']:+.2f})" for i in s.index)

    n_spanned = int(np.sum((rho_oos > 0.3) & (perm_p_dim < 0.05)))
    L: list[str] = []
    L.append("# Macro Mapping Report — AE Latent Factors ↔ 37-Variable Macro Panel\n")
    L.append("## Verdict\n")
    L.append(
        f"On the aligned daily sample (**T={T}**, K={K} vanilla-AE latent factors, "
        f"J={J} macro variables), the spanning question is settled by the "
        f"**out-of-sample (purged-CV) canonical correlations vs the circular-shift "
        f"OOS permutation null** — not the in-sample numbers, which a J={J} collinear "
        f"panel inflates by construction.\n"
    )
    L.append(
        f"- **ρ_min**: in-sample {rho_is.min():.3f}, **OOS {rho_oos.min():.3f}**, "
        f"OOS perm-null p95 {oos_perm['null_min_p95']:.3f}, "
        f"**OOS perm-p {oos_perm['p_min']:.4f}**, "
        f"bootstrap 95% CI [{boot['min_lo']:.3f}, {boot['min_hi']:.3f}].\n"
    )
    L.append(
        f"- **ρ_mean**: in-sample {rho_is.mean():.3f}, OOS {np.nanmean(rho_oos):.3f}, "
        f"OOS perm-null p95 {oos_perm['null_mean_p95']:.3f}, "
        f"OOS perm-p {oos_perm['p_mean']:.4f}.\n"
    )
    L.append(
        f"- **{n_spanned} of {K}** factor directions are macro-spanned at "
        f"significance (OOS ρ>0.3 and OOS perm-p<0.05). ρ_min is the all-five-spanned "
        f"statistic; clearing the (high) null band is what counts, not the raw level.\n"
    )
    L.append(
        "- Per-dimension **OOS perm-p** is the headline significance column. "
        "In-sample perm-p (`perm_p_insample` in CSV) is retained as a necessary-condition "
        "check only — weak dims can be in-sample significant yet have OOS ρ that collapses "
        "toward / below the null band.\n"
    )
    L.append(
        "> Caveat carried throughout: in-sample ρ is an optimistic ceiling. The "
        f"permutation null sits high precisely because J={J} is large and "
        "collinear; significance = observed OOS ρ clears that band.\n"
    )

    L.append("\n## Per-dimension interpretation\n")
    L.append(
        "| dim | OOS ρ | OOS perm-p | in-sample perm-p | top macro (struct corr) | bloc reading |"
    )
    L.append("|---|---|---|---|---|---|")
    for k in range(r):
        L.append(
            f"| dim{k + 1} | {rho_oos[k]:.3f} | {perm_p_dim[k]:.3f} | "
            f"{perm_p_insample[k]:.3f} | {top_macro(k)} | {top_bloc(k)} |"
        )

    L.append("\n## Linear vs kernel\n")
    L.append(
        f"Exact KCCA (reg={cfg.kcca_reg}): top-5 {np.round(kcca_is[:5], 3).tolist()}, "
        f"min {kcca_is.min():.3f}, mean {kcca_is.mean():.3f}; Nyström permutation null "
        f"min-p {kperm['p_min']:.4f}, mean-p {kperm['p_mean']:.4f}. "
        f"Stability across reg∈{cfg.kcca_reg_grid} × gamma_scale"
        f"∈{cfg.kcca_gamma_scale}: kcca_min ranges "
        f"[{stab['kcca_min'].min():.3f}, {stab['kcca_min'].max():.3f}] "
        f"(see kcca_stability.csv/png). "
    )
    L.append(f"**KCCA verdict: {kcca_verdict}** — {kcca_verdict_detail}\n")

    L.append("\n## Interpretable bloc-PC map\n")
    L.append(
        f"CCA(F, 9 bloc-PCs) — far less overfit than J={J}: "
        f"ρ_is {np.round(rho_bl, 3).tolist()}, ρ_oos {np.round(rho_bl_oos, 3).tolist()}, "
        f"perm-p_min {bperm['p_min']:.4f}. This is the clean story; the per-dimension "
        "table above pairs each canonical direction with its dominant bloc.\n"
    )

    L.append("\n## Encoder activation experiment\n")
    L.append(
        "ReLU one-sided latents may suppress weak macro-spanned dimensions. "
        "Each row retrains the vanilla AE with a different encoder activation on the "
        "full commodity panel and recounts OOS macro-spanning (ρ>0.3, OOS perm-p<0.05).\n"
    )
    L.append("| activation | n_spanned | OOS ρ_min | OOS ρ_mean | OOS perm-p_min | ridge |")
    L.append("|---|---|---|---|---|---|")
    for _, row in encoder_exp.iterrows():
        L.append(
            f"| {row['activation']} | {int(row['n_spanned'])} | "
            f"{row['rho_oos_min']:.3f} | {row['rho_oos_mean']:.3f} | "
            f"{row['perm_p_min']:.4f} | {row['ridge']:.4g} |"
        )
    L.append("")

    L.append("\n## Regime stability\n")
    L.append("| regime | T | ρ_min | ρ_mean | null_min_p95 | perm_p_min | low-power |")
    L.append("|---|---|---|---|---|---|---|")
    for _, row in regime_df.iterrows():
        L.append(
            f"| {row['regime']} | {int(row['T'])} | "
            f"{_fmt(row['rho_min'])} | {_fmt(row['rho_mean'])} | "
            f"{_fmt(row['null_min_p95'])} | {_fmt(row['perm_p_min'])} | "
            f"{'yes' if row['low_power'] else ''} |"
        )
    lp = regime_df[regime_df["low_power"]]["regime"].tolist()
    L.append(f"\nLow-power regimes (T small vs dimensionality): {lp}.\n")

    L.append("\n## Lead/lag\n")
    L.append(
        f"Mean canonical correlation peaks at **lag={peak}** "
        f"(0 = contemporaneous; >0 = macro leads commodities). See leadlag_scan.csv/png. "
        "Vanilla AE is cross-sectional, so a contemporaneous peak is expected.\n"
    )

    L.append("\n## Robustness\n")
    rdf = pd.DataFrame(rob_rows)
    for test in rdf["test"].unique():
        sub = rdf[rdf["test"] == test]
        items = "; ".join(
            f"{r2['setting']}/{r2['metric']}={r2['value']:.3f}" for _, r2 in sub.iterrows()
        )
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
        "F–M leakage — a strength.\n"
    )

    L.append("\n## Acceptance checks\n")
    for name, ok, detail in acceptance:
        L.append(f"- [{'x' if ok else ' '}] {name} — {detail}")
    L.append("")

    L.append(
        f"\n*Factors source: `{factor_src}`. Ridge CV-selected = {ridge} "
        f"(OOS-mean grid { {k: round(v, 4) for k, v in ridge_scores.items()} }). "
        f"Deterministic given --seed.*\n"
    )

    (rep / "macro_mapping_report.md").write_text("\n".join(L))
