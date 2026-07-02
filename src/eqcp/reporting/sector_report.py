"""Sector-analysis narrative report."""

from __future__ import annotations

from pathlib import Path

from eqcp.sectors import SectorSpanning


def write_sector_report(
    rep: Path,
    results: dict[str, SectorSpanning],
    *,
    seed: int,
    n_perm: int,
) -> None:
    """Write ``reports/sector_analysis_report.md``."""
    L: list[str] = []
    L.append("# Sector-Wise Macro Spanning Report\n")
    L.append(
        "Each sector's commodity return panel is compressed with the same vanilla "
        "autoencoder and tested for macro spanning with purged-CV out-of-sample "
        "canonical correlations vs a circular-shift permutation null "
        f"(n_perm={n_perm}, seed={seed}). A direction counts as macro-spanned when "
        "its OOS ρ>0.3 and OOS perm-p<0.05.\n"
    )

    L.append("## Headline by block\n")
    L.append("| block | commodities | factors | macro-spanned | OOS ρ (per dim) |")
    L.append("|---|---|---|---|---|")
    for name, r in results.items():
        rho = ", ".join(f"{x:.3f}" for x in r.rho_oos)
        L.append(
            f"| {name} | {r.n_commodities} | {r.n_factors} | "
            f"{r.n_spanned}/{len(r.rho_oos)} | {rho} |"
        )
    L.append("")

    for name, r in results.items():
        L.append(f"\n## {name.capitalize()}\n")
        L.append(
            f"{r.n_commodities} commodities → {r.n_factors} AE factors vs "
            f"{r.n_macro} macro variables; CV-selected ridge={r.ridge:g}. "
            f"**{r.n_spanned} of {len(r.rho_oos)}** directions macro-spanned OOS.\n"
        )
        L.append(
            "| dim | in-sample ρ | OOS ρ | OOS perm-p | spanned | "
            "top macro (struct corr) | bloc reading |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for k in range(len(r.rho_oos)):
            spanned = (r.rho_oos[k] > 0.3) and (r.perm_p_oos[k] < 0.05)
            L.append(
                f"| dim{k + 1} | {r.rho_insample[k]:.3f} | {r.rho_oos[k]:.3f} | "
                f"{r.perm_p_oos[k]:.3f} | {'yes' if spanned else ''} | "
                f"{r.top_macro[k]} | {r.top_bloc[k]} |"
            )
        L.append("")

    L.append("\n## Reading this\n")
    L.append(
        "- The **overall** row is the whole-panel benchmark (same probe as the "
        "macro-mapping report); sector rows decompose it.\n"
        "- OOS ρ near the permutation null band means the direction is *not* linearly "
        "recoverable from macro out-of-sample — it is sector-idiosyncratic.\n"
        "- `top macro` / `bloc reading` name the macro variables and 9-bloc PCs that "
        "load on each direction (structure correlations), i.e. the interpretable driver.\n"
    )
    L.append(
        f"\n*Deterministic given --seed={seed}. Autoencoder latents are identified only "
        "up to rotation/sign/permutation, so CCA scores the factor space (the invariant "
        "object), not individual coordinates.*\n"
    )

    (rep / "sector_analysis_report.md").write_text("\n".join(L))
