"""Sector-wise factor extraction and macro-spanning analysis.

Runs the same autoencoder -> CCA machinery used for the whole panel, but on each
commodity sector (energy / agriculture / metals) independently. The output tells
the economics team *which macro directions span each sector* and how strongly,
so the aggregate finding can be decomposed into sector-level stories.

The design deliberately reuses the validated numerics (``train_factors``,
``linear_cca_full``, ``purged_cv_canon``, ``perdim_perm_null_oos``, ``bloc_reduce``)
so a sector result is directly comparable to the overall macro-mapping result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eqcp.cca.inference import perdim_perm_null_oos, purged_cv_canon, select_ridge_oos
from eqcp.cca.linear import align_xy, linear_cca_full
from eqcp.cca.reduce import bloc_reduce
from eqcp.config import FactorModelConfig
from eqcp.factors.extract import FactorModelType, train_factors
from eqcp.io.commodities import SECTORS, load_return_panel

# Sensible default bottleneck width for a single sector (5-9 series). Kept a
# genuine compression relative to the sector size; capped per sector below.
DEFAULT_SECTOR_FACTORS = 3


@dataclass
class SectorFactors:
    """Full-sample AE latents for one sector, on the panel calendar."""

    sector: str
    commodities: list[str]
    n_factors: int
    factors: pd.DataFrame


@dataclass
class SectorSpanning:
    """CCA spanning result of one factor block against the macro panel."""

    name: str
    n_commodities: int
    n_factors: int
    n_macro: int
    ridge: float
    rho_insample: np.ndarray
    rho_oos: np.ndarray
    perm_p_oos: np.ndarray
    n_spanned: int
    struct_macro: np.ndarray  # (J, r) structure correlations
    macro_cols: list[str]
    bloc_struct: pd.DataFrame  # (n_blocs, r) bloc structure correlations
    top_macro: list[str] = field(default_factory=list)
    top_bloc: list[str] = field(default_factory=list)


def _factor_count(n_members: int, n_factors: int) -> int:
    """A genuine bottleneck: never ask for >= as many factors as series."""
    return int(max(1, min(n_factors, n_members - 1)))


def extract_sector_factors(
    sector: str,
    members: list[str],
    factor_cfg: FactorModelConfig,
    n_factors: int = DEFAULT_SECTOR_FACTORS,
    activation: str | None = None,
    model_type: FactorModelType = "vanilla",
) -> SectorFactors:
    """Train the factor model on one sector's standardized returns (full sample)."""
    panel = load_return_panel(commodities=members)
    k = _factor_count(len(members), n_factors)
    act = activation or factor_cfg.activation
    fc = FactorModelConfig(
        n_factors=k,
        epochs=factor_cfg.epochs,
        batch_size=factor_cfg.batch_size,
        learning_rate=factor_cfg.learning_rate,
        patience=factor_cfg.patience,
        seed=factor_cfg.seed,
        activation=act,
        beta=factor_cfg.beta,
    )
    _, factors = train_factors(panel.standardized, fc, model_type=model_type, seed=fc.seed)
    prefix = sector[:3]
    F = pd.DataFrame(
        factors.astype(np.float64),
        index=panel.dates,
        columns=[f"{prefix}_f{i + 1}" for i in range(k)],
    )
    F.index.name = "date"
    return SectorFactors(sector=sector, commodities=list(members), n_factors=k, factors=F)


def spanning_analysis(
    name: str,
    F: pd.DataFrame,
    M: pd.DataFrame,
    *,
    n_commodities: int,
    bloc_map: dict[str, list[str]],
    ridge_grid: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 1.0),
    n_folds: int = 5,
    embargo: int = 10,
    n_perm: int = 500,
    spanned_rho_min: float = 0.3,
    spanned_p_max: float = 0.05,
    seed: int = 0,
) -> SectorSpanning:
    """Run the OOS CCA spanning probe of a factor block against macro."""
    Fa, Ma = align_xy(F, M)
    r = min(Fa.shape[1], Ma.shape[1])
    ridge, _ = select_ridge_oos(Fa, Ma, ridges=ridge_grid, n_folds=n_folds, embargo=embargo)
    rho_is, _A, _B, _U, _V, _sf, sm = linear_cca_full(Fa, Ma, ridge)
    rho_oos = purged_cv_canon(Fa, Ma, n_folds=n_folds, embargo=embargo, ridge=ridge)
    pd_null = perdim_perm_null_oos(Fa, Ma, n_perm, seed + 1, r, n_folds, embargo, ridge)
    perm_p = (1 + np.sum(pd_null >= rho_oos[None, :], axis=0)) / (n_perm + 1)
    n_spanned = int(np.sum((rho_oos > spanned_rho_min) & (perm_p < spanned_p_max)))

    bloc = bloc_reduce(Ma, bloc_map)
    _rb, _Ab, _Bb, _Ub, _Vb, _sfb, smb = linear_cca_full(Fa, bloc, ridge)
    rb = smb.shape[1]
    bloc_struct = pd.DataFrame(
        smb, index=list(bloc.columns), columns=[f"dim{k + 1}" for k in range(rb)]
    )

    macro_cols = list(Ma.columns)
    top_macro: list[str] = []
    top_bloc: list[str] = []
    for k in range(r):
        order = np.argsort(-np.abs(sm[:, k]))[:4]
        top_macro.append("; ".join(f"{macro_cols[j]}({sm[j, k]:+.2f})" for j in order))
        if k < rb:
            border = np.argsort(-np.abs(smb[:, k]))[:3]
            top_bloc.append("; ".join(f"{bloc.columns[j]}({smb[j, k]:+.2f})" for j in border))
        else:
            top_bloc.append("")

    return SectorSpanning(
        name=name,
        n_commodities=n_commodities,
        n_factors=Fa.shape[1],
        n_macro=Ma.shape[1],
        ridge=ridge,
        rho_insample=rho_is,
        rho_oos=rho_oos,
        perm_p_oos=perm_p,
        n_spanned=n_spanned,
        struct_macro=sm,
        macro_cols=macro_cols,
        bloc_struct=bloc_struct,
        top_macro=top_macro,
        top_bloc=top_bloc,
    )


def run_sector_spanning(
    macro: pd.DataFrame,
    factor_cfg: FactorModelConfig,
    bloc_map: dict[str, list[str]],
    *,
    sectors: dict[str, list[str]] | None = None,
    n_factors: int = DEFAULT_SECTOR_FACTORS,
    ridge_grid: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 1.0),
    n_folds: int = 5,
    embargo: int = 10,
    n_perm: int = 500,
    spanned_rho_min: float = 0.3,
    spanned_p_max: float = 0.05,
    seed: int = 0,
    include_overall: bool = True,
    model_type: FactorModelType = "vanilla",
) -> dict[str, SectorSpanning]:
    """Extract factors and run the spanning probe for every sector (+ overall)."""
    sectors = sectors or SECTORS
    out: dict[str, SectorSpanning] = {}

    if include_overall:
        overall = load_return_panel()
        _, fac = train_factors(
            overall.standardized,
            factor_cfg,
            model_type=model_type,
            seed=factor_cfg.seed,
        )
        F_all = pd.DataFrame(
            fac.astype(np.float64),
            index=overall.dates,
            columns=[f"f{i + 1}" for i in range(factor_cfg.n_factors)],
        )
        F_all.index.name = "date"
        out["overall"] = spanning_analysis(
            "overall",
            F_all,
            macro,
            n_commodities=overall.n_assets,
            bloc_map=bloc_map,
            ridge_grid=ridge_grid,
            n_folds=n_folds,
            embargo=embargo,
            n_perm=n_perm,
            spanned_rho_min=spanned_rho_min,
            spanned_p_max=spanned_p_max,
            seed=seed,
        )

    for sector, members in sectors.items():
        sf = extract_sector_factors(
            sector, members, factor_cfg, n_factors=n_factors, model_type=model_type
        )
        out[sector] = spanning_analysis(
            sector,
            sf.factors,
            macro,
            n_commodities=len(members),
            bloc_map=bloc_map,
            ridge_grid=ridge_grid,
            n_folds=n_folds,
            embargo=embargo,
            n_perm=n_perm,
            spanned_rho_min=spanned_rho_min,
            spanned_p_max=spanned_p_max,
            seed=seed,
        )
    return out


def spanning_summary_table(results: dict[str, SectorSpanning]) -> pd.DataFrame:
    """Flat one-row-per-(block, dim) table of the spanning results."""
    rows: list[dict] = []
    for name, r in results.items():
        for k in range(len(r.rho_oos)):
            rows.append(
                {
                    "block": name,
                    "n_commodities": r.n_commodities,
                    "n_factors": r.n_factors,
                    "dim": f"dim{k + 1}",
                    "rho_insample": float(r.rho_insample[k]),
                    "rho_oos": float(r.rho_oos[k]),
                    "perm_p_oos": float(r.perm_p_oos[k]),
                    "spanned": bool(
                        (r.rho_oos[k] > 0.3) and (r.perm_p_oos[k] < 0.05)
                    ),
                    "top_macro": r.top_macro[k] if k < len(r.top_macro) else "",
                    "top_bloc": r.top_bloc[k] if k < len(r.top_bloc) else "",
                    "ridge": r.ridge,
                    "n_spanned_block": r.n_spanned,
                }
            )
    return pd.DataFrame(rows)
