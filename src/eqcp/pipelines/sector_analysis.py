"""Sector-wise CCA spanning pipeline.

For each commodity sector (energy / agriculture / metals) and the overall panel,
extract AE latent factors and test — with purged-CV out-of-sample canonical
correlations vs a circular-shift permutation null — whether and how those factors
are spanned by the macro panel. Writes CSV summaries, figures, and a report.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from eqcp.config import (
    load_factor_model_config,
    load_macro_mapping_config,
)
from eqcp.factors.extract import SECTOR_SUBDIRS, FactorModelType
from eqcp.io.macro import load_macro_stationary
from eqcp.reporting.sector_figures import make_sector_figures
from eqcp.reporting.sector_report import write_sector_report
from eqcp.sectors import run_sector_spanning, spanning_summary_table

logger = logging.getLogger(__name__)


def run_sector_analysis(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    map_cfg = load_macro_mapping_config(args.mapping_config)
    factor_cfg = load_factor_model_config(args.factor_config)
    model_type: FactorModelType = getattr(args, "factor_model", "vanilla")

    out = Path(args.outdir)
    res = out / "results" / SECTOR_SUBDIRS[model_type]
    figs = out / "figures" / SECTOR_SUBDIRS[model_type]
    rep = out / "reports"
    for d in (res, figs, rep):
        d.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    macro = load_macro_stationary(args.macro, manifest)
    logger.info(
        "sector analysis (%s): macro %d series; seed=%d",
        model_type,
        macro.shape[1],
        args.seed,
    )

    results = run_sector_spanning(
        macro,
        factor_cfg,
        bloc_map=map_cfg.bloc_map,
        n_factors=args.n_factors,
        ridge_grid=map_cfg.ridge_grid,
        n_folds=map_cfg.n_folds,
        embargo=map_cfg.embargo,
        n_perm=args.n_perm,
        seed=args.seed,
        include_overall=True,
        model_type=model_type,
    )
    blocks = list(results.keys())

    summary = spanning_summary_table(results)
    summary.to_csv(res / "sector_cca_summary.csv", index=False)

    loadings = []
    for name, r in results.items():
        df = pd.DataFrame(
            r.struct_macro,
            index=r.macro_cols,
            columns=[f"dim{k + 1}" for k in range(r.struct_macro.shape[1])],
        )
        df.insert(0, "block", name)
        df.index.name = "macro"
        loadings.append(df.reset_index())
    pd.concat(loadings, ignore_index=True).to_csv(res / "sector_macro_loadings.csv", index=False)

    bloc_rows = []
    for name, r in results.items():
        bs = r.bloc_struct.copy()
        bs.insert(0, "block", name)
        bs.index.name = "bloc"
        bloc_rows.append(bs.reset_index())
    pd.concat(bloc_rows, ignore_index=True).to_csv(res / "sector_bloc_loadings.csv", index=False)

    make_sector_figures(figs, summary, blocks)
    write_sector_report(rep, results, seed=args.seed, n_perm=args.n_perm, model_type=model_type)

    logger.info("sector analysis: wrote outputs under %s, %s, %s", res, figs, rep)
    for name, r in results.items():
        logger.info(
            "  %-11s: %d commodities -> %d factors, %d macro-spanned (rho_oos=%s)",
            name,
            r.n_commodities,
            r.n_factors,
            r.n_spanned,
            [round(float(x), 3) for x in r.rho_oos],
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sector-wise CCA spanning analysis")
    p.add_argument("--macro", default="data/processed/macro_stationary.csv")
    p.add_argument("--manifest", default="data/processed/transform_manifest.csv")
    p.add_argument("--mapping-config", default=None)
    p.add_argument("--factor-config", default=None)
    p.add_argument("--outdir", default=".")
    p.add_argument("--n-factors", type=int, default=3, dest="n_factors")
    p.add_argument("--n-perm", type=int, default=500, dest="n_perm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--factor-model",
        choices=("vanilla", "beta_vae"),
        default="vanilla",
        help="Factor extraction backend (vanilla AE or beta-VAE)",
    )
    return p


def main() -> None:
    run_sector_analysis(build_parser().parse_args())


if __name__ == "__main__":
    main()
