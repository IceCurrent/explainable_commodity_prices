"""Rolling-window sector decomposition.

The overall-panel rolling study (:mod:`eqcp.pipelines.rolling_forecast`) asks
whether the macro *explanation* of the 21-commodity factor space survives when
the AE and the CCA basis are re-fit on every window. This pipeline runs the
**same** walk-forward engine independently on each commodity sector
(energy / agriculture / metals) — plus the overall panel as a reference row — so
the aggregate finding can be decomposed: *which sectors keep a stable,
out-of-sample macro axis under rolling, and which rotate?*

Nothing is fit on the full sample. For every block and every window the AE is
retrained on the trailing ``train_window`` days, the CCA basis is refit on the
train block only, and the factor-augmented AR forecasts the next ``test_block``
days before the window advances by ``step``. Outputs are consolidated
one-row-per-(block, rank) / (block, horizon) tables so the sectors are directly
comparable to the overall panel and to each other.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from eqcp.config import (
    FactorModelConfig,
    RollingForecastConfig,
    load_factor_model_config,
    load_rolling_forecast_config,
)
from eqcp.factors.extract import FactorModelType
from eqcp.forecasting.rolling import RollingResult, rank_stability, rolling_forecasts
from eqcp.io.commodities import SECTORS, load_return_panel
from eqcp.io.macro import load_macro_stationary
from eqcp.pipelines._rolling_common import (
    fingerprint_frame,
    grouping_boundary,
    summarize_forecasts,
    top_drivers,
)
from eqcp.pipelines._runlog import RunLogger
from eqcp.reporting.rolling_sector_figures import make_rolling_sector_figures
from eqcp.reporting.rolling_sector_report import write_rolling_sector_report

logger = logging.getLogger(__name__)

ROLLING_SECTOR_SUBDIRS: dict[FactorModelType, str] = {
    "vanilla": "rolling_sectors",
    "beta_vae": "rolling_sectors_beta",
}

# The overall panel is included as a reference block; sectors follow so the
# consolidated tables read overall -> energy -> agriculture -> metals.
OVERALL_BLOCK = "overall"
SECTOR_MAX_FACTORS = 3  # genuine compression for a 5-9 series sector


def _block_members() -> dict[str, list[str] | None]:
    """Block -> commodity list (``None`` = the full retained panel)."""
    blocks: dict[str, list[str] | None] = {OVERALL_BLOCK: None}
    for sector, members in SECTORS.items():
        blocks[sector] = list(members)
    return blocks


def _block_factor_cfg(
    factor_cfg: FactorModelConfig, block: str, n_members: int
) -> FactorModelConfig:
    """Per-block bottleneck: full K on overall, capped compression per sector."""
    if block == OVERALL_BLOCK:
        return factor_cfg
    k = max(1, min(SECTOR_MAX_FACTORS, n_members - 1))
    return replace(factor_cfg, n_factors=k)


def _run_block(
    block: str,
    members: list[str] | None,
    macro: pd.DataFrame,
    cfg: RollingForecastConfig,
    factor_cfg: FactorModelConfig,
    model_type: FactorModelType,
    seed: int,
    log: RunLogger,
) -> dict:
    """Run the rolling engine on one block and reduce it to comparable rows."""
    panel = load_return_panel(commodities=members) if members else load_return_panel()
    returns = panel.raw
    dates = panel.dates
    n_members = panel.n_assets
    block_cfg = _block_factor_cfg(factor_cfg, block, n_members)
    log(
        f"[{block}] {n_members} commodities -> K={block_cfg.n_factors}; "
        f"rolling {cfg.train_window}d/{cfg.test_block}d step {cfg.step}"
    )

    roll: RollingResult = rolling_forecasts(
        returns, dates, macro, cfg, block_cfg, model_type, seed, log=None
    )
    K = roll.n_dims
    stab = rank_stability(roll, cfg.fingerprint_cos_stable, n_perm=cfg.n_perm_stability, seed=seed)
    fp = fingerprint_frame(roll)
    _, groups = grouping_boundary(roll)
    summ = summarize_forecasts(roll, returns, cfg, stab, groups, seed)

    n_stable = int(stab["macro_stable"].sum())
    n_above_null = int((stab["p_value"] < 0.05).sum())
    lead = stab.iloc[0]
    log(
        f"[{block}] {roll.n_windows} windows; ladder="
        + ", ".join(f"{r['rank']}={r['stability_class']}" for _, r in stab.iterrows())
        + f"; lead {lead['rank']} med|cos|={lead['median_abs_cos']:.2f} "
        f"rho_oos={lead['median_rho_oos']:.2f}"
    )

    stab_out = stab.copy()
    stab_out.insert(0, "block", block)
    stab_out["n_commodities"] = n_members
    stab_out["n_factors"] = K

    pooled_out = summ["pooled"].copy()
    pooled_out.insert(0, "block", block)

    fp_rows = pd.DataFrame(
        {
            "block": block,
            "rank": [f"V{k + 1}" for k in range(K)],
            "top_macro": [top_drivers(fp, f"V{k + 1}") for k in range(K)],
        }
    )

    best_cw = float(summ["pooled"]["cw_p"].min())
    summary_row = {
        "block": block,
        "n_commodities": n_members,
        "n_factors": K,
        "n_windows": roll.n_windows,
        "n_stable_ranks": n_stable,
        "n_ranks_above_null": n_above_null,
        "lead_rank_class": str(lead["stability_class"]),
        "lead_median_abs_cos": float(lead["median_abs_cos"]),
        "lead_median_rho_oos": float(lead["median_rho_oos"]),
        "best_forecast_cw_p": best_cw,
        "any_forecast_beats_ar1": bool((summ["pooled"]["cw_p"] < 0.05).any()),
    }
    return {
        "stability": stab_out,
        "pooled": pooled_out,
        "fingerprint": fp_rows,
        "summary": summary_row,
        "acceptance": summ["acceptance"],
    }


def run_rolling_sectors(args: argparse.Namespace) -> None:
    """Run the rolling engine per sector (+ overall) and write consolidated tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg: RollingForecastConfig = load_rolling_forecast_config(args.config)
    factor_cfg: FactorModelConfig = load_factor_model_config(args.factor_config)
    seed = args.seed
    model_type: FactorModelType = getattr(args, "factor_model", "vanilla")

    out = Path(args.outdir)
    res = out / "results" / ROLLING_SECTOR_SUBDIRS[model_type]
    figs = out / "figures" / ROLLING_SECTOR_SUBDIRS[model_type]
    rep = out / "reports"
    for d in (res, figs, rep):
        d.mkdir(parents=True, exist_ok=True)

    log = RunLogger()
    manifest = pd.read_csv(args.manifest)
    macro = load_macro_stationary(args.macro, manifest)
    log(f"rolling sectors ({model_type}): macro {macro.shape[1]} series; seed={seed}")

    stability_frames: list[pd.DataFrame] = []
    pooled_frames: list[pd.DataFrame] = []
    fingerprint_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    acceptance: list[tuple[str, bool, str]] = []

    for block, members in _block_members().items():
        r = _run_block(block, members, macro, cfg, factor_cfg, model_type, seed, log)
        stability_frames.append(r["stability"])
        pooled_frames.append(r["pooled"])
        fingerprint_frames.append(r["fingerprint"])
        summary_rows.append(r["summary"])
        acceptance.extend(r["acceptance"])

    stability = pd.concat(stability_frames, ignore_index=True)
    pooled = pd.concat(pooled_frames, ignore_index=True)
    fingerprint = pd.concat(fingerprint_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    stability.to_csv(res / "sector_rank_stability.csv", index=False)
    pooled.to_csv(res / "sector_forecast_pooled.csv", index=False)
    fingerprint.to_csv(res / "sector_macro_fingerprint.csv", index=False)
    summary.to_csv(res / "sector_summary.csv", index=False)

    make_rolling_sector_figures(figs, summary=summary, stability=stability, pooled=pooled)
    write_rolling_sector_report(
        rep,
        cfg=cfg,
        seed=seed,
        model_type=model_type,
        summary=summary,
        stability=stability,
        pooled=pooled,
        fingerprint=fingerprint,
    )

    log.write(res / "run_log.txt")
    print("\n=== SECTOR SUMMARY (rolling) ===")
    for _, row in summary.iterrows():
        print(
            f"  {row['block']:<12} {row['n_commodities']:>2} commodities, K={row['n_factors']}: "
            f"lead rank {row['lead_rank_class']:<8} "
            f"(med|cos|={row['lead_median_abs_cos']:.2f}, "
            f"OOS rho={row['lead_median_rho_oos']:.2f}); "
            f"best forecast CW p={row['best_forecast_cw_p']:.3f}"
        )
    print(f"\nDone. Outputs under {res} , {figs} , {rep}")


def build_parser() -> argparse.ArgumentParser:
    from eqcp.config import PROJECT_ROOT

    dp = PROJECT_ROOT / "data" / "processed"
    cfg_dir = PROJECT_ROOT / "configs"
    p = argparse.ArgumentParser(description="Rolling-window sector decomposition")
    p.add_argument("--macro", type=Path, default=dp / "macro_stationary.csv")
    p.add_argument("--manifest", type=Path, default=dp / "transform_manifest.csv")
    p.add_argument("--outdir", type=Path, default=PROJECT_ROOT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=Path, default=cfg_dir / "rolling_forecast.yaml")
    p.add_argument("--factor-config", type=Path, default=cfg_dir / "factor_model.yaml")
    p.add_argument(
        "--factor-model",
        choices=("vanilla", "beta_vae"),
        default="vanilla",
        help="Factor extraction backend (vanilla AE or beta-VAE)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    run_rolling_sectors(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
