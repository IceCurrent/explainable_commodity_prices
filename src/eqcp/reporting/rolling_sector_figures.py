"""Figures for the rolling-window sector decomposition."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_BLOCK_ORDER = ["overall", "energy", "agriculture", "metals"]
_BLOCK_COLOR = {
    "overall": "#2A6F97",
    "energy": "#DD8452",
    "agriculture": "#55A868",
    "metals": "#8172B3",
}


def make_rolling_sector_figures(
    figs: Path, summary: pd.DataFrame, stability: pd.DataFrame, pooled: pd.DataFrame
) -> None:
    figs.mkdir(parents=True, exist_ok=True)
    _leading_axis_by_block(figs, summary)
    _oos_rho_by_block(figs, stability)
    _forecast_by_block(figs, pooled)


def _ordered(blocks: list[str]) -> list[str]:
    present = [b for b in _BLOCK_ORDER if b in blocks]
    present += [b for b in blocks if b not in present]
    return present


def _leading_axis_by_block(figs: Path, summary: pd.DataFrame) -> None:
    """Leading-rank cross-window macro cosine + its class, one bar per block."""
    blocks = _ordered(summary["block"].tolist())
    s = summary.set_index("block").reindex(blocks)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(blocks))
    colors = [_BLOCK_COLOR.get(b, "#8C8C8C") for b in blocks]
    ax.bar(x, s["lead_median_abs_cos"], color=colors, edgecolor="black", lw=0.6)
    ax.axhline(0.7, color="crimson", ls="--", lw=1, label="stability bar (0.7)")
    for xi, b in zip(x, blocks):
        ax.annotate(
            f"{s.loc[b, 'lead_rank_class']}\nOOS ρ={s.loc[b, 'lead_median_rho_oos']:.2f}",
            (float(xi), float(s.loc[b, "lead_median_abs_cos"])),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("leading-rank cross-window median |cosine|")
    ax.set_title("Does each sector keep a stable macro axis under rolling?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "sector_leading_axis.png", dpi=130)
    plt.close(fig)


def _oos_rho_by_block(figs: Path, stability: pd.DataFrame) -> None:
    """Median OOS |rho| per rank, grouped by block."""
    blocks = _ordered(stability["block"].unique().tolist())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    max_ranks = int(stability.groupby("block")["rank"].count().max())
    width = 0.8 / max(len(blocks), 1)
    x = np.arange(max_ranks)
    for i, b in enumerate(blocks):
        d = stability[stability["block"] == b].reset_index(drop=True)
        vals = d["median_rho_oos"].to_numpy()
        padded = np.full(max_ranks, np.nan)
        padded[: len(vals)] = vals
        ax.bar(x + i * width, padded, width, label=b, color=_BLOCK_COLOR.get(b, "#8C8C8C"))
    ax.axhline(0.3, color="crimson", ls="--", lw=1, label=r"$\rho=0.3$")
    ax.set_xticks(x + width * (len(blocks) - 1) / 2)
    ax.set_xticklabels([f"V{k + 1}" for k in range(max_ranks)])
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"median OOS $|\rho|$ (frozen-in-window, test block)")
    ax.set_title("Out-of-sample macro correlation per rank, by sector")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "sector_oos_rho_by_rank.png", dpi=130)
    plt.close(fig)


def _forecast_by_block(figs: Path, pooled: pd.DataFrame) -> None:
    """Pooled OOS R2 vs AR(1) by horizon, grouped by block (the sector null check)."""
    blocks = _ordered(pooled["block"].unique().tolist())
    horizons = sorted(pooled["horizon"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / max(len(blocks), 1)
    x = np.arange(len(horizons))
    for i, b in enumerate(blocks):
        d = pooled[pooled["block"] == b].set_index("horizon").reindex(horizons)
        ax.bar(x + i * width, d["r2_vs_ar1"], width, label=b, color=_BLOCK_COLOR.get(b, "#8C8C8C"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + width * (len(blocks) - 1) / 2)
    ax.set_xticklabels([f"h={int(h)}" for h in horizons])
    ax.set_ylabel(r"pooled OOS $R^2$ vs AR(1)")
    ax.set_title("Rolling factor-augmented AR vs AR(1), by sector and horizon")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "sector_forecast_r2.png", dpi=130)
    plt.close(fig)
