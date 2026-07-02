"""Sector-analysis figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SPANNED = "#4C72B0"
WEAK = "#C44E52"
NEUTRAL = "#8C8C8C"


def make_sector_figures(figs: Path, summary: pd.DataFrame, blocks: list[str]) -> None:
    """OOS canonical-correlation bars per block and a spanned-count summary."""
    figs.mkdir(parents=True, exist_ok=True)

    ncols = len(blocks)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, block in zip(axes, blocks):
        sub = summary[summary["block"] == block].reset_index(drop=True)
        x = np.arange(len(sub))
        colors = [SPANNED if s else WEAK for s in sub["spanned"]]
        ax.bar(x, sub["rho_oos"], 0.6, color=colors)
        ax.axhline(0.3, color=NEUTRAL, ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["dim"], fontsize=8)
        ax.set_ylim(0, 1.0)
        n_c = int(sub["n_commodities"].iloc[0]) if len(sub) else 0
        ax.set_title(f"{block}\n({n_c} commodities)", fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0].set_ylabel("OOS canonical correlation (purged CV)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SPANNED),
        plt.Rectangle((0, 0), 1, 1, color=WEAK),
    ]
    fig.legend(
        handles,
        ["macro-spanned (rho>0.3, p<0.05)", "weakly macro-correlated"],
        loc="lower center",
        ncol=2,
        fontsize=8,
    )
    fig.suptitle("Macro spanning of commodity factor directions, by sector", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(figs / "sector_spanning_bars.png", dpi=130)
    plt.close(fig)

    counts = (
        summary.groupby("block")["spanned"]
        .sum()
        .reindex(blocks)
        .fillna(0)
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.to_numpy(), color=SPANNED)
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v + 0.05, str(int(v)), ha="center", fontsize=10)
    ax.set_ylabel("# macro-spanned directions (OOS)")
    ax.set_title("Number of macro-spanned factor directions per sector")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(figs / "sector_spanned_counts.png", dpi=130)
    plt.close(fig)
