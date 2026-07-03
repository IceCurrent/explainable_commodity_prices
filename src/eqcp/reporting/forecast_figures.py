"""Forecast-PBSV pipeline figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE = "#4C72B0"  # spanned block (fixed categorical order, matches repo figures)
GREEN = "#55A868"  # weakly-macro-correlated block
GRAY = "#8C8C8C"  # placebo / null band, always neutral


def make_forecast_figures(
    figs: Path,
    shapley: pd.DataFrame,
    placebo: pd.DataFrame,
    grouped: pd.DataFrame,
    by_year: pd.DataFrame,
    per_commodity: pd.DataFrame,
    substitution: pd.DataFrame,
    headline_horizon: int,
    transmission: pd.DataFrame | None = None,
) -> None:
    """Write PBSV bars+bands, grouped-by-horizon, per-year, heatmap, substitution PNGs."""
    sh = shapley[shapley["horizon"] == headline_horizon].reset_index(drop=True)
    k = len(sh)
    x = np.arange(k)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [BLUE if s else GREEN for s in sh["spanned"]]
    ax.bar(x, sh["phi_std"], 0.55, color=colors)
    ax.errorbar(
        x,
        sh["phi_std"],
        yerr=[sh["phi_std"] - sh["boot_lo"], sh["boot_hi"] - sh["phi_std"]],
        fmt="none",
        ecolor="black",
        elinewidth=1,
        capsize=3,
        label="bootstrap 95% CI (conditional)",
    )
    ax.fill_between(
        np.arange(-0.5, k + 0.5),
        np.interp(
            np.arange(-0.5, k + 0.5), x, placebo["placebo_lo"], left=np.nan, right=np.nan
        ),
        np.interp(
            np.arange(-0.5, k + 0.5), x, placebo["placebo_hi"], left=np.nan, right=np.nan
        ),
        color=GRAY,
        alpha=0.25,
        label="placebo band (zero-signal, matched cardinality)",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{d}\n({'spanned' if s else 'weak'})" for d, s in zip(sh["dim"], sh["spanned"])],
        fontsize=8,
    )
    ax.set_ylabel("phi (standardized MSE gain units)")
    ax.set_title(f"PBSV per canonical direction, h={headline_horizon}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(figs / "pbsv_headline.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    hx = np.arange(len(grouped))
    w = 0.38
    ax.bar(hx - w / 2, grouped["phi_spanned"], w, color=BLUE, label="spanned block")
    ax.bar(hx + w / 2, grouped["phi_weak"], w, color=GREEN, label="weakly-correlated block")
    for i, row in grouped.reset_index(drop=True).iterrows():
        ax.plot(
            [i - w / 2, i - w / 2],
            [row["boot_lo_spanned"], row["boot_hi_spanned"]],
            color="black",
            lw=1,
        )
        ax.plot(
            [i + w / 2, i + w / 2], [row["boot_lo_weak"], row["boot_hi_weak"]], color="black", lw=1
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(hx)
    ax.set_xticklabels([f"h={int(h)}" for h in grouped["horizon"]])
    ax.set_ylabel("grouped phi (standardized units)")
    ax.set_title("Grouped PBSV by horizon (blocks, rotation-invariant)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(figs / "pbsv_grouped_by_horizon.png", dpi=130)
    plt.close(fig)

    yr = by_year[by_year["year"] > 0]
    phi_cols = [c for c in yr.columns if c.startswith("phi_cv")]
    fig, ax = plt.subplots(figsize=(7, 4))
    yx = np.arange(len(yr))
    bw = 0.8 / max(len(phi_cols), 1)
    shades = plt.get_cmap("Blues")(np.linspace(0.85, 0.35, len(phi_cols)))
    for j, c in enumerate(phi_cols):
        ax.bar(yx + (j - len(phi_cols) / 2) * bw, yr[c], bw, color=shades[j], label=c[4:])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(yx)
    ax.set_xticklabels(yr["year"].astype(int), fontsize=8)
    ax.set_ylabel("phi (standardized units)")
    ax.set_title("PBSV by target year (headline horizon)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(figs / "pbsv_by_year.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 8))
    vmax = float(np.nanmax(np.abs(per_commodity.to_numpy()))) or 1.0
    im = ax.imshow(per_commodity.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(per_commodity.index)))
    ax.set_yticklabels(per_commodity.index, fontsize=6)
    ax.set_xticks(range(per_commodity.shape[1]))
    ax.set_xticklabels(per_commodity.columns, fontsize=8)
    ax.set_title("Per-commodity PBSV (raw squared-error units)")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(figs / "pbsv_per_commodity.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sx = np.arange(len(substitution))
    ax.barh(sx, substitution["v_full_std"], color=BLUE)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(sx)
    ax.set_yticklabels(substitution["arm"], fontsize=8)
    ax.set_xlabel("v(full) with substituted directions (standardized units)")
    ax.set_title("Macro-substitution ladder (common macro calendar)")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(figs / "substitution_ladder.png", dpi=130)
    plt.close(fig)

    if transmission is not None and len(transmission):
        tr = transmission.sort_values("horizon").reset_index(drop=True)
        tx = np.arange(len(tr))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
        # Panel 1: forecast skill of the factor state vs AR(1), by horizon.
        colors = [BLUE if g else GRAY for g in tr["gate_passed"]]
        ax1.bar(tx, tr["r2_oos_vs_ar1"], 0.55, color=colors)
        ax1.axhline(0, color="black", lw=0.8)
        for i, row in tr.iterrows():
            ax1.annotate(
                f"CW p={row['cw_p_overlap']:.2f}\n(n.o. {row['cw_p_nonoverlap']:.2f})",
                (i, row["r2_oos_vs_ar1"]),
                textcoords="offset points",
                xytext=(0, 6 if row["r2_oos_vs_ar1"] >= 0 else -18),
                ha="center",
                fontsize=7,
            )
        ax1.set_ylabel("OOS $R^2$ (full vs AR(1))")
        ax1.set_title("Does the (macro-linked) factor state forecast commodities, by horizon?")
        ax1.grid(axis="y", alpha=0.25, lw=0.5)
        # Panel 2: macro-transmissible share of any gain (substitution retained share).
        # Retained shares are ratios of (near-zero) v(full) when the gate fails, so they can
        # explode; clip the view to a readable window and mark bars that run off-scale.
        lo, hi = -1.5, 2.0
        for off, col, color, lab in (
            (-0.2, "retained_share_spanned", BLUE, "spanned block"),
            (0.2, "retained_share_all", GREEN, "all directions"),
        ):
            vals = tr[col].to_numpy(float)
            ax2.bar(tx + off, np.clip(vals, lo, hi), 0.38, color=color, label=lab)
            for i, v in enumerate(vals):
                if v < lo or v > hi:
                    ax2.annotate(
                        f"{v:.1f}",
                        (i + off, hi if v > hi else lo),
                        textcoords="offset points",
                        xytext=(0, 4 if v > hi else -10),
                        ha="center",
                        fontsize=6,
                        color=color,
                    )
        ax2.set_ylim(lo, hi)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_ylabel("macro-retained share of v(full)")
        ax2.set_xticks(tx)
        ax2.set_xticklabels(
            [f"h={int(h)}\n(eff n={int(n)})" for h, n in zip(tr["horizon"], tr["effective_n"])]
        )
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.25, lw=0.5)
        ax2.annotate(
            "shares interpretable only where the gate passes (blue in top panel)",
            (0.5, 0.02),
            xycoords="axes fraction",
            ha="center",
            fontsize=7,
            style="italic",
            color=GRAY,
        )
        fig.tight_layout()
        fig.savefig(figs / "transmission_by_horizon.png", dpi=130)
        plt.close(fig)
