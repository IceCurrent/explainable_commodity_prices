"""Figures for the rolling-window forecast + rolling-explainability pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_rolling_figures(
    figs: Path,
    windows: pd.DataFrame,
    stability: pd.DataFrame,
    fingerprint: pd.DataFrame,
    pooled: pd.DataFrame,
    shapley: pd.DataFrame,
    n_dims: int,
) -> None:
    figs.mkdir(parents=True, exist_ok=True)
    _rho_by_window(figs, windows, n_dims)
    _rank_stability(figs, stability)
    _fingerprint_heatmap(figs, fingerprint)
    _forecast_r2(figs, pooled)
    _pbsv_by_horizon(figs, shapley)


_CLASS_COLOR = {
    "stable": "#2b8cbe",
    "partial": "#f4a261",
    "weak": "#bdbdbd",
    "rotating": "#bdbdbd",
}


def _rho_by_window(figs: Path, windows: pd.DataFrame, n_dims: int) -> None:
    x = pd.to_datetime(windows["test_start"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k in range(n_dims):
        col = f"rho{k + 1}"
        if col in windows:
            ax.plot(x, windows[col], marker="o", ms=2.5, lw=1.1, label=f"V{k + 1}")
    ax.set_title("Train canonical correlation per rolling window (factor \u2194 macro)")
    ax.set_xlabel("test-block start")
    ax.set_ylabel(r"$\rho_k$ (train)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(ncol=n_dims, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "rho_by_window.png", dpi=130)
    plt.close(fig)


def _rank_stability(figs: Path, stability: pd.DataFrame) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    ranks = stability["rank"].tolist()
    med = stability["median_abs_cos"].to_numpy()
    lo = med - stability["q10_abs_cos"].to_numpy()
    hi = stability["q90_abs_cos"].to_numpy() - med
    classes = stability.get("stability_class", pd.Series(["stable"] * len(ranks)))
    colors = [_CLASS_COLOR.get(c, "#bdbdbd") for c in classes]
    x = np.arange(len(ranks))
    ax.bar(x, med, yerr=[lo, hi], color=colors, capsize=4, edgecolor="black", lw=0.6)

    # Per-rank shuffle-null p95 markers + theoretical chance level.
    if "null_p95" in stability:
        ax.scatter(x, stability["null_p95"], marker="_", s=420, color="black", lw=1.6,
                   label="shuffle null p95", zorder=5)
    if "chance_abs_cos" in stability:
        ax.axhline(float(stability["chance_abs_cos"].iloc[0]), color="grey", ls=":", lw=1,
                   label="random |cos|")
    ax.axhline(0.7, color="crimson", ls="--", lw=1, label="stability bar")
    for xi, c in zip(x, classes):
        ax.annotate(c, (xi, 0.02), ha="center", va="bottom", fontsize=7, rotation=90,
                    color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(ranks)
    ax.set_ylim(0, 1)
    ax.set_ylabel("cross-window median |cosine| of macro fingerprint")
    ax.set_title("Does each rank keep its macro identity as the AE rotates?")
    ax.legend(fontsize=8)

    # OOS panel: does the direction still track macro out-of-sample?
    if "median_rho_oos" in stability:
        ax2.bar(x, stability["median_rho_oos"], color=colors, edgecolor="black", lw=0.6)
        ax2.axhline(0.3, color="crimson", ls="--", lw=1, label=r"$\rho=0.3$")
        ax2.set_xticks(x)
        ax2.set_xticklabels(ranks)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel(r"median OOS $|\rho|$ (frozen-in-window, test block)")
        ax2.set_title("Out-of-sample macro correlation per rank")
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "rank_stability.png", dpi=130)
    plt.close(fig)


def _fingerprint_heatmap(figs: Path, fingerprint: pd.DataFrame) -> None:
    # Show the macro series with the strongest average loading on any rank.
    top = fingerprint.abs().max(axis=1).sort_values(ascending=False).head(18).index
    sub = fingerprint.loc[top]
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    im = ax.imshow(sub.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(sub.shape[1]))
    ax.set_xticklabels(sub.columns)
    ax.set_yticks(range(sub.shape[0]))
    ax.set_yticklabels(sub.index, fontsize=8)
    ax.set_title("Average macro fingerprint per rank\n(mean structure corr across windows)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figs / "macro_fingerprint_heatmap.png", dpi=130)
    plt.close(fig)


def _forecast_r2(figs: Path, pooled: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(pooled))
    ax.bar(x, pooled["r2_vs_ar1"], color="#7fbf7b", edgecolor="black", lw=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={int(h)}" for h in pooled["horizon"]])
    ax.set_ylabel(r"pooled OOS $R^2$ vs AR(1)")
    ax.set_title("Rolling factor-augmented AR vs AR(1), pooled across windows")
    for xi, (r2, p) in enumerate(zip(pooled["r2_vs_ar1"], pooled["cw_p"])):
        ax.annotate(f"CW p={p:.2f}", (xi, r2), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "forecast_r2_by_horizon.png", dpi=130)
    plt.close(fig)


def _pbsv_by_horizon(figs: Path, shapley: pd.DataFrame) -> None:
    horizons = sorted(shapley["horizon"].unique())
    ranks = shapley["rank"].unique().tolist()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(horizons), 1)
    x = np.arange(len(ranks))
    for i, h in enumerate(horizons):
        sub = shapley[shapley["horizon"] == h].set_index("rank").reindex(ranks)
        ax.bar(x + i * width, sub["phi"], width, label=f"h={int(h)}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x + width * (len(horizons) - 1) / 2)
    ax.set_xticklabels(ranks)
    ax.set_ylabel(r"rank-pooled Shapley $\phi_k$ (MSE units)")
    ax.set_title("Forecast-based Shapley by canonical rank (pooled over windows)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "pbsv_by_horizon.png", dpi=130)
    plt.close(fig)
