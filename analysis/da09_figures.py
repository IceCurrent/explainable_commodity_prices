"""Deep-analysis 09: figures for the diagnosis report (matplotlib, static PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "deep_analysis"
FIGS = ROOT / "figures" / "deep_analysis"
FIGS.mkdir(parents=True, exist_ok=True)

BLUE = "#2a78d6"
AQUA = "#1baf7a"
RED = "#e34948"
INK = "#333230"
MUTED = "#6f6e66"
GRID = "#e5e4dd"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_power_curve():
    df = pd.read_csv(RES / "da04_power_summary.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    x = df["r2_injected"] * 100
    ax.plot(x, df["detect_rate"], color=BLUE, linewidth=2, marker="o", markersize=6, zorder=3)
    ax.axhline(0.8, color=MUTED, linewidth=1, linestyle="--")
    ax.text(2.02, 0.815, "80% power", fontsize=8, color=MUTED, ha="right")
    ax.axhline(0.05, color=MUTED, linewidth=1, linestyle=":")
    ax.text(2.02, 0.065, "5% size", fontsize=8, color=MUTED, ha="right")
    for _, r in df.iterrows():
        ax.annotate(f"{r['detect_rate']:.0%}", (r["r2_injected"] * 100, r["detect_rate"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, color=INK, ha="center")
    ax.set_xlabel("injected pooled predictive R² (%)", fontsize=10, color=INK)
    ax.set_ylabel("detection rate (pooled CW p < 0.05)", fontsize=10, color=INK)
    ax.set_ylim(-0.04, 1.09)
    ax.set_title("The engine detects any planted signal ≥ 0.25% R² — and none was there",
                 fontsize=11, color=INK, loc="left", pad=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "power_curve.png")
    plt.close(fig)


def fig_embargo():
    grp = pd.read_csv(RES / "da03_embargo.csv")
    same = grp.loc[grp["arm"] == "same_day"].iloc[0]
    emb = grp.loc[grp["arm"] == "embargo_1d"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    bars = ax.bar([0, 1], [same["real_count"], emb["real_count"]], width=0.5,
                  color=BLUE, zorder=3)
    hi = max(same["placebo_p95"], emb["placebo_p95"])
    ax.axhspan(0, hi, color=GRID, zorder=1)
    ax.text(1.42, hi + 3, "placebo 95% range\n(joint circular shifts)", fontsize=8,
            color=MUTED, ha="right")
    for b, v, p in zip(bars, [same["real_count"], emb["real_count"]],
                       [same["p_value"], emb["p_value"]]):
        ax.annotate(f"{int(v)}\n(p={p:.3f})", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 6), fontsize=9, color=INK,
                    ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["macro at t\n(closes AFTER commodity settle)",
                        "macro at t−1\n(24h embargo)"], fontsize=9, color=INK)
    ax.set_ylabel("lag-1 predictive rejections (of 777 pairs)", fontsize=10, color=INK)
    ax.set_ylim(0, 205)
    ax.set_title("All apparent lag-1 macro→commodity signal dies under a 1-day embargo",
                 fontsize=11, color=INK, loc="left", pad=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "embargo.png")
    plt.close(fig)


def fig_fingerprint():
    fp = pd.read_csv(RES / "da03_settle_fingerprint.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)
    x = fp["after_settle_hours"] + np.random.default_rng(0).uniform(-0.06, 0.06, len(fp))
    y = fp["mean_abs_t_after_predictors"]
    ax.scatter(x, y, s=42, color=BLUE, zorder=3)
    b, a = np.polyfit(fp["after_settle_hours"], y, 1)
    xs = np.linspace(2.3, 5.2, 10)
    ax.plot(xs, a + b * xs, color=MUTED, linewidth=1.2, linestyle="--", zorder=2)
    for _, r in fp.iterrows():
        if r["after_settle_hours"] >= 3.4 or r["mean_abs_t_after_predictors"] > 1.2:
            ax.annotate(r["commodity"],
                        (r["after_settle_hours"], r["mean_abs_t_after_predictors"]),
                        textcoords="offset points", xytext=(6, -3), fontsize=7.5, color=INK)
    rho = np.corrcoef(fp["after_settle_hours"], y)[0, 1]
    ax.text(0.02, 0.95, f"corr = {rho:+.2f}", transform=ax.transAxes, fontsize=10,
            color=INK, va="top")
    ax.set_xlabel("hours between commodity settlement and 17:00 ET macro close",
                  fontsize=10, color=INK)
    ax.set_ylabel("mean |t| of lag-1 macro predictors", fontsize=10, color=INK)
    ax.set_title("'Predictability' scales with the after-settle overlap window\n"
                 "(LME metals settle noon ET: 5h of same-day macro news lands in t+1)",
                 fontsize=11, color=INK, loc="left", pad=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "settle_fingerprint.png")
    plt.close(fig)


def fig_vol_vs_mean():
    vol = pd.read_csv(RES / "da06_vol_control.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    xs = np.arange(len(vol))
    w = 0.34
    b1 = ax.bar(xs - w / 2, vol["r2_oos_ar_vs_mean"] * 100, width=w, color=AQUA,
                label="volatility target (|r|): AR-in-vol vs expanding mean", zorder=3)
    b2 = ax.bar(xs + w / 2, vol["r2_oos_full_vs_ar"] * 100, width=w, color=BLUE,
                label="mean-return target: factor state vs AR(1)", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.annotate(f"{v:+.1f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4 if v >= 0 else -12),
                        fontsize=8.5, color=INK, ha="center")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"h = {int(h)}" for h in vol["horizon"]], fontsize=10, color=INK)
    ax.set_ylabel("out-of-sample R² (%)", fontsize=10, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_title("Same data, same engine: vol is forecastable, mean is not",
                 fontsize=11, color=INK, loc="left", pad=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "vol_vs_mean.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_power_curve()
    fig_embargo()
    fig_fingerprint()
    fig_vol_vs_mean()
    print("wrote figures to", FIGS)
