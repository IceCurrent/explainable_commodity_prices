"""Macro-mapping pipeline figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def make_figures(
    figs: Path,
    rho_is: np.ndarray,
    rho_oos: np.ndarray,
    pd_null: np.ndarray,
    struct_m: np.ndarray,
    macro_cols: list[str],
    regime_df: pd.DataFrame,
    stab: pd.DataFrame,
    ll_df: pd.DataFrame,
) -> None:
    """Write canon-vs-null, loadings, regime, KCCA stability, and lead/lag PNGs."""
    r = len(rho_is)
    dims = np.arange(1, r + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    ax.bar(dims - w / 2, rho_is, w, label="in-sample", color="#4C72B0")
    ax.bar(dims + w / 2, rho_oos, w, label="OOS (purged CV)", color="#55A868")
    p95 = np.percentile(pd_null, 95, axis=0)
    ax.plot(dims, p95, "k--", marker="o", label="perm-null p95 (OOS)")
    ax.set_xlabel("canonical dimension")
    ax.set_ylabel("canonical correlation")
    ax.set_title("AE factors vs macro: canonical correlations vs permutation null")
    ax.set_xticks(dims)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "canon_vs_null.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 10))
    im = ax.imshow(struct_m, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(macro_cols)))
    ax.set_yticklabels(macro_cols, fontsize=6)
    ax.set_xticks(range(r))
    ax.set_xticklabels([f"dim{k + 1}" for k in range(r)])
    ax.set_title("Macro structure correlations")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(figs / "loadings_macro_heatmap.png", dpi=130)
    plt.close(fig)

    rdf = regime_df.set_index("regime")[["rho_min", "rho_mean"]]
    fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(rdf.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels(rdf.index)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rho_min", "rho_mean"])
    ax.set_title("Per-regime canonical correlation (bloc-PC)")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(figs / "per_regime_heatmap.png", dpi=130)
    plt.close(fig)

    piv = stab.pivot(index="reg", columns="gamma_scale", values="kcca_min")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("gamma scale")
    ax.set_ylabel("reg")
    ax.set_title("KCCA min canonical corr (stability)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(
                j,
                i,
                f"{piv.to_numpy()[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(figs / "kcca_stability.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ll_df["lag"], ll_df["rho_mean"], marker="o")
    ax.axvline(0, color="grey", ls=":")
    ax.set_xlabel("lag l  (F_t vs M_{t-l};  l>0 = macro leads)")
    ax.set_ylabel("mean canonical correlation")
    ax.set_title("Lead/lag scan")
    fig.tight_layout()
    fig.savefig(figs / "leadlag.png", dpi=130)
    plt.close(fig)
