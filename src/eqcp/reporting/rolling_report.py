"""Markdown report for the rolling-window forecast + explainability experiment."""

from __future__ import annotations

import numbers
from pathlib import Path

import pandas as pd

from eqcp.config import RollingForecastConfig
from eqcp.factors.extract import FactorModelType

_INT_COLS = {"horizon", "n_origins", "n_nonoverlap", "n_train", "n_test", "n_spanned", "window"}


def _md_table(df: pd.DataFrame, floatfmt: str = "{:+.5f}") -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, bool):
                cells.append(str(v))
            elif c in _INT_COLS and isinstance(v, numbers.Real):
                cells.append(str(int(float(v))))
            elif isinstance(v, numbers.Integral):
                cells.append(str(int(v)))
            elif isinstance(v, numbers.Real):
                cells.append(floatfmt.format(float(v)))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


def write_rolling_report(
    rep: Path,
    cfg: RollingForecastConfig,
    seed: int,
    model_type: FactorModelType,
    n_windows: int,
    n_windows_total: int,
    span: tuple[str, str],
    stability: pd.DataFrame,
    fingerprint: pd.DataFrame,
    pooled: pd.DataFrame,
    shapley: pd.DataFrame,
    grouped: pd.DataFrame,
    median_boundary: int,
    forecast_gate: bool,
    explain_survives: bool,
    acceptance: list[tuple[str, bool, str]],
) -> None:
    suffix = "" if model_type == "vanilla" else "_beta"
    path = rep / f"rolling_forecast{suffix}_report.md"
    K = int(fingerprint.shape[1])
    n_stable = int(stability["macro_stable"].sum())
    n_above_null = int((stability["p_value"] < 0.05).sum())
    chance = float(stability["chance_abs_cos"].iloc[0])
    class_counts = stability["stability_class"].value_counts().to_dict()
    ladder = ", ".join(f"{v}\u00d7 {kk}" for kk, v in class_counts.items())

    if explain_survives and forecast_gate:
        verdict = (
            "**Rolling + explainability coexist, with forecast value.** The leading "
            "canonical rank stays macro-anchored across windows AND the pooled forecast "
            "gate passes."
        )
    elif explain_survives:
        verdict = (
            "**Explainability survives rolling; forecast value does not.** Even as the AE "
            "latent axes rotate window to window, the leading canonical rank keeps the same "
            "macro fingerprint far above the label-shuffle null (see \u00a71), and still tracks "
            "macro out-of-sample, so rank-level macro attribution remains meaningful under "
            "rolling. The pooled forecast gate still fails \u2014 the EMH-coherent null carries "
            "over from the frozen-basis study."
        )
    else:
        verdict = (
            "**Rolling breaks explainability.** No canonical rank keeps a stable macro "
            "fingerprint across windows \u2014 the AE re-fit rotates the latent space enough that "
            "rank identity (and therefore rank-level attribution) is not transportable across "
            "windows. This is the disclosed limitation: with per-window re-fitting, a single "
            "global 'macro dimension' cannot be named. Use the frozen-basis branch for "
            "attribution claims."
        )

    lines: list[str] = []
    a = lines.append
    a("# Rolling-Window Forecast + Rolling-Explainability Report")
    a("")
    a("## Verdict")
    a("")
    a(verdict)
    a("")
    a(
        f"Setup: walk-forward roll, **{cfg.train_window}d** train / **{cfg.test_block}d** test, "
        f"step {cfg.step}; model **{model_type}**, seed {seed}; sample {span[0]} \u2192 {span[1]}; "
        f"**{n_windows}/{n_windows_total}** windows usable. Everything re-fits per window: the AE "
        f"is retrained (budget {cfg.ae_epochs} epochs), the CCA basis is refit, ranks are sorted "
        f"by train canonical correlation and sign-anchored on their dominant macro loading so "
        f"they are comparable across windows. Stability ladder: **{ladder}**; "
        f"**{n_stable}/{K}** ranks clear the stability bar ({cfg.fingerprint_cos_stable}) and "
        f"**{n_above_null}/{K}** beat the label-shuffle null (random |cos| \u2248 {chance:.2f})."
    )
    a("")

    a("## 1. Does explainability survive rolling?")
    a("")
    a(
        "The AE is retrained every window, so its latent coordinates rotate freely (gauge + "
        "optimizer multiplicity + regime drift). The macro side, however, lives in the same "
        "fixed macro-series space every window, so we judge rank identity by the cross-window "
        "cosine of each rank's macro structure-correlation vector. Two controls make this "
        "rigorous: a **label-shuffle null** (independently permuting each window's macro labels "
        f"destroys cross-window identity; random |cos| \u2248 {chance:.2f}) with a one-sided "
        "`p_value`, and an **out-of-sample check** (`median_rho_oos` = the frozen-in-window "
        "canonical correlation measured on the held-out test block). A rank earns 'stable' only "
        "if it clears the bar AND beats the shuffle null; 'partial'/'weak' are above-null but "
        "below the bar; 'rotating' is indistinguishable from the null."
    )
    a("")
    a(_md_table(
        stability[["rank", "median_abs_cos", "null_p95", "chance_abs_cos", "p_value",
                   "median_rho_oos", "stability_class"]],
        floatfmt="{:.3f}"))
    a("")
    a("See `figures/rolling_forecast" + suffix + "/rank_stability.png` and `rho_by_window.png`.")
    a("")

    a("## 2. Average macro fingerprint per rank")
    a("")
    a("Mean (sign-anchored) macro structure correlation across all windows; top drivers per rank:")
    a("")
    top_rows = []
    for k in range(K):
        col = f"V{k + 1}"
        s = fingerprint[col].abs().sort_values(ascending=False).head(4)
        names = "; ".join(f"{n}({fingerprint.loc[n, col]:+.2f})" for n in s.index)
        top_rows.append({"rank": col, "top_macro (mean across windows)": names})
    a(_md_table(pd.DataFrame(top_rows)))
    a("")

    a("## 3. Pooled forecast accuracy (across all rolling windows)")
    a("")
    a(_md_table(pooled[["horizon", "n_origins", "n_nonoverlap", "r2_vs_ar1", "r2_vs_zero",
                        "cw_p", "cw_nonoverlap_p", "v_full", "beats_zero"]]))
    a("")
    a(
        "`r2_vs_ar1` is the pooled standardized OOS R\u00b2 of the full factor-augmented AR vs "
        "AR(1); `cw_p` is the Clark\u2013West one-sided p (nested), `cw_nonoverlap_p` restricts to "
        f"non-overlapping targets. Forecast gate (h={int(pooled['horizon'].min())}): "
        f"**{'PASS' if forecast_gate else 'FAIL'}**."
    )
    a("")

    a("## 4. Rank-pooled Shapley attribution")
    a("")
    a(
        f"Players are canonical ranks V1..V{K}; grouping boundary is the median spanned-block "
        f"size across windows (V1..V{median_boundary} | V{median_boundary + 1}..V{K}). "
        "Attribution is only interpretable for macro-stable ranks (see \u00a71)."
    )
    a("")
    a(_md_table(shapley[["horizon", "rank", "macro_stable", "phi", "boot_lo", "boot_hi"]],
                floatfmt="{:+.3e}"))
    a("")
    a("### Grouped (spanned vs weak)")
    a("")
    a(_md_table(grouped[["horizon", "phi_spanned", "phi_weak", "v_full"]], floatfmt="{:+.3e}"))
    a("")

    a("## Acceptance checks")
    a("")
    for name, ok, detail in acceptance:
        a(f"- [{'x' if ok else ' '}] {name} \u2014 {detail}")
    a("")

    path.write_text("\n".join(lines) + "\n")
