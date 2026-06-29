"""Run the synthetic AE-explainability sweeps.

Three sweeps, each isolating one mechanism of explainability loss:

  A. bottleneck rank    -- K in {r-1, r, r+1, 2r} on a clean linear DGP.
                           Tests rank mismatch and ReLU factor-splitting
                           (does a signed rank-r subspace need K>r ReLU units?).
  B. encoder vs noise   -- relu vs linear vs pca across SNR, at K=r.
                           Isolates the cost of the ReLU and of noise.
  C. linear vs nonlinear-- linear / interaction / tanh relations, with E2 on.
                           Tests when the nonlinearity hides the relation from a
                           linear probe (E2 premium) vs destroys it outright.

Writes results/synthetic/sweep_{a,b,c}.csv and a markdown summary.

    python -m scripts.run_synthetic_recovery
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.synthetic_recovery import (
    SyntheticConfig,
    result_to_row,
    run_recovery,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "synthetic"


def sweep_a_rank() -> pd.DataFrame:
    rows = []
    r = 3
    for encoder in ("relu", "linear", "pca"):
        for k in (r - 1, r, r + 1, 2 * r):
            cfg = SyntheticConfig(true_rank=r, relation="linear", snr=6.0)
            rows.append(result_to_row(run_recovery(cfg, n_factors=k, encoder=encoder)))
    return pd.DataFrame(rows)


def sweep_b_encoder_noise() -> pd.DataFrame:
    rows = []
    r = 3
    for snr in (1.0, 2.0, 4.0, 8.0, 16.0):
        for encoder in ("relu", "linear", "pca"):
            cfg = SyntheticConfig(true_rank=r, relation="linear", snr=snr)
            rows.append(result_to_row(run_recovery(cfg, n_factors=r, encoder=encoder)))
    return pd.DataFrame(rows)


def sweep_c_nonlinearity() -> pd.DataFrame:
    rows = []
    r = 3
    for relation in ("linear", "interaction", "tanh"):
        for encoder in ("relu", "linear", "pca"):
            cfg = SyntheticConfig(true_rank=r, relation=relation, snr=6.0)
            # K = r+1 so the interaction's extra signal dimension has room.
            rows.append(result_to_row(run_recovery(cfg, n_factors=r + 1, encoder=encoder, run_e2=True)))
    return pd.DataFrame(rows)


def _fmt(df: pd.DataFrame) -> str:
    return df.round(3).to_markdown(index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = sweep_a_rank()
    b = sweep_b_encoder_noise()
    c = sweep_c_nonlinearity()
    a.to_csv(OUT / "sweep_a_rank.csv", index=False)
    b.to_csv(OUT / "sweep_b_encoder_noise.csv", index=False)
    c.to_csv(OUT / "sweep_c_nonlinearity.csv", index=False)

    report = "\n\n".join([
        "# Synthetic AE-explainability sweeps",
        "Ground-truth recovery of a known relation through the project AE. "
        "`span_mean_r2` is the E1 per-factor (coordinate-level) recovery; "
        "`canon_min` is the E1 Bai-Ng *linear* (space-level) recovery; "
        "`kcca_min` is the E2 kernel-CCA *nonlinear* (space-level) recovery -- "
        "`kcca_min` >> `canon_min` means the f->g relation is real but nonlinear, "
        "so the linear probe understates it (space-level nonlinearity premium); "
        "`subspace_min` is decoder-span vs true-loading-span overlap.",
        "## A. Bottleneck rank (linear DGP, true_rank=3, snr=6)\n\n" + _fmt(a),
        "## B. Encoder vs noise (K=true_rank=3)\n\n" + _fmt(b),
        "## C. Linear vs nonlinear relation (K=4, E2 on)\n\n" + _fmt(c),
    ])
    (OUT / "synthetic_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
