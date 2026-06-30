#!/usr/bin/env python3
"""CLI entry point for synthetic recovery sweeps."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from eqcp.synthetic.recovery import SyntheticConfig, result_to_row, run_recovery


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
            rows.append(
                result_to_row(run_recovery(cfg, n_factors=r + 1, encoder=encoder, run_e2=True))
            )
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Synthetic AE-explainability sweeps.")
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.info("seed=%s", args.seed)
    out = args.outdir / "results" / "synthetic"
    out.mkdir(parents=True, exist_ok=True)

    a = sweep_a_rank()
    b = sweep_b_encoder_noise()
    c = sweep_c_nonlinearity()
    a.to_csv(out / "sweep_a_rank.csv", index=False)
    b.to_csv(out / "sweep_b_encoder_noise.csv", index=False)
    c.to_csv(out / "sweep_c_nonlinearity.csv", index=False)

    def fmt(df: pd.DataFrame) -> str:
        return df.round(3).to_markdown(index=False)

    report = "\n\n".join(
        [
            "# Synthetic AE-explainability sweeps",
            "Ground-truth recovery through the project AE.",
            "## A. Bottleneck rank\n\n" + fmt(a),
            "## B. Encoder vs noise\n\n" + fmt(b),
            "## C. Linear vs nonlinear\n\n" + fmt(c),
        ]
    )
    (out / "synthetic_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
