"""Save per-architecture factor and reconstruction error CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _mse_table(errors_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, err in errors_dict.items():
        per_com = (err ** 2).mean()
        per_com["ALL"] = float((err.to_numpy() ** 2).mean())
        rows[name] = per_com
    return pd.DataFrame(rows).T


def summary_table(
    results: dict[str, dict], baseline: str = "VanillaAE"
) -> pd.DataFrame:
    train_errs = {n: r["train"]["recon_error"] for n, r in results.items()}
    test_errs = {n: r["test"]["recon_error"] for n, r in results.items()}
    train_mses = _mse_table(train_errs)
    test_mses = _mse_table(test_errs)
    base_test_all = test_mses.loc[baseline, "ALL"]
    base_per_com = test_mses.loc[baseline].drop("ALL")

    rows = []
    for name in results:
        per_com = test_mses.loc[name].drop("ALL")
        ratios = per_com / base_per_com
        rows.append(
            {
                "architecture": name,
                "train_recon_mse": train_mses.loc[name, "ALL"],
                "test_recon_mse": test_mses.loc[name, "ALL"],
                "test_mse_ratio": test_mses.loc[name, "ALL"] / base_test_all,
                "median_commodity_ratio": float(ratios.median()),
                "n_commodities_better": int((ratios < 1).sum()),
            }
        )
    out = pd.DataFrame(rows).set_index("architecture")
    return out.sort_values("test_mse_ratio")


def write_all_outputs(
    results: dict[str, dict],
    results_dir: str | Path,
    baseline: str = "VanillaAE",
) -> pd.DataFrame:
    results_dir = Path(results_dir)
    factors_dir = results_dir / "factors"
    errors_dir = results_dir / "errors"
    factors_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(exist_ok=True)

    for name, res in results.items():
        res["train"]["factors"].to_csv(factors_dir / f"{name}_train_factors.csv")
        res["test"]["factors"].to_csv(factors_dir / f"{name}_test_factors.csv")
        res["train"]["recon_error"].to_csv(errors_dir / f"{name}_train_recon_error.csv")
        res["test"]["recon_error"].to_csv(errors_dir / f"{name}_test_recon_error.csv")

    return summary_table(results, baseline)
