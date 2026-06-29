from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.autoencoder import AETrainConfig, train_vanilla_autoencoder
from src.data import ReturnPanel, load_return_panel, zscore_window
from src.forecast_models import (
    fit_ar1,
    fit_factor_augmented_ar1,
    forecast_ar1,
    forecast_factor_augmented_ar1,
)


@dataclass
class RollingForecastConfig:
    window_size: int = 1260
    n_factors: int = 5
    ae_epochs: int = 50
    ae_patience: int = 5
    seed: int = 42
    max_windows: int | None = None


@dataclass
class ForecastResults:
    forecasts: pd.DataFrame
    errors: pd.DataFrame
    summary: pd.DataFrame


def run_rolling_forecasts(
    panel: ReturnPanel | None = None,
    config: RollingForecastConfig | None = None,
) -> ForecastResults:
    """
    Rolling-window one-step-ahead forecasts for AR(1) and AE factor-augmented AR(1).

    For each forecast date d = M+1, ..., T:
      1. Estimate window on [d-M, ..., d-1]
      2. Z-score returns inside the window (no look-ahead)
      3. Train vanilla AE and extract latent factors
      4. Fit AR(1) and factor-augmented AR(1) per commodity
      5. Forecast return at date d
    """
    panel = panel or load_return_panel()
    config = config or RollingForecastConfig()

    if config.window_size >= panel.n_obs:
        raise ValueError(
            f"window_size ({config.window_size}) must be smaller than T ({panel.n_obs})"
        )

    ae_config = AETrainConfig(
        n_factors=config.n_factors,
        epochs=config.ae_epochs,
        patience=config.ae_patience,
        seed=config.seed,
    )

    forecast_start = config.window_size
    forecast_end = panel.n_obs
    forecast_indices = range(forecast_start, forecast_end)
    if config.max_windows is not None:
        forecast_indices = list(forecast_indices)[: config.max_windows]

    records: list[dict[str, object]] = []
    forecast_indices = list(forecast_indices)
    n_total = len(forecast_indices)

    for step, end_idx in enumerate(forecast_indices, start=1):
        if step == 1 or step % 50 == 0 or step == n_total:
            print(
                f"  window {step}/{n_total} (forecast date {panel.dates[end_idx].date()})",
                flush=True,
            )
        start_idx = end_idx - config.window_size
        window_raw = panel.raw[start_idx:end_idx]
        window_z, _, _ = zscore_window(window_raw)

        _, factors = train_vanilla_autoencoder(window_z, ae_config)

        y_t = panel.raw[end_idx - 1]
        f_t = factors[-1]
        actual = panel.raw[end_idx]
        date = panel.dates[end_idx]

        for asset_idx, commodity in enumerate(panel.commodities):
            y_window = window_raw[:, asset_idx]

            ar1_coef = fit_ar1(y_window)
            ar1_pred = forecast_ar1(ar1_coef, y_t[asset_idx])

            fa_coef = fit_factor_augmented_ar1(y_window, factors)
            fa_pred = forecast_factor_augmented_ar1(
                fa_coef,
                y_t[asset_idx],
                f_t,
            )

            records.append(
                {
                    "date": date,
                    "commodity": commodity,
                    "actual": actual[asset_idx],
                    "forecast_ar1": ar1_pred,
                    "forecast_vanilla_ae": fa_pred,
                    "error_ar1": actual[asset_idx] - ar1_pred,
                    "error_vanilla_ae": actual[asset_idx] - fa_pred,
                }
            )

    forecasts = pd.DataFrame(records)
    forecasts["date"] = pd.to_datetime(forecasts["date"])

    errors = forecasts[
        ["date", "commodity", "error_ar1", "error_vanilla_ae"]
    ].copy()

    summary_rows = []
    for commodity in panel.commodities:
        sub = forecasts[forecasts["commodity"] == commodity]
        summary_rows.append(
            {
                "commodity": commodity,
                "n_forecasts": len(sub),
                "mse_ar1": float(np.mean(sub["error_ar1"] ** 2)),
                "mse_vanilla_ae": float(np.mean(sub["error_vanilla_ae"] ** 2)),
                "mae_ar1": float(np.mean(np.abs(sub["error_ar1"]))),
                "mae_vanilla_ae": float(np.mean(np.abs(sub["error_vanilla_ae"]))),
            }
        )

    summary = pd.DataFrame(summary_rows)
    overall = {
        "commodity": "ALL",
        "n_forecasts": len(forecasts),
        "mse_ar1": float(np.mean(forecasts["error_ar1"] ** 2)),
        "mse_vanilla_ae": float(np.mean(forecasts["error_vanilla_ae"] ** 2)),
        "mae_ar1": float(np.mean(np.abs(forecasts["error_ar1"]))),
        "mae_vanilla_ae": float(np.mean(np.abs(forecasts["error_vanilla_ae"]))),
    }
    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

    return ForecastResults(forecasts=forecasts, errors=errors, summary=summary)


def save_results(results: ForecastResults, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.forecasts.to_csv(output_dir / "forecasts.csv", index=False)
    results.errors.to_csv(output_dir / "errors.csv", index=False)
    results.summary.to_csv(output_dir / "summary.csv", index=False)

    with (output_dir / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write("# Rolling forecast summary\n\n")
        fh.write("Models: AR(1) baseline and factor-augmented AR(1) with vanilla autoencoder (K=5).\n\n")
        fh.write(results.summary.to_markdown(index=False))
        fh.write("\n")
