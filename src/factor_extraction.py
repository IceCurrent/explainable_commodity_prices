"""Re-estimate the rolling autoencoder and persist its latent factors.

The forecasting pipeline (``src/rolling_forecast.py``) re-trains the AE on every
rolling window but only keeps the most-recent factor vector ``f_t`` to produce a
one-step forecast, then discards the full latent series and the decoder weights.

Every mapping experiment in ``AE_factor_macro_mapping_directive.md`` consumes
those discarded objects:

* E0 (alignment) needs each window's *in-window* factor series ``f_{k,t}`` and
  the decoder weight columns.
* E1/E2 (macro mapping) need the *aligned daily* factor panel.
* E1 incremental-content + forecast-Shapley need the in-window factor series to
  cheaply refit the factor-augmented AR(1) on factor subsets.
* E3 needs the decoder Jacobian (= decoder weight, since the decoder is linear).

This module re-runs the identical rolling-window AE estimation and saves a
``FactorBundle`` containing all of the above so the downstream experiments never
have to retrain the AE.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.autoencoder import AETrainConfig, train_vanilla_autoencoder
from src.data import ReturnPanel, load_return_panel, zscore_window


@dataclass
class FactorExtractionConfig:
    window_size: int = 1260
    n_factors: int = 5
    ae_epochs: int = 50
    ae_patience: int = 5
    seed: int = 42
    stride: int = 1
    max_windows: int | None = None
    warm_start: bool = False


@dataclass
class FactorBundle:
    """Persisted output of the rolling AE estimation.

    Attributes
    ----------
    commodities : list[str]
        Asset names, length N (column order of every weight/factor array).
    forecast_dates : DatetimeIndex
        Forecast date d for each window (the day whose return the window is used
        to predict). Length W.
    as_of_dates : DatetimeIndex
        Information date of the live factor ``factors[w, -1]`` (= last in-window
        day, ``end_idx - 1``). This is the date the factor should be aligned to
        when regressing on a contemporaneous macro panel. Length W.
    window_start_idx, window_end_idx : ndarray[int]
        Global panel indices [start, end) of each window's estimation sample.
        ``factors[w]`` corresponds to panel rows ``start_idx[w]:end_idx[w]``.
    factors : ndarray, shape (W, M, K)
        In-window encoded latent factors (from z-scored window returns).
        ``factors[w, -1]`` is the live factor ``f_t`` used to forecast date d.
    decoder_weight : ndarray, shape (W, N, K)
        Decoder linear map / reconstruction Jacobian per window.
    decoder_bias : ndarray, shape (W, N)
    config : dict
        Extraction config used to produce the bundle.
    """

    commodities: list[str]
    forecast_dates: pd.DatetimeIndex
    as_of_dates: pd.DatetimeIndex
    window_start_idx: np.ndarray
    window_end_idx: np.ndarray
    factors: np.ndarray
    decoder_weight: np.ndarray
    decoder_bias: np.ndarray
    config: dict

    @property
    def n_windows(self) -> int:
        return self.factors.shape[0]

    @property
    def window_size(self) -> int:
        return self.factors.shape[1]

    @property
    def n_factors(self) -> int:
        return self.factors.shape[2]

    @property
    def n_assets(self) -> int:
        return len(self.commodities)

    def live_factors(self) -> np.ndarray:
        """Most-recent factor vector per window, shape (W, K)."""
        return self.factors[:, -1, :]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            commodities=np.array(self.commodities, dtype=object),
            forecast_dates=self.forecast_dates.values.astype("datetime64[ns]"),
            as_of_dates=self.as_of_dates.values.astype("datetime64[ns]"),
            window_start_idx=self.window_start_idx,
            window_end_idx=self.window_end_idx,
            factors=self.factors.astype(np.float32),
            decoder_weight=self.decoder_weight.astype(np.float32),
            decoder_bias=self.decoder_bias.astype(np.float32),
            config_keys=np.array(list(self.config.keys()), dtype=object),
            config_vals=np.array([str(v) for v in self.config.values()], dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "FactorBundle":
        with np.load(path, allow_pickle=True) as data:
            config = dict(zip(data["config_keys"].tolist(), data["config_vals"].tolist()))
            return cls(
                commodities=data["commodities"].tolist(),
                forecast_dates=pd.DatetimeIndex(data["forecast_dates"]),
                as_of_dates=pd.DatetimeIndex(data["as_of_dates"]),
                window_start_idx=data["window_start_idx"],
                window_end_idx=data["window_end_idx"],
                factors=data["factors"].astype(np.float64),
                decoder_weight=data["decoder_weight"].astype(np.float64),
                decoder_bias=data["decoder_bias"].astype(np.float64),
                config=config,
            )


def extract_factors(
    panel: ReturnPanel | None = None,
    config: FactorExtractionConfig | None = None,
    verbose: bool = True,
) -> FactorBundle:
    """Re-estimate the rolling AE and collect its latent factors + decoder.

    Mirrors ``run_rolling_forecasts`` window-for-window (same z-scoring, same AE
    training) so the persisted factors are exactly those the forecaster used.
    """
    panel = panel or load_return_panel()
    config = config or FactorExtractionConfig()

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

    forecast_indices = list(
        range(config.window_size, panel.n_obs, config.stride)
    )
    if config.max_windows is not None:
        forecast_indices = forecast_indices[: config.max_windows]

    n_windows = len(forecast_indices)
    M, N, K = config.window_size, panel.n_assets, config.n_factors

    factors = np.empty((n_windows, M, K), dtype=np.float32)
    decoder_weight = np.empty((n_windows, N, K), dtype=np.float32)
    decoder_bias = np.empty((n_windows, N), dtype=np.float32)
    start_idx_arr = np.empty(n_windows, dtype=np.int64)
    end_idx_arr = np.empty(n_windows, dtype=np.int64)
    fc_dates: list[pd.Timestamp] = []
    asof_dates: list[pd.Timestamp] = []

    warm_state = None
    for w, end_idx in enumerate(forecast_indices, start=0):
        if verbose and (w == 0 or (w + 1) % 50 == 0 or w == n_windows - 1):
            print(
                f"  window {w + 1}/{n_windows} "
                f"(forecast date {panel.dates[end_idx].date()})",
                flush=True,
            )
        start_idx = end_idx - config.window_size
        window_raw = panel.raw[start_idx:end_idx]
        window_z, _, _ = zscore_window(window_raw)

        model, win_factors = train_vanilla_autoencoder(
            window_z, ae_config, init_state=warm_state if config.warm_start else None
        )
        if config.warm_start:
            warm_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        factors[w] = win_factors.astype(np.float32)
        decoder_weight[w] = model.decoder_weight().astype(np.float32)
        decoder_bias[w] = model.decoder_bias().astype(np.float32)
        start_idx_arr[w] = start_idx
        end_idx_arr[w] = end_idx
        fc_dates.append(panel.dates[end_idx])
        asof_dates.append(panel.dates[end_idx - 1])

    return FactorBundle(
        commodities=panel.commodities,
        forecast_dates=pd.DatetimeIndex(fc_dates),
        as_of_dates=pd.DatetimeIndex(asof_dates),
        window_start_idx=start_idx_arr,
        window_end_idx=end_idx_arr,
        factors=factors,
        decoder_weight=decoder_weight,
        decoder_bias=decoder_bias,
        config=asdict(config),
    )
