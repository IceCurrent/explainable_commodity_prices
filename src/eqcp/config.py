"""Typed configuration loaders for YAML config files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


@dataclass
class FactorModelConfig:
    n_factors: int = 5
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 5
    seed: int = 0
    activation: str = "relu"
    beta: float = 4.0


@dataclass
class MacroProcessingConfig:
    window: tuple[str, str] = ("2013-10-18", "2026-06-01")
    bridge_limit: int = 10
    long_bridge_flag: int = 7
    transforms: dict[str, str] = field(default_factory=dict)
    bbg_sheets: list[str] = field(default_factory=list)


@dataclass
class RollingForecastConfig:
    """Walk-forward rolling-window forecast + rolling-explainability experiment.

    Everything re-fits on each window: the AE is retrained on ``train_window``
    days, the CCA attribution basis is refit, then the factor-augmented AR
    forecasts the next ``test_block`` days before the window advances by
    ``step``. The scientific target is whether the factor<->macro canonical
    structure survives this rolling (ranks stay macro-anchored even as the AE
    latent coordinates rotate window to window).
    """

    train_window: int = 252
    test_block: int = 21
    step: int = 21
    horizons: tuple[int, ...] = (1, 5, 21, 63)
    min_train_pairs: int = 40
    ae_epochs: int = 60  # per-window AE budget (windows are short; keeps the roll fast)
    ridge_grid: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 1.0)
    n_folds: int = 4
    embargo: int = 10
    spanned_rho_min: float = 0.3
    n_boot: int = 1000
    mean_block: int = 21
    stability_seeds: tuple[int, ...] = (0, 1, 2)
    lagged_macro_series: tuple[str, ...] = ("gpr", "epu")
    fingerprint_cos_stable: float = 0.7  # rank is "macro-stable" above this median cosine
    n_perm_stability: int = 200  # label-shuffle permutations for the stability null


def load_rolling_forecast_config(path: Path | None = None) -> RollingForecastConfig:
    data = _load_yaml(path or CONFIG_DIR / "rolling_forecast.yaml")
    for key in ("horizons", "ridge_grid", "stability_seeds", "lagged_macro_series"):
        if key in data:
            data[key] = tuple(data[key])
    return RollingForecastConfig(**data)


def load_factor_model_config(path: Path | None = None) -> FactorModelConfig:
    data = _load_yaml(path or CONFIG_DIR / "factor_model.yaml")
    return FactorModelConfig(**data)


def load_macro_processing_config(path: Path | None = None) -> MacroProcessingConfig:
    data = _load_yaml(path or CONFIG_DIR / "macro_processing.yaml")
    window = tuple(data.pop("window"))
    return MacroProcessingConfig(window=window, **data)
