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


@dataclass
class MacroProcessingConfig:
    window: tuple[str, str] = ("2013-10-18", "2026-06-01")
    bridge_limit: int = 10
    long_bridge_flag: int = 7
    transforms: dict[str, str] = field(default_factory=dict)
    bbg_sheets: list[str] = field(default_factory=list)


@dataclass
class MacroMappingConfig:
    ridge_grid: tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1, 1.0)
    n_perm_linear: int = 1000
    n_perm_kcca: int = 200
    n_perm_regime: int = 500
    n_boot: int = 1000
    mean_block: int = 21
    n_folds: int = 5
    embargo: int = 10
    kcca_reg: float = 1.0
    kcca_reg_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    kcca_gamma_scale: tuple[float, ...] = (0.5, 1.0, 2.0)
    kcca_stability_iqr_threshold: float = 0.15
    nystrom_landmarks: int = 400
    lags: list[int] = field(default_factory=lambda: list(range(-5, 6)))
    boot_blocks: tuple[int, ...] = (10, 21, 42)
    china_seam: str = "2016-08-02"
    aligned_t_min: int = 2690
    bloc_map: dict[str, list[str]] = field(default_factory=dict)
    encoder_activations: tuple[str, ...] = ("relu", "tanh", "linear")


def load_factor_model_config(path: Path | None = None) -> FactorModelConfig:
    data = _load_yaml(path or CONFIG_DIR / "factor_model.yaml")
    return FactorModelConfig(**data)


def load_macro_processing_config(path: Path | None = None) -> MacroProcessingConfig:
    data = _load_yaml(path or CONFIG_DIR / "macro_processing.yaml")
    window = tuple(data.pop("window"))
    return MacroProcessingConfig(window=window, **data)


def load_macro_mapping_config(path: Path | None = None) -> MacroMappingConfig:
    data = _load_yaml(path or CONFIG_DIR / "macro_mapping.yaml")
    for key in (
        "ridge_grid",
        "kcca_reg_grid",
        "kcca_gamma_scale",
        "boot_blocks",
        "encoder_activations",
    ):
        if key in data:
            data[key] = tuple(data[key])
    if "lags" in data:
        data["lags"] = list(data["lags"])
    return MacroMappingConfig(**data)
