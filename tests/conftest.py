"""Shared pytest fixtures (editable install: ``pip install -e .``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
