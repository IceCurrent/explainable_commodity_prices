.PHONY: install lint typecheck test macro-panel mapping synthetic all slow

PYTHON ?= python3
PIP ?= pip
PYTEST ?= pytest

install:
	$(PIP) install -e ".[dev]"

lint:
	ruff check src/eqcp tests

typecheck:
	mypy src/eqcp

test:
	$(PYTEST) -q -m "not slow"

slow:
	$(PYTEST) -q -m slow

macro-panel:
	$(PYTHON) -m eqcp.macro_processing.process_macro \
		--input data/macro/NEW_MACRO_COMMODITY_PANEL.xlsx \
		--outdir .

mapping:
	$(PYTHON) scripts/run_macro_mapping.py \
		--factors data/processed/ae_factors_vanilla.csv \
		--macro data/processed/macro_stationary.csv \
		--levels data/processed/macro_levels_aligned.csv \
		--regimes data/processed/regimes.csv \
		--outdir . \
		--seed 0

synthetic:
	$(PYTHON) scripts/run_synthetic_recovery.py

all: install lint typecheck test
