.PHONY: install lint typecheck test macro-panel mapping mapping-beta synthetic forecast forecast-beta sectors all slow

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
		--seed 0 \
		--factor-model vanilla

mapping-beta:
	$(PYTHON) scripts/run_macro_mapping.py \
		--factors data/processed/ae_factors_beta.csv \
		--macro data/processed/macro_stationary.csv \
		--levels data/processed/macro_levels_aligned.csv \
		--regimes data/processed/regimes.csv \
		--outdir . \
		--seed 0 \
		--factor-model beta_vae

synthetic:
	$(PYTHON) scripts/run_synthetic_recovery.py

forecast:
	$(PYTHON) -m eqcp.pipelines.forecast_pbsv_cli \
		--macro data/processed/macro_stationary.csv \
		--manifest data/processed/transform_manifest.csv \
		--outdir . \
		--seed 0 \
		--factor-model vanilla

forecast-beta:
	$(PYTHON) -m eqcp.pipelines.forecast_pbsv_cli \
		--macro data/processed/macro_stationary.csv \
		--manifest data/processed/transform_manifest.csv \
		--outdir . \
		--seed 0 \
		--factor-model beta_vae

sectors:
	$(PYTHON) -m eqcp.pipelines.sector_analysis \
		--macro data/processed/macro_stationary.csv \
		--manifest data/processed/transform_manifest.csv \
		--outdir . \
		--seed 0

all: install lint typecheck test
