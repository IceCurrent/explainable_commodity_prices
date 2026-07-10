.PHONY: install lint typecheck test macro-panel rolling rolling-beta \
        rolling-sectors rolling-sectors-beta all slow

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

# ---------------------------------------------------------------- rolling engine
# Everything re-fits per window: AE retrain -> CCA basis refit -> forecast the
# next test block -> advance. Nothing is trained on the full sample.

rolling:            # overall 21-commodity panel (vanilla AE)
	$(PYTHON) -m eqcp.pipelines.rolling_forecast_cli --seed 0 --factor-model vanilla

rolling-beta:       # overall panel (beta-VAE robustness arm)
	$(PYTHON) -m eqcp.pipelines.rolling_forecast_cli --seed 0 --factor-model beta_vae

rolling-sectors:    # per-sector decomposition (energy / agriculture / metals) + overall
	$(PYTHON) -m eqcp.pipelines.rolling_sectors --seed 0 --factor-model vanilla

rolling-sectors-beta:
	$(PYTHON) -m eqcp.pipelines.rolling_sectors --seed 0 --factor-model beta_vae

# Regenerate every rolling artifact the notebook reads (a few minutes total).
rolling-all: rolling rolling-beta rolling-sectors rolling-sectors-beta

all: install lint typecheck test
