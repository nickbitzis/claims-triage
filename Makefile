PORT ?= 8834
CLAIMS ?= 60000

.PHONY: install test lint format pipeline evaluate gate tune run docker-build compose-up screenshot clean

install:
	uv sync --python 3.12

test:
	uv run pytest -q

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff format .
	uvx ruff check . --fix

pipeline:  ## generate, process, train, route, evaluate (about 75 seconds at 60,000 claims)
	uv run claimstriage all --claims $(CLAIMS)

evaluate:
	uv run claimstriage evaluate

gate:  ## the CI gate: fail if model-policy leakage plus missed fraud exceeds the rules-only baseline
	uv run claimstriage --data-dir data/ci all --claims $(CLAIMS) --gate

tune:  ## grid search routing thresholds on the calibration months
	uv run claimstriage tune

run:
	PORT=$(PORT) uv run claimstriage serve --port $(PORT)

docker-build:
	docker build -t claims-triage .

compose-up:
	docker compose up --build

screenshot:  ## needs the server running on $(PORT) and Google Chrome
	uv run python scripts/screenshot.py --port $(PORT)

clean:
	rm -rf .pytest_cache .ruff_cache data
