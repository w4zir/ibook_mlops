.PHONY: setup start stop restart logs seed-data feast-apply test clean format deploy-prod serve-fraud serve-pricing serve-bento stream-start stream-logs sim-list sim-run sim-run-all sim-mix sim-realtime sim-stop

PYTHON ?= python

setup:
	@echo "Setting up Python environment (venv creation is optional)..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt
	@$(PYTHON) -m pip install -e .

start:
	@docker compose up -d

stop:
	@docker compose down

restart: stop start

logs:
	@docker compose logs -f

seed-data:
	@echo "Generating synthetic Feast data into data/processed/feast/..."
	@$(PYTHON) scripts/seed-data.py

feast-apply:
	@echo "Applying Feast feature repository (local DuckDB + Redis)..."
	@feast -c services/feast/feature_repo apply

test:
	@$(PYTHON) -m pytest tests/ -v --tb=short

serve-fraud:
	@echo "Starting BentoML fraud detection service (port 7001)..."
	@docker compose up -d bentoml-fraud

serve-pricing:
	@echo "Starting BentoML dynamic pricing service (port 7002)..."
	@docker compose up -d bentoml-pricing

serve-bento: serve-fraud serve-pricing

stream-start:
	@docker compose up -d faust-worker parquet-sink

stream-logs:
	@docker compose logs -f faust-worker parquet-sink

clean:
	@docker compose down -v

format:
	@$(PYTHON) -m black common tests scripts || true
	@$(PYTHON) -m flake8 common tests scripts || true

deploy-prod:
	@echo "Phase 1: production deployment comes later (see scripts/deploy-production.sh in Phase 8)."
	@exit 0

# Simulator commands (run from project root after: pip install -e .)
sim-list:
	@$(PYTHON) -m simulator.cli list-scenarios

sim-run:
	@$(PYTHON) -m simulator.cli run $(scenario) -o reports/$(scenario)-report.html

sim-run-all:
	@$(PYTHON) -m simulator.cli run-all -o reports/

sim-mix:
	@$(PYTHON) -m simulator.cli mix --duration $(if $(duration),$(duration),30) -o reports/mix-report.html

sim-realtime:
	@$(PYTHON) -m simulator.cli realtime $(if $(scenario),$(scenario),normal-traffic) --duration $(if $(duration),$(duration),60) --rps $(if $(rps),$(rps),100) -o reports/realtime-report.html

sim-stop:
	@echo "Simulator runs in-process; use Ctrl+C to stop."

