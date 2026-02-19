## Ibook MLOps Platform

Production-grade MLOps platform for Ibook (ticketing): Feast feature store, MLflow experiment tracking and model registry, BentoML model serving (fraud detection and dynamic pricing), Airflow DAGs for feature engineering and training, Prometheus and Grafana for observability, and an event ticketing simulator for stress and scenario testing. Runs locally via Docker Compose.

### Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\pip install -e .
docker compose up -d --build
```

Then open **Airflow** http://localhost:8080 (admin/admin), **MLflow** http://localhost:5000, **Grafana** http://localhost:3000 (admin/admin). Full run options: [docs/how_to_run.md](docs/how_to_run.md).

### How to run

- Start the stack: `docker compose up -d --build`
- Optional: `make seed-data` then `make feast-apply` for Feast data
- Run the simulator: `make sim-list`, `make sim-run scenario=normal-traffic` (or use `python -m simulator.cli` as in [docs/how_to_run.md](docs/how_to_run.md))

Details and troubleshooting: [docs/how_to_run.md](docs/how_to_run.md).

### How to visualize

- **Grafana** (http://localhost:3000) — MLOps Overview dashboard (latency, request rate, error rate, drift)
- **MLflow** (http://localhost:5000) — Experiments and model registry
- **Airflow** (http://localhost:8080) — DAG runs and task logs

Observability and metrics: [docs/Observability.md](docs/Observability.md).

### Testing

Unit and integration: `make test` or `python -m pytest tests/ -v`. Simulator: `make sim-list`, `make sim-run scenario=normal-traffic`, `make sim-run-all`. Load tests (BentoML up): `locust -f tests/e2e/test_load_performance.py --host http://localhost:7001`. See [docs/how_to_run.md](docs/how_to_run.md)#run-tests and [docs/simulator_testing.md](docs/simulator_testing.md).

### Simulator

Scenarios: `normal-traffic`, `flash-sale`, `fraud-attack`, `gradual-drift`, `system-degradation`, `black-friday`, `mix`. Design and testing: [docs/SIMULATOR.md](docs/SIMULATOR.md), [docs/simulator_testing.md](docs/simulator_testing.md).

### More documentation

- [docs/how_it_works.md](docs/how_it_works.md) — Architecture, algorithms, data flows
- [docs/how_to_run.md](docs/how_to_run.md) — Run instructions and troubleshooting
- [docs/Observability.md](docs/Observability.md) — Metrics, dashboards, health checks
- [docs/SIMULATOR.md](docs/SIMULATOR.md) — Simulator design and usage
- [docs/simulator_testing.md](docs/simulator_testing.md) — Scenario testing guide
- [docs/qa.md](docs/qa.md) — Q&A
- [docs/PLAN.md](docs/PLAN.md) — Implementation plan and status
- [docs/tech_blog_self_healing_fraud_mlops.md](docs/tech_blog_self_healing_fraud_mlops.md) — Tech blog: self-healing fraud MLOps
