## How to run (Phases 1–5)

This repo currently covers **Phases 1–5**:

- Phase 1: local service stack + config module + basic unit tests.
- Phase 2: Feast feature repository + synthetic sample data + feature utilities.
- Phase 3: MLflow service hardening + fraud model training utilities and notebook.
- Phase 4: BentoML model serving for fraud detection and dynamic pricing.
- Phase 5: Airflow orchestration DAGs for feature engineering, training, and monitoring.

### Prerequisites

- **Docker Desktop** (with Compose)
- **Python** 3.10+ (this repo was tested with Python 3.12)
- **Git**

Optional:
- **WSL / Git Bash** if you want to use the provided `Makefile`

---

## One-time setup (PowerShell)

From repo root (`d:\\ai_ws\\projects\\ibook_ai_ops`):

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\pip install -e .
```

Notes:
- `.env` holds local defaults (safe for dev). Update values if ports or credentials conflict on your machine.

---

## Start the local stack

### PowerShell (recommended on Windows)

```bash
docker compose up -d --build
docker compose ps
```

Stop:

```bash
docker compose down
```

Clean (also removes volumes):

```bash
docker compose down -v
```

### Make (WSL / Git Bash / Linux / macOS)

```bash
make setup
make start
make logs
make stop
```

---

## Service URLs (defaults)

- **MLflow**: `http://localhost:5000`
- **Airflow**: `http://localhost:8080` (default user/pass: `admin` / `admin`)
- **MinIO API**: `http://localhost:9000`
- **MinIO console**: `http://localhost:9001` (default user/pass: `minioadmin` / `minioadmin`)
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (default user/pass: `admin` / `admin`)
- **Jupyter**: `http://localhost:8888` (token disabled in Phase 1 container)
- **BentoML fraud service**: `http://localhost:7001`
- **BentoML dynamic pricing service**: `http://localhost:7002`

---

## Run tests (no Docker required)

### PowerShell

```bash
.venv\Scripts\python -m pytest tests\ -v --tb=short
```

### Make

```bash
make test
```

For Phase 3 specifically, the `tests/unit/test_models.py` suite exercises the
fraud training utilities and their MLflow integration.

For Phase 4, additional tests cover:

- BentoML shared utilities and runtime helpers: `tests/unit/test_bentoml_common.py`
- Service-level fraud and pricing logic (without starting HTTP servers):
  - `tests/integration/test_api_endpoints.py`

You can run just the Phase 4-related tests with:

```bash
pytest tests/unit/test_bentoml_common.py tests/integration/test_api_endpoints.py -v --tb=short
```

For Phase 5, additional tests cover:

- Airflow DAG importability and basic topology: `tests/unit/test_airflow_dags.py`

You can run just the Phase 5-related tests with:

```bash
pytest tests/unit/test_airflow_dags.py -v --tb=short
```

---

## Phase 2: Feature store & sample data

Phase 2 adds:
- A Feast feature repo in `services/feast/feature_repo/`.
- Synthetic Parquet datasets under `data/processed/feast/` via `scripts/seed-data.py`.
- Convenience helpers in `common/feature_utils.py`.

### Generate synthetic data

From the repo root:

#### PowerShell

```bash
.venv\Scripts\python scripts\seed-data.py
```

#### Make (WSL / Git Bash / Linux / macOS)

```bash
make seed-data
```

This will create:
- `data/processed/feast/event_metrics.parquet`
- `data/processed/feast/user_metrics.parquet`

### Apply the Feast feature repo (local)

After generating data, apply the Feast definitions so the registry is created:

```bash
feast -c services/feast/feature_repo apply
```

If you prefer using `make`:

```bash
make feast-apply
```

### Example: fetch online features (Python)

Once Redis and the local stack are running (`docker compose up -d`), you can
experiment with online features from a Python REPL or notebook:

```python
from common.feature_utils import fetch_online_features

rows = [{"event_id": 1}]
features = ["event_realtime_metrics:current_inventory"]

df = fetch_online_features(features=features, entity_rows=rows)
print(df)
```

---

## Phase 3: MLflow & fraud model training

Phase 3 adds:
- A hardened MLflow service container with a `/healthz` endpoint.
- Training utilities in `common/model_utils.py` (Optuna + XGBoost + SHAP + MLflow).
- A runnable notebook `notebooks/03_model_training_fraud.ipynb`.

### Ensure MLflow is running

From the repo root:

```bash
docker compose up -d --build
docker compose ps
```

You should see `ibook-mlflow` healthy (the container healthcheck hits `/healthz`
on port `5001`), and the UI available at:

- **MLflow UI**: `http://localhost:5000`

### Run the fraud training notebook

1. Start the Jupyter container (already part of `docker compose up`).
2. Open `http://localhost:8888` in your browser.
3. In the Jupyter file browser, open `notebooks/03_model_training_fraud.ipynb`.
4. Run the cells top‑to‑bottom.

The notebook will:
- Generate a small synthetic dataset (reusing `scripts/seed-data.py`).
- Build a training DataFrame via `build_fraud_training_dataframe`.
- Call `train_fraud_model`, which logs parameters, metrics, model, and SHAP
  artifacts to MLflow under the `fraud_detection` experiment.

You can inspect the runs in the MLflow UI at `http://localhost:5000`.

---

## Phase 4: Model serving with BentoML

Phase 4 adds:

- A BentoML fraud detection service under `services/bentoml/services/fraud_detection/`.
- A BentoML dynamic pricing service under `services/bentoml/services/dynamic_pricing/`.
- Shared BentoML utilities under `services/bentoml/common/` (config, MLflow/Feast clients, Prometheus metrics).

Both services are included in the Docker Compose stack and can also be started via the `Makefile`.

### Start BentoML services with Docker Compose

If you started the full stack with:

```bash
docker compose up -d --build
```

then the BentoML services will be built and started alongside the other containers. You should see:

- Fraud service: `ibook-bentoml-fraud` (port `7001`)
- Dynamic pricing service: `ibook-bentoml-pricing` (port `7002`)

You can also start them explicitly:

```bash
docker compose up -d bentoml-fraud
docker compose up -d bentoml-pricing
```

### Start BentoML services via Makefile

From environments with `make` available:

```bash
make serve-fraud   # starts fraud detection service on http://localhost:7001
make serve-pricing # starts dynamic pricing service on http://localhost:7002
make serve-bento   # starts both services
```

### Quick smoke tests (curl)

With the stack running, you can hit the health endpoints:

```bash
curl http://localhost:7001/healthz
curl http://localhost:7002/healthz
```

Example fraud prediction request (JSON batch with a single element):

```bash
curl -X POST http://localhost:7001/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"requests\":[{\"user_id\":1,\"event_id\":2,\"amount\":100.0}]}"
```

Example pricing recommendation request:

```bash
curl -X POST http://localhost:7002/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"requests\":[{\"event_id\":1,\"current_price\":100.0}]}"
```

---

## Troubleshooting

- **Ports already in use**: stop conflicting local services or change the exposed ports in `docker-compose.yml`.
- **Reset everything**:

```bash
docker compose down -v
docker compose up -d --build
```

---

## Phase 5: Airflow workflows

Phase 5 adds three orchestration DAGs under `services/airflow/dags/`:

- `feature_engineering_pipeline.py`
- `model_training_pipeline.py`
- `ml_monitoring_pipeline.py`

These DAGs are mounted into the official `apache/airflow:2.8.1` containers via
Docker volumes and are safe to run locally with the existing stack.

### View and run DAGs in the Airflow UI

With the stack running:

```bash
docker compose up -d --build
docker compose ps
```

Open the Airflow UI:

- **Airflow**: `http://localhost:8080` (user/pass: `admin` / `admin`)

In the UI:

1. Locate the DAGs named:
   - `feature_engineering_pipeline`
   - `model_training_pipeline`
   - `ml_monitoring_pipeline`
2. Unpause each DAG.
3. Trigger a manual run for smoke testing (e.g., via the \"Play\" button).

### What the Phase 5 DAGs do (locally)

- **Feature engineering DAG (`feature_engineering_pipeline`)**:
  - Optionally reads the synthetic Feast Parquet data under `data/processed/feast/`
    if you have already run `scripts/seed-data.py`.
  - Computes a small aggregate file `data/processed/feast/event_aggregates.parquet`.
  - Performs basic data checks (standing in for Great Expectations).
  - Touches the Feast repo via a lightweight healthcheck.
  - Logs whether it would trigger training when drift is detected.

- **Model training DAG (`model_training_pipeline`)**:
  - Builds a small synthetic user-metrics DataFrame in memory.
  - Uses `common/model_utils.py` to construct a fraud training dataset and run a
    short XGBoost + Optuna training loop that logs to MLflow.
  - Compares metrics to simple static baselines and logs whether the candidate
    model would be accepted.
  - Includes stub tasks that represent MLflow model registration, canary deploy,
    and final promotion decisions.

- **Monitoring DAG (`ml_monitoring_pipeline`)**:
  - Generates synthetic predictions and labels to mimic production behavior.
  - Writes a small JSON summary under `data/monitoring/daily_drift_summary.json`.
  - Applies a simple drift threshold and logs whether alerts/retraining would
    be triggered.
  - Contains stub tasks for alerting (Slack/PagerDuty) and retraining triggers.

All external integrations (Kafka, Evidently, Slack/PagerDuty, GCS) are stubbed
out so that the DAGs are fast and reliable in local development.


