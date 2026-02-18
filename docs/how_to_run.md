## How to run

This repo covers the full MLOps platform: local service stack, Feast feature store, MLflow, BentoML model serving, Airflow DAGs, monitoring (Prometheus/Grafana/Evidently), and the **event ticketing simulator** for stress and scenario testing.

---

### Quick Start (TL;DR)

From repo root (PowerShell; use `;` between commands, not `&&`):

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\pip install -e .
docker compose up -d --build
```

Then open: **Airflow** http://localhost:8080 (admin/admin), **MLflow** http://localhost:5000, **Grafana** http://localhost:3000 (admin/admin).  
Optional: `make seed-data` then `make feast-apply` for Feast data; run simulator with `make sim-list` and `make sim-run scenario=normal-traffic`.

---

### Prerequisites

- **Docker Desktop** (with Compose)
- **Python** 3.10+ (tested with Python 3.12)
- **Git**

Optional: **WSL / Git Bash** if you want to use the provided `Makefile`.

---

### One-time setup

From repo root:

**PowerShell:**

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\pip install -e .
```

**Make (WSL / Git Bash / Linux / macOS):**

```bash
make setup
```

Notes:

- `.env` holds local defaults (PostgreSQL, Redis, MinIO, MLflow, Airflow, Feast). Update values if ports or credentials conflict on your machine.

---

### Start the local stack

**PowerShell** (run from repo root; if you see conda/pydantic errors, they are from your host Python and can be ignored):

```powershell
docker compose up -d --build
docker compose ps
```

**Make:**

```bash
make start
make logs   # follow logs
make stop  # stop all services
```

Stop and remove volumes:

```bash
docker compose down -v
```

---

### Service URLs (defaults)

| Service | URL | Notes |
|--------|-----|--------|
| **MLflow** | http://localhost:5000 | Experiment tracking & model registry |
| **MLflow health** | http://localhost:5001/healthz | Health check (container) |
| **Airflow** | http://localhost:8080 | User: `admin` / Pass: `admin` |
| **MinIO API** | http://localhost:9000 | S3-compatible storage |
| **MinIO console** | http://localhost:9001 | User: `minioadmin` / Pass: `minioadmin` |
| **Prometheus** | http://localhost:9090 | Metrics & alerts |
| **Grafana** | http://localhost:3000 | User: `admin` / Pass: `admin` |
| **Jupyter** | http://localhost:8888 | Token disabled in dev container |
| **BentoML fraud** | http://localhost:7001 | Fraud detection API |
| **BentoML pricing** | http://localhost:7002 | Dynamic pricing API |
| **PostgreSQL** | localhost:5432 | Airflow + MLflow metadata (user: `ibook`) |
| **Redis** | localhost:6379 | Feast online store |
| **Kafka** | localhost:9092 | Event streaming. Host access: `localhost:9092`; from other containers use `kafka:29092`. |
| **Zookeeper** | localhost:2181 | Kafka coordination |
| **Faust worker** | (no port) | Consumes Kafka, pushes real-time features to Feast; internal service. |
| **Parquet sink** | (no port) | Consumes Kafka, writes Parquet to MinIO `raw-events`; internal service. |

---

### Streaming Pipeline

The simulator (realtime mode) produces each transaction to **Kafka** as well as to the BentoML fraud API. Two services consume from Kafka:

- **Faust worker** — maintains per-user aggregates and writes to the Feast online store (Redis).
- **Parquet sink** — batches events and writes Parquet files to MinIO (`raw-events` bucket).

**Start only the streaming components:**

```bash
make stream-start
# or
docker compose up -d faust-worker parquet-sink
```

**View streaming logs:**

```bash
make stream-logs
# or
docker compose logs -f faust-worker parquet-sink
```

**Verify events are flowing:**

- MinIO console: http://localhost:9001 — check bucket `raw-events` for `transactions/dt=YYYY-MM-DD/*.parquet`.
- After running the simulator in realtime mode with Kafka reachable, the Faust worker will push features to Redis; the fraud service can use `user_realtime_fraud_features` from the online store.

**Full stack (including streaming):** `docker compose up -d --build` starts Kafka, Faust worker, and Parquet sink along with the rest. Ensure `kafka-init` has run (it creates the `raw.transactions` topic) and MinIO has the `raw-events` bucket (created by `minio-init`).

---

### Simulator

The **event ticketing simulator** generates realistic traffic (events, users, transactions, fraud patterns) to stress-test the platform under scenarios such as normal traffic, flash sales, fraud attacks, configurable drift, system degradation, and Black Friday. In **realtime mode**, the simulator also produces each transaction to Kafka (`raw.transactions`) so the Faust worker can compute real-time features and the Parquet sink can archive raw events to MinIO.

**Available scenarios:** `normal-traffic`, `flash-sale`, `fraud-attack`, `drift`, `system-degradation`, `black-friday`, `mix`.

#### Drift scenario (trigger model retraining)

The **drift** scenario generates seed-compatible data with a configurable drift level (0 = no drift, 1 = maximum drift). Use it to trigger the monitoring pipeline’s drift check (threshold 0.30) and model retraining:

```bash
# Default drift level 0.5 (moderate drift)
python -m simulator.cli run drift -o reports/drift.html

# No drift (same distribution as seed)
python -m simulator.cli run drift --drift-level 0 -o reports/drift-none.html

# Strong drift (guarantees retrain trigger)
python -m simulator.cli run drift --drift-level 0.9 -o reports/drift-strong.html

# Optional: override seed and data size
python -m simulator.cli run drift --drift-level 0.7 --drift-seed 42 --drift-transactions 3000 -o reports/drift-7.html
```

After running, the scenario report includes the computed drift score. The **feature_engineering_pipeline** compares a seed-derived reference (regenerated each run) to the last 24h of data; when you run the simulator then the pipeline, drift_score >= 0.30 triggers model training.

#### Run via CLI (from repo root, with venv active)

```bash
# List scenarios
python -m simulator.cli list-scenarios

# Run one scenario (output HTML report)
python -m simulator.cli run normal-traffic -o reports/normal-traffic-report.html
python -m simulator.cli run flash-sale -o reports/flash-sale-report.html

# Optional: override duration (minutes)
python -m simulator.cli run normal-traffic --duration 10 -o reports/normal-10min.html

# Dry run (setup only, no traffic)
python -m simulator.cli run normal-traffic --dry-run

# Run all scenarios
python -m simulator.cli run-all -o reports/

# Mix mode: weighted scenarios for N minutes
python -m simulator.cli mix --duration 30 --scenarios "normal-traffic:40,flash-sale:20,fraud-attack:20,system-degradation:10,black-friday:10" -o reports/mix-report.html

# Realtime: stream at fixed RPS for wall-clock seconds
python -m simulator.cli realtime normal-traffic --duration 60 --rps 100 -o reports/realtime-report.html
```

#### Run via Make

Requires `pip install -e .` and `make` (e.g. WSL / Git Bash):

```bash
make sim-list
make sim-run scenario=normal-traffic   # writes reports/normal-traffic-report.html
make sim-run scenario=flash-sale
make sim-run scenario=drift             # drift (default level 0.5); for strong drift run CLI with --drift-level 0.9
make sim-run-all                       # runs all, output to reports/
make sim-mix                           # mix mode (default 30 min); use duration=10 for 10 min
make sim-realtime                      # realtime 60s @ 100 rps; use scenario= duration= rps= to override
```

#### Run via Docker Compose

Start the main stack and the simulator overlay, then run the simulator in a one-off container:

```bash
docker compose -f docker-compose.yml -f docker-compose.simulator.yml up -d
docker compose -f docker-compose.yml -f docker-compose.simulator.yml run --rm simulator python -m simulator.cli run normal-traffic -o /app/reports/out.html
```

Reports are written to the `reports/` directory (or `/app/reports/` inside the container; the compose file mounts `./reports` there).

To run against the live fraud API, ensure `bentoml-fraud` is up and set `API_BASE_URL` (e.g. in the simulator service env: `http://bentoml-fraud:7001`). The simulator can also run offline with synthetic responses.


### Feature store and sample data

1. **Generate synthetic Feast data:**

   ```bash
   # PowerShell
   .venv\Scripts\python scripts/seed-data.py
   # Make
   make seed-data
   ```

   Creates `data/processed/feast/event_metrics.parquet` and `user_metrics.parquet`.

2. **Apply Feast feature repo:**

   ```bash
   feast -c services/feast/feature_repo apply
   # or
   make feast-apply
   ```

3. **Fetch online features** (with Redis and stack running):

   ```python
   from common.feature_utils import fetch_online_features
   rows = [{"event_id": 1}]
   features = ["event_realtime_metrics:current_inventory"]
   df = fetch_online_features(features=features, entity_rows=rows)
   ```

---

### MLflow and fraud model training

1. Start the stack (`docker compose up -d`). MLflow UI: http://localhost:5000.
2. Open Jupyter at http://localhost:8888 and run `notebooks/03_model_training_fraud.ipynb` top to bottom. The notebook builds a fraud training dataset, runs XGBoost + Optuna training, and logs runs and artifacts to MLflow.

---

### Model serving (BentoML)

Fraud and pricing services start with the stack. To start only BentoML:

```bash
make serve-fraud    # http://localhost:7001
make serve-pricing  # http://localhost:7002
make serve-bento   # both
```

**Health checks:**

```bash
curl -X POST http://localhost:7001/healthz
curl -X POST http://localhost:7002/healthz
```

**Checking BentoML fraud service:**

- **Docker:** `docker compose ps` — look for `bentoml-fraud` or `ibook-bentoml-fraud` (port 7001). If the stack is up, the fraud service starts with it.
- **Health:**  
  ```bash
  curl -X POST http://localhost:7001/healthz -H "Content-Type: application/json" -d "{}"
  ```  
  Expect `{"ok": true, "detail": "model_loaded"}` when the service and model are OK.

**Example requests (PowerShell):**

```bash
curl -X POST http://localhost:7001/predict -H "Content-Type: application/json" -d "{\"requests\":[{\"user_id\":1,\"event_id\":2,\"amount\":100.0}]}"
curl -X POST http://localhost:7002/recommend -H "Content-Type: application/json" -d "{\"requests\":[{\"event_id\":1,\"current_price\":100.0}]}"
```

---

### Airflow workflows

With the stack running, open http://localhost:8080 (admin/admin). Three DAGs are under `services/airflow/dags/`:

- **feature_engineering_pipeline** — hourly: read raw events from MinIO (`raw-events` bucket), compute user_realtime_fraud_features (same logic as Faust) and event aggregates, validate, materialize to Feast (including user_realtime_fraud_features when data exists), check drift.
- **model_training_pipeline** — weekly: build dataset, train XGBoost, evaluate, register in MLflow, promote, then notify BentoML to reload.
- **ml_monitoring_pipeline** — daily: collect metrics, compute drift, check thresholds, alert/retrain stubs.

Unpause each DAG and trigger a run from the UI. Kafka/Evidently/Slack integrations are stubbed for local use. The model_training_pipeline uses `BENTOML_BASE_URL` (or `FRAUD_API_BASE_URL`) to reach the fraud service for `/admin/reload` after promotion.

**Note:** The project root is mounted into the Airflow containers as `/opt/airflow/workspace` and `PYTHONPATH` is set to that path so the DAGs can import the `common` package (e.g. `common.feature_utils`, `common.model_utils`). DAGs resolve `data/` paths from the workspace, so parquet files under `data/processed/feast/` and `data/monitoring/` are used when present. Task retries are configurable via env: `AIRFLOW_TASK_RETRIES` (default 3) and per-DAG overrides `FEATURE_ENGINEERING_PIPELINE_RETRIES`, `MODEL_TRAINING_PIPELINE_RETRIES`, `ML_MONITORING_PIPELINE_RETRIES` (set to 0 to fail immediately). See `.env.example`.

---

### Monitoring (Prometheus & Grafana)

- **Prometheus:** http://localhost:9090 — metrics and alert rules (Alerts / Status → Rules).
- **Grafana:** http://localhost:3000 — open **Dashboards** → **MLOps Overview** (provisioned from `services/monitoring/grafana/dashboards/`).

BentoML fraud (7001) and pricing (7002) services expose `/metrics`; Prometheus scrapes them. For how to observe drift, retraining, fraud, and degradation, see [Observability.md](Observability.md).
---

### Run tests

No Docker required; use the project venv.

**PowerShell:**

```bash
.venv\Scripts\python -m pytest tests\ -v --tb=short
```

**Make:**

```bash
make test
```

Notable test modules: `tests/unit/test_models.py` (fraud training + MLflow), `tests/unit/test_bentoml_common.py`, `tests/integration/test_api_endpoints.py`, `tests/unit/test_airflow_dags.py`, `tests/unit/test_monitoring_utils.py`. Run a subset, e.g.:

```bash
pytest tests/unit/test_airflow_dags.py tests/unit/test_monitoring_utils.py -v --tb=short
```

---

### Troubleshooting

- **Ports in use:** Stop conflicting services or change ports in `docker-compose.yml` / `.env`.
- **Full reset:**

  ```bash
  docker compose down -v
  docker compose up -d --build
  ```

For more detail on architecture, algorithms, and observability, see **how_it_works.md** and **Observability.md**.
