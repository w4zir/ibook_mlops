## How to run (Phases 1–3)

This repo currently covers **Phases 1–3**:

- Phase 1: local service stack + config module + basic unit tests.
- Phase 2: Feast feature repository + synthetic sample data + feature utilities.
- Phase 3: MLflow service hardening + fraud model training utilities and notebook.

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
- `.env.local` is committed with safe local defaults. Update values if ports or credentials conflict on your machine.

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

## Troubleshooting

- **Ports already in use**: stop conflicting local services or change the exposed ports in `docker-compose.yml`.
- **Reset everything**:

```bash
docker compose down -v
docker compose up -d --build
```

