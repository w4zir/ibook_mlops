FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies: curl for container health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install MLflow and minimal DB/storage drivers.
RUN pip install --no-cache-dir \
    "mlflow>=2.11,<2.12" \
    psycopg2-binary==2.9.9 \
    boto3==1.34.34

# Copy lightweight health server.
COPY services/mlflow/health_server.py /app/health_server.py

EXPOSE 5000 5001

# Expected env vars (set by docker-compose / Kubernetes):
# - MLFLOW_BACKEND_STORE_URI (e.g. postgresql+psycopg2://user:pass@host:5432/mlflow)
# - MLFLOW_DEFAULT_ARTIFACT_ROOT (e.g. s3://mlflow/)
# - Optional HEALTH_BASIC_AUTH_USER / HEALTH_BASIC_AUTH_PASSWORD
# - Optional MLFLOW_HEALTH_TARGET (default: http://localhost:5000/)

# Start MLflow server and the health probe sidecar in the same container.
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --default-artifact-root \"$MLFLOW_DEFAULT_ARTIFACT_ROOT\" & python /app/health_server.py"]

