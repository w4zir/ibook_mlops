FROM python:3.11-slim

WORKDIR /app

# Pin aiokafka to 0.10.x: faust-streaming 0.11 expects aiokafka>=0.10; 0.11+ has API changes
RUN pip install --no-cache-dir \
    "aiokafka>=0.10.0,<0.11.0" \
    "faust-streaming[rocksdb]>=0.11.0" \
    "feast[redis,duckdb]>=0.40.0" \
    "duckdb>=0.10.0" \
    pandas

# Copy minimal surface: feature repo for Feast, faust worker, common config
COPY common /app/common
COPY services/feast /app/services/feast
COPY services/faust_worker /app/services/faust_worker

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["faust", "-A", "services.faust_worker.app", "worker", "-l", "info"]
