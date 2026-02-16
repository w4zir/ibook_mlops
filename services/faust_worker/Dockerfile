FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "faust-streaming[rocksdb]>=0.11.0" \
    "feast[redis]>=0.40.0" \
    pandas

# Copy minimal surface: feature repo for Feast, faust worker, common config
COPY common /app/common
COPY services/feast /app/services/feast
COPY services/faust_worker /app/services/faust_worker

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["faust", "-A", "services.faust_worker.app", "worker", "-l", "info"]
