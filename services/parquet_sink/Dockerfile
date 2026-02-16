FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    confluent-kafka>=2.3.0 \
    pyarrow>=14.0.0 \
    boto3>=1.34.0

COPY services/parquet_sink/sink.py /app/sink.py

ENV PYTHONUNBUFFERED=1
CMD ["python", "-u", "/app/sink.py"]
