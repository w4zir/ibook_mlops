"""
Parquet sink: consume raw.transactions from Kafka, batch events, write Parquet to MinIO.

Path layout: s3://raw-events/transactions/dt=YYYY-MM-DD/batch_{timestamp}.parquet
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from confluent_kafka import Consumer, KafkaError, KafkaException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Config from env (container-friendly)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_RAW_TOPIC", "raw.transactions")
KAFKA_GROUP = os.getenv("KAFKA_GROUP_ID", "parquet-sink")
BATCH_SIZE = int(os.getenv("PARQUET_SINK_BATCH_SIZE", "500"))
FLUSH_INTERVAL_SEC = float(os.getenv("PARQUET_SINK_FLUSH_INTERVAL_SEC", "30.0"))
RAW_EVENTS_BUCKET = os.getenv("RAW_EVENTS_BUCKET", "raw-events")


def _upload_parquet_to_minio(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    try:
        import io

        import boto3
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Infer schema from first row; use string for complex types
        table = pa.Table.from_pylist(rows)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)

        endpoint = os.getenv("AWS_S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"))
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        # Partition by date from first event timestamp
        ts_str = rows[0].get("timestamp") or datetime.now(timezone.utc).isoformat()
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).date()
        except Exception:
            dt = datetime.now(timezone.utc).date()
        date_prefix = dt.isoformat()
        key = f"transactions/dt={date_prefix}/batch_{int(time.time() * 1000)}.parquet"
        client.upload_fileobj(buf, RAW_EVENTS_BUCKET, key)
        logger.info("Uploaded %d events to s3://%s/%s", len(rows), RAW_EVENTS_BUCKET, key)
    except Exception as e:
        logger.exception("Failed to upload Parquet to MinIO: %s", e)
        raise


def main() -> None:
    conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": KAFKA_GROUP,
        "auto.offset.reset": "earliest",
    }
    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_TOPIC])

    batch: List[Dict[str, Any]] = []
    last_flush = time.monotonic()
    running = True

    def shutdown(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(
        "Parquet sink started (topic=%s, batch_size=%s, flush_interval_sec=%s)",
        KAFKA_TOPIC,
        BATCH_SIZE,
        FLUSH_INTERVAL_SEC,
    )

    while running:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            if batch and (time.monotonic() - last_flush) >= FLUSH_INTERVAL_SEC:
                _upload_parquet_to_minio(batch)
                batch = []
                last_flush = time.monotonic()
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            raise KafkaException(msg.error())
        try:
            payload = json.loads(msg.value().decode("utf-8"))
            batch.append(payload)
        except Exception as e:
            logger.warning("Skip invalid message: %s", e)
            continue
        if len(batch) >= BATCH_SIZE:
            _upload_parquet_to_minio(batch)
            batch = []
            last_flush = time.monotonic()

    if batch:
        _upload_parquet_to_minio(batch)
    consumer.close()
    logger.info("Parquet sink stopped.")


if __name__ == "__main__":
    main()
