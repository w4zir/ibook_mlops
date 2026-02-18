from __future__ import annotations

"""
Thin Kafka producer wrapper used by the simulator.

We keep this module intentionally small so it can be reused from
CLI tools or other services if needed.
"""

import json
import logging
from typing import Any, Dict, Optional

from confluent_kafka import Producer

from common.config import get_config


logger = logging.getLogger(__name__)


_producer: Optional[Producer] = None


def _get_producer() -> Producer:
    """
    Lazily construct a global Kafka producer.

    Configuration is derived from AppConfig.kafka; by default this will
    point at localhost:9092 when running on the host, and kafka:29092
    when running inside Docker (via KAFKA_BOOTSTRAP_SERVERS).
    """
    global _producer
    if _producer is not None:
        return _producer

    cfg = get_config()
    bootstrap_servers = cfg.kafka.bootstrap_servers
    logger.info("Initialising Kafka producer (bootstrap_servers=%s)", bootstrap_servers)
    _producer = Producer({"bootstrap.servers": bootstrap_servers})
    return _producer


def send_raw_transaction(event: Dict[str, Any]) -> None:
    """
    Produce a single raw transaction event to the raw.transactions topic.

    - Keyed by user_id (when present) to keep per-user ordering.
    - Value is a compact JSON-encoded dict.
    - Any serialization / network errors are logged and swallowed so the
      simulator can continue to make progress.
    """
    cfg = get_config()
    topic = cfg.kafka.raw_transactions_topic
    producer = _get_producer()

    try:
        key = None
        user_id = event.get("user_id")
        if user_id is not None:
            key = str(user_id).encode("utf-8")

        value = json.dumps(event, default=str).encode("utf-8")
        producer.produce(topic, key=key, value=value)
        # Trigger delivery callbacks; we do not block on full flush.
        producer.poll(0)
    except Exception as exc:  # pragma: no cover - best-effort logging
        logger.warning("Failed to send event to Kafka (topic=%s): %s", topic, exc)


def flush_producer(timeout_sec: float = 10.0) -> None:
    """
    Flush outstanding producer messages. Use after batch emission so messages
    are delivered before the process exits (e.g. simulator scenario run).
    """
    global _producer
    if _producer is None:
        return
    try:
        remaining = _producer.flush(timeout=timeout_sec)
        if remaining > 0:
            logger.warning("Kafka producer flush timed out; %d messages may be lost", remaining)
    except Exception as exc:  # pragma: no cover
        logger.warning("Kafka producer flush failed: %s", exc)

