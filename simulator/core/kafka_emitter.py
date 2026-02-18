"""
Reusable Kafka emission for simulator scenarios.

Ensures generated transactions are published to raw.transactions with a
schema compatible with the Parquet sink and feature pipeline (user_id,
event_id, total_amount, timestamp, is_fraud, plus optional ingestion metadata).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Required fields for downstream (sink + feature_engineering_pipeline).
_REQUIRED_KEYS = ("user_id", "event_id", "total_amount", "timestamp")


def prepare_transaction_for_kafka(
    txn: Dict[str, Any],
    scenario_tag: str | None = None,
) -> Dict[str, Any]:
    """
    Return a Kafka-compatible payload from a simulator transaction.

    Ensures required keys exist and optionally adds ingestion metadata.
    Does not mutate the input dict.
    """
    out: Dict[str, Any] = dict(txn)
    if scenario_tag is not None:
        out["_scenario"] = scenario_tag
    return out


def emit_transactions_to_kafka(
    transactions: List[Dict[str, Any]],
    scenario_tag: str | None = None,
) -> int:
    """
    Publish a list of transactions to the raw.transactions topic.

    Best-effort: logs and continues on per-message failures. Returns the
    number of messages successfully produced (or attempted; we do not
    wait for delivery confirmation).
    """
    try:
        from services.kafka.producer import send_raw_transaction
    except ImportError as e:
        logger.warning("Kafka producer not available; skipping emit: %s", e)
        return 0

    sent = 0
    for txn in transactions:
        try:
            payload = prepare_transaction_for_kafka(txn, scenario_tag=scenario_tag)
            # Ensure required keys for sink/feature pipeline
            for key in _REQUIRED_KEYS:
                if key not in payload:
                    logger.debug("Transaction missing %s; skipping", key)
                    break
            else:
                send_raw_transaction(payload)
                sent += 1
        except Exception as e:
            logger.debug("Failed to send transaction to Kafka: %s", e)
    if transactions and sent > 0:
        logger.info(
            "Emitted %d/%d transactions to Kafka (scenario_tag=%s)",
            sent,
            len(transactions),
            scenario_tag,
        )
        try:
            from services.kafka.producer import flush_producer
            flush_producer(timeout_sec=10.0)
        except Exception as e:
            logger.debug("Flush after batch emit: %s", e)
    return sent
