"""
Faust application: consumes raw.transactions, maintains per-user aggregates, pushes to Feast online store.
"""

from __future__ import annotations

import os

from faust import App

from services.faust_worker.models import RawTransaction

BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
RAW_TOPIC_NAME = os.getenv("KAFKA_RAW_TOPIC", "raw.transactions")

app = App(
    id="ibook-faust-worker",
    broker=BROKER,
    store="memory://",
    topic_partitions=3,
)

raw_transactions_topic = app.topic(RAW_TOPIC_NAME, value_type=RawTransaction, key_type=str)

# Import agents so they are registered
from services.faust_worker import agents  # noqa: E402, F401

__all__ = ["app", "raw_transactions_topic"]
