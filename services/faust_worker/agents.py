"""
Faust agents: consume raw.transactions, maintain per-user aggregates, push to Feast online store.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from services.faust_worker.app import app, raw_transactions_topic
from services.faust_worker.models import UserAggregate

logger = logging.getLogger(__name__)

# Per-user state: count, amount_sum, list of event_ids (for distinct count)
# We cap event_ids to avoid unbounded growth; aggregates still reflect all seen counts/amounts
MAX_EVENT_IDS_PER_USER = 5000

user_aggregates = app.Table(
    "user_aggregates",
    default=lambda: UserAggregate(count=0, amount_sum=0.0, event_ids=[]),
    key_type=int,
    value_type=UserAggregate,
)


def _push_to_feast(df: pd.DataFrame) -> None:
    """Write aggregated features to Feast online store (Redis)."""
    if df.empty:
        return
    try:
        # Resolve repo path: same as common/feature_utils (workspace root when run in container)
        repo_path = os.getenv("FEAST_REPO_PATH")
        if not repo_path:
            for base in [Path.cwd(), Path("/opt/airflow/workspace"), Path("/app")]:
                candidate = base / "services" / "feast" / "feature_repo"
                if candidate.exists():
                    repo_path = str(candidate)
                    break
        if not repo_path:
            repo_path = "services/feast/feature_repo"
        from feast import FeatureStore

        fs = FeatureStore(repo_path=repo_path)
        redis_host = os.getenv("REDIS_HOST")
        redis_port = os.getenv("REDIS_PORT")
        if redis_host or redis_port:
            host = redis_host or "localhost"
            port = redis_port or "6379"
            if hasattr(fs.config, "online_store") and hasattr(fs.config.online_store, "connection_string"):
                fs.config.online_store.connection_string = f"{host}:{port}"
        fs.write_to_online_store("user_realtime_fraud_features", df=df)
        logger.info("Pushed %d user rows to Feast online store (user_realtime_fraud_features).", len(df))
    except Exception as e:
        logger.exception("Failed to push to Feast: %s", e)


# Push to Feast every N seconds for all users we've seen
PUSH_INTERVAL_SEC = float(os.getenv("FAUST_PUSH_INTERVAL_SEC", "30.0"))
_last_push_time = 0.0


@app.agent(raw_transactions_topic)
async def process_transactions(stream):
    global _last_push_time
    async for event in stream:
        user_id = getattr(event, "user_id", None)
        if user_id is None:
            continue
        event_id = getattr(event, "event_id", 0)
        total_amount = float(getattr(event, "total_amount", 0.0))
        agg = user_aggregates[user_id]
        if not agg:
            agg = UserAggregate(count=0, amount_sum=0.0, event_ids=[])
        count = agg.count + 1
        amount_sum = agg.amount_sum + total_amount
        eids = list(agg.event_ids or [])
        eids.append(event_id)
        if len(eids) > MAX_EVENT_IDS_PER_USER:
            eids = eids[-MAX_EVENT_IDS_PER_USER:]
        user_aggregates[user_id] = UserAggregate(count=count, amount_sum=amount_sum, event_ids=eids)

        now = time.monotonic()
        if now - _last_push_time >= PUSH_INTERVAL_SEC:
            _last_push_time = now
            rows = []
            for uid, data in user_aggregates.items():
                if not data or data.count == 0:
                    continue
                count = data.count
                amount_sum = data.amount_sum
                event_ids = data.event_ids or []
                distinct_events = len(set(event_ids))
                avg_amount = amount_sum / count if count else 0.0
                rows.append({
                    "user_id": uid,
                    "event_timestamp": datetime.now(timezone.utc),
                    "user_txn_count_1h": count,
                    "user_txn_amount_1h": amount_sum,
                    "user_distinct_events_1h": distinct_events,
                    "user_avg_amount_24h": avg_amount,
                })
            if rows:
                df = pd.DataFrame(rows)
                _push_to_feast(df)
