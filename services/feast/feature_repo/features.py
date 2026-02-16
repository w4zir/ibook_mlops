from __future__ import annotations

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64, String

from common.config import get_config
from services.feast.feature_repo.data_sources import (
    get_event_metrics_source,
    get_user_metrics_source,
    get_user_realtime_source,
)


_cfg = get_config()

# Entities
event = Entity(name="event", join_keys=["event_id"])
user = Entity(name="user", join_keys=["user_id"])
promoter = Entity(name="promoter", join_keys=["promoter_id"])


_event_source = get_event_metrics_source(_cfg)
_user_source = get_user_metrics_source(_cfg)
_user_realtime_source = get_user_realtime_source(_cfg)


event_realtime_metrics = FeatureView(
    name="event_realtime_metrics",
    entities=[event],
    ttl=timedelta(minutes=60),
    schema=[
        Field(name="current_inventory", dtype=Int64),
        Field(name="sell_through_rate_5min", dtype=Float32),
        Field(name="concurrent_viewers", dtype=Int64),
    ],
    online=True,
    source=_event_source,
)


event_historical_metrics = FeatureView(
    name="event_historical_metrics",
    entities=[event, promoter],
    ttl=timedelta(days=365),
    schema=[
        Field(name="total_tickets_sold", dtype=Int64),
        Field(name="avg_ticket_price", dtype=Float32),
        Field(name="promoter_success_rate", dtype=Float32),
    ],
    online=True,
    source=_event_source,
)


user_purchase_behavior = FeatureView(
    name="user_purchase_behavior",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="lifetime_purchases", dtype=Int64),
        Field(name="fraud_risk_score", dtype=Float32),
        Field(name="preferred_category", dtype=String),
    ],
    online=True,
    source=_user_source,
)


user_realtime_fraud_features = FeatureView(
    name="user_realtime_fraud_features",
    entities=[user],
    ttl=timedelta(hours=2),
    schema=[
        Field(name="user_txn_count_1h", dtype=Int64),
        Field(name="user_txn_amount_1h", dtype=Float32),
        Field(name="user_distinct_events_1h", dtype=Int64),
        Field(name="user_avg_amount_24h", dtype=Float32),
    ],
    online=True,
    source=_user_realtime_source,
)


# Backward compatibility: newer Feast stores the datasource on `batch_source`
# while older code/tests expect `.source`.
for _fv in (
    event_realtime_metrics,
    event_historical_metrics,
    user_purchase_behavior,
    user_realtime_fraud_features,
):
    if not hasattr(_fv, "source"):
        _fv.source = _fv.batch_source


__all__ = [
    "event",
    "user",
    "promoter",
    "event_realtime_metrics",
    "event_historical_metrics",
    "user_purchase_behavior",
    "user_realtime_fraud_features",
]

