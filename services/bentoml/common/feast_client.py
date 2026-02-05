from __future__ import annotations

"""
Thin wrappers around `common.feature_utils` for online feature access from
BentoML services.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from common import feature_utils

# Default feature references aligned with PLAN.md Phase 2 definitions.
DEFAULT_FRAUD_FEATURES: List[str] = [
    "user_purchase_behavior:lifetime_purchases",
    "user_purchase_behavior:fraud_risk_score",
]

DEFAULT_PRICING_FEATURES: List[str] = [
    "event_realtime_metrics:current_inventory",
    "event_realtime_metrics:sell_through_rate_5min",
    "event_historical_metrics:avg_ticket_price",
]


def _ensure_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def get_fraud_features_for_entities(
    entity_rows: Sequence[Mapping[str, Any]],
    *,
    feature_refs: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Fetch fraud-related online features from Feast for the given entities.
    """
    refs = list(feature_refs) if feature_refs is not None else DEFAULT_FRAUD_FEATURES
    return feature_utils.fetch_online_features(features=refs, entity_rows=_ensure_rows(entity_rows))


def get_pricing_features_for_entities(
    entity_rows: Sequence[Mapping[str, Any]],
    *,
    feature_refs: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Fetch dynamic pricing-related features (inventory, velocity, etc.) from Feast.
    """
    refs = list(feature_refs) if feature_refs is not None else DEFAULT_PRICING_FEATURES
    return feature_utils.fetch_online_features(features=refs, entity_rows=_ensure_rows(entity_rows))


__all__ = [
    "DEFAULT_FRAUD_FEATURES",
    "DEFAULT_PRICING_FEATURES",
    "get_fraud_features_for_entities",
    "get_pricing_features_for_entities",
]

