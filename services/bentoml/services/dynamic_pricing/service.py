from __future__ import annotations

"""
BentoML service entrypoint for dynamic pricing.
"""

from typing import List

try:  # pragma: no cover - import guard for environments without BentoML
    import bentoml
    from bentoml.io import JSON
except Exception:  # pragma: no cover
    bentoml = None  # type: ignore[assignment]
    JSON = None  # type: ignore[assignment]
import numpy as np
import pandas as pd

from services.bentoml.common.feast_client import get_pricing_features_for_entities
from services.bentoml.common.metrics import record_error, record_request, track_latency
from services.bentoml.services.dynamic_pricing.model import (
    PricingBatchRequest,
    PricingBatchResponse,
    PricingRequest,
    PricingResponse,
    ThompsonSamplingBandit,
    rule_based_fallback,
)


SERVICE_NAME = "dynamic_pricing"


_BANDIT = ThompsonSamplingBandit()


def _build_pricing_frame(batch: PricingBatchRequest) -> pd.DataFrame:
    entity_rows = [{"event_id": r.event_id} for r in batch.requests]
    return get_pricing_features_for_entities(entity_rows)


def _choose_price(
    request: PricingRequest,
    features_row: pd.Series,
    arm: str,
) -> PricingResponse:
    """
    Derive a recommended price based on the chosen arm and features.
    """
    base_price = float(request.current_price)

    if arm == "aggressive":
        uplift = 0.1
    else:
        uplift = 0.0

    # A basic demand signal from features; default to neutral if missing.
    velocity = float(features_row.get("event_realtime_metrics__sell_through_rate_5min", 1.0))
    inventory = float(features_row.get("event_realtime_metrics__current_inventory", 1.0))
    demand_factor = np.clip(velocity / max(inventory, 1.0), 0.5, 2.0)

    recommended = base_price * (1.0 + uplift) * demand_factor
    recommended = float(max(recommended, 0.0))

    # Confidence interval is intentionally coarse; it can be refined later.
    lower = float(recommended * 0.9)
    upper = float(recommended * 1.1)

    return PricingResponse(
        recommended_price=recommended,
        arm=arm,
        lower_confidence=lower,
        upper_confidence=upper,
    )


def handle_pricing(batch: PricingBatchRequest) -> PricingBatchResponse:
    """
    Core pricing logic, independent of BentoML IO.
    """
    features_df = _build_pricing_frame(batch)
    responses: List[PricingResponse] = []

    for idx, req in enumerate(batch.requests):
        try:
            arm = req.arm_hint or _BANDIT.choose_arm()
            row = features_df.iloc[idx]
            resp = _choose_price(req, row, arm)
        except Exception:
            # Circuit breaker: use rule-based pricing if anything goes wrong.
            fallback_price = rule_based_fallback(req.current_price)
            resp = PricingResponse(
                recommended_price=fallback_price,
                arm="fallback",
                lower_confidence=fallback_price * 0.9,
                upper_confidence=fallback_price * 1.1,
            )
        responses.append(resp)

    return PricingBatchResponse(predictions=responses)


def handle_healthcheck() -> dict:
    # For now, if the process is alive and bandit is initialised, consider it healthy.
    return {"ok": True, "detail": "bandit_ready"}


svc = None
if bentoml is not None and JSON is not None:  # pragma: no cover - HTTP layer
    svc = bentoml.Service(SERVICE_NAME)

    @svc.api(input=JSON(pydantic_model=PricingBatchRequest), output=JSON(pydantic_model=PricingBatchResponse))
    def recommend(batch: PricingBatchRequest) -> PricingBatchResponse:
        endpoint = "/recommend"
        with track_latency(SERVICE_NAME, endpoint):
            try:
                response = handle_pricing(batch)
                record_request(SERVICE_NAME, endpoint, http_status=200)
                return response
            except Exception:
                record_error(SERVICE_NAME, endpoint)
                record_request(SERVICE_NAME, endpoint, http_status=500)
                raise

    @svc.api(input=JSON(), output=JSON())
    def health(_: dict) -> dict:
        endpoint = "/healthz"
        with track_latency(SERVICE_NAME, endpoint):
            payload = handle_healthcheck()
            status = 200 if payload.get("ok") else 500
            record_request(SERVICE_NAME, endpoint, http_status=status)
            return payload

