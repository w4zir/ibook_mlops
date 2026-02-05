from __future__ import annotations

"""
BentoML service entrypoint for fraud detection.

The BentoML-specific decorators are intentionally kept thin; the core request
handling logic is implemented in pure Python functions so it can be exercised
directly from unit tests without running a HTTP server.
"""

from typing import List

try:  # pragma: no cover - import guard for environments without BentoML
    import bentoml
    from bentoml.io import JSON
except Exception:  # pragma: no cover
    bentoml = None  # type: ignore[assignment]
    JSON = None  # type: ignore[assignment]

import mlflow.pyfunc
import pandas as pd

from services.bentoml.common.config import get_bentoml_settings
from services.bentoml.common.feast_client import get_fraud_features_for_entities
from services.bentoml.common.metrics import record_error, record_request, track_latency
from services.bentoml.common.mlflow_client import ResolvedModel, resolve_latest_model
from services.bentoml.services.fraud_detection.model import (
    FraudRequest,
    FraudResponse,
    FraudBatchRequest,
    FraudBatchResponse,
    FraudModelRuntime,
)


SERVICE_NAME = "fraud_detection"


def _load_model() -> FraudModelRuntime:
    settings = get_bentoml_settings()
    resolved: ResolvedModel = resolve_latest_model(settings=settings)
    model = mlflow.pyfunc.load_model(resolved.model_uri)
    # Threshold can later be made configurable if needed.
    return FraudModelRuntime(model=model, threshold=0.5)


_RUNTIME: FraudModelRuntime | None = None


def _get_runtime() -> FraudModelRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _load_model()
    return _RUNTIME


def _build_feature_frame(batch: FraudBatchRequest) -> pd.DataFrame:
    """
    Build the feature DataFrame for a batch of fraud requests.

    For now this uses Feast to look up features by (user_id, event_id) and
    applies any per-request `feature_overrides` if present.
    """
    entity_rows = [{"user_id": r.user_id, "event_id": r.event_id} for r in batch.requests]
    features_df = get_fraud_features_for_entities(entity_rows)

    # Apply overrides column-wise.
    for idx, req in enumerate(batch.requests):
        if not req.feature_overrides:
            continue
        for name, value in req.feature_overrides.items():
            features_df.loc[features_df.index[idx], name] = value

    return features_df


def handle_predict(batch: FraudBatchRequest) -> FraudBatchResponse:
    """
    Core prediction logic, independent of BentoML's IO system.
    """
    runtime = _get_runtime()
    features_df = _build_feature_frame(batch)
    predictions = runtime.predict_batch(features_df)
    return FraudBatchResponse(predictions=predictions)


def _health_payload(ok: bool, detail: str) -> dict:
    return {"ok": ok, "detail": detail}


def handle_healthcheck() -> dict:
    """
    Simple healthcheck: checks that the model can be loaded.
    """
    try:
        _ = _get_runtime()
        return _health_payload(True, "model_loaded")
    except Exception as exc:  # pragma: no cover - defensive
        return _health_payload(False, f"error: {exc}")


svc = None
if bentoml is not None and JSON is not None:  # pragma: no cover - HTTP layer
    # --- BentoML service definition -------------------------------------------
    svc = bentoml.Service(SERVICE_NAME)

    @svc.api(input=JSON(pydantic_model=FraudBatchRequest), output=JSON(pydantic_model=FraudBatchResponse))
    def predict(batch: FraudBatchRequest) -> FraudBatchResponse:
        endpoint = "/predict"
        with track_latency(SERVICE_NAME, endpoint):
            try:
                response = handle_predict(batch)
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

