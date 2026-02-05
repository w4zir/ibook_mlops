from __future__ import annotations

import types
from typing import Any, Dict

import pandas as pd
import pytest

from services.bentoml.common import config as bentoml_config
from services.bentoml.common import feast_client, mlflow_client, metrics
from services.bentoml.services.fraud_detection import model as fraud_model
from services.bentoml.services.dynamic_pricing import model as pricing_model


def test_get_bentoml_settings_uses_app_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force specific MLflow tracking URI via env and ensure it flows through.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-test:5000")
    monkeypatch.setenv("BENTOML_MODEL_NAME", "fraud_detection_test")
    monkeypatch.setenv("BENTOML_MODEL_STAGE", "Staging")

    bentoml_config.get_bentoml_settings.cache_clear()
    settings = bentoml_config.get_bentoml_settings()

    assert settings.tracking_uri == "http://mlflow-test:5000"
    assert settings.model_name == "fraud_detection_test"
    assert settings.model_stage == "Staging"


def test_mlflow_resolve_latest_model_prefers_highest_version(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyVersion:
        def __init__(self, version: str, stage: str) -> None:
            self.version = version
            self.current_stage = stage
            self.run_id = f"run-{version}"
            self.source = f"s3://bucket/{version}"

    dummy_client = types.SimpleNamespace(
        search_model_versions=lambda _: [
            DummyVersion("1", "None"),
            DummyVersion("2", "Production"),
            DummyVersion("3", "None"),
        ]
    )

    def fake_client(_: Any) -> Any:
        return dummy_client

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-test:5000")
    bentoml_config.get_bentoml_settings.cache_clear()

    monkeypatch.setattr(mlflow_client, "_get_client", fake_client)

    resolved = mlflow_client.resolve_latest_model(model_name="fraud_detection_test", stage="Production")
    assert resolved.version == "2"
    assert resolved.stage == "Production"
    assert resolved.model_uri.endswith("/2")


def test_feast_client_wrappers_forward_to_feature_utils(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Dict[str, Any] = {}

    def fake_fetch_online_features(features, entity_rows, store=None, use_cache=True):  # type: ignore[override]
        calls["features"] = list(features)
        calls["entity_rows"] = list(entity_rows)
        return pd.DataFrame({"dummy": [1] * len(entity_rows)})

    monkeypatch.setattr("common.feature_utils.fetch_online_features", fake_fetch_online_features)

    df = feast_client.get_fraud_features_for_entities([{"user_id": 1, "event_id": 2}])
    assert "dummy" in df.columns
    assert calls["features"] == feast_client.DEFAULT_FRAUD_FEATURES
    assert calls["entity_rows"][0]["user_id"] == 1


def test_metrics_helpers_increment_counters() -> None:
    # Smoke-test that the helpers can be called without raising and that the
    # underlying Prometheus collectors accept the labels.
    metrics.record_request("svc", "/x", 200)
    metrics.record_error("svc", "/x")
    metrics.observe_latency("svc", "/x", 0.01)


def test_fraud_model_runtime_predict_batch() -> None:
    class DummyModel:
        def predict_proba(self, X):
            import numpy as np

            return np.column_stack([1 - np.asarray(X)[:, 0], np.asarray(X)[:, 0]])

    runtime = fraud_model.FraudModelRuntime(model=DummyModel(), threshold=0.5)
    df = pd.DataFrame({"x": [0.2, 0.8]})
    preds = runtime.predict_batch(df)
    assert preds[0].is_fraud is False
    assert preds[1].is_fraud is True


def test_pricing_bandit_updates_and_chooses() -> None:
    bandit = pricing_model.ThompsonSamplingBandit()
    # Update the aggressive arm with strong rewards.
    for _ in range(10):
        bandit.update("aggressive", reward=1.0)
    # Over many samples, aggressive should win more often.
    wins = {"baseline": 0, "aggressive": 0}
    for _ in range(100):
        wins[bandit.choose_arm()] += 1
    assert wins["aggressive"] > wins["baseline"]

