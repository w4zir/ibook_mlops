"""End-to-end pipeline test: data -> training -> serving -> monitoring."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("mlflow")

from common.model_utils import (
    TrainingResult,
    build_fraud_training_dataframe,
    train_fraud_model,
)
from common.monitoring_utils import check_alert_thresholds, generate_drift_report
from scripts.seed_data import generate_synthetic_data
from services.bentoml.services.dynamic_pricing import service as pricing_service
from services.bentoml.services.fraud_detection import service as fraud_service


@pytest.fixture(autouse=True)
def _patch_feast_and_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fraud_features(entity_rows, feature_refs=None):
        return pd.DataFrame(
            {
                "user_purchase_behavior__lifetime_purchases": [10] * len(entity_rows),
                "user_purchase_behavior__fraud_risk_score": [0.3] * len(entity_rows),
            }
        )

    def fake_pricing_features(entity_rows, feature_refs=None):
        return pd.DataFrame(
            {
                "event_realtime_metrics__current_inventory": [100] * len(entity_rows),
                "event_realtime_metrics__sell_through_rate_5min": [1.0] * len(entity_rows),
            }
        )

    monkeypatch.setattr(
        "services.bentoml.common.feast_client.get_fraud_features_for_entities",
        fake_fraud_features,
    )
    monkeypatch.setattr(
        "services.bentoml.services.fraud_detection.service.get_fraud_features_for_entities",
        fake_fraud_features,
    )
    monkeypatch.setattr(
        "services.bentoml.common.feast_client.get_pricing_features_for_entities",
        fake_pricing_features,
    )
    monkeypatch.setattr(
        "services.bentoml.services.dynamic_pricing.service.get_pricing_features_for_entities",
        fake_pricing_features,
    )

    class DummyPyfuncModel:
        def predict(self, X):
            import numpy as np
            return np.full(shape=(len(X),), fill_value=0.7)

        def predict_proba(self, X):
            import numpy as np
            scores = np.full(shape=(len(X),), fill_value=0.7)
            return np.column_stack([1 - scores, scores])

    def fake_resolve(*_: Any, **__: Any):
        return types.SimpleNamespace(model_uri="models:/fraud_detection/1")

    def fake_load(*_: Any, **__: Any):
        return DummyPyfuncModel()

    monkeypatch.setattr(
        "services.bentoml.services.fraud_detection.service.resolve_latest_model",
        fake_resolve,
    )
    monkeypatch.setattr("mlflow.pyfunc.load_model", fake_load)


def test_e2e_data_generation_to_training(minimal_local_env) -> None:
    """Generate synthetic data and build training dataframe."""
    event_metrics, user_metrics = generate_synthetic_data(
        n_events=30, n_users=100, n_transactions=500, seed=42
    )
    assert len(event_metrics) > 0
    assert len(user_metrics) > 0
    assert "fraud_risk_score" in user_metrics.columns
    assert "lifetime_purchases" in user_metrics.columns

    train_df = build_fraud_training_dataframe(user_metrics, fraud_threshold=0.05)
    assert "is_fraud_label" in train_df.columns
    assert len(train_df) == len(user_metrics)


def test_e2e_training_to_mlflow(tmp_path: Path, minimal_local_env) -> None:
    """Train model and verify MLflow logging."""
    _, user_metrics = generate_synthetic_data(
        n_events=20, n_users=80, n_transactions=300, seed=7
    )
    train_df = build_fraud_training_dataframe(user_metrics, fraud_threshold=0.08)
    tracking_uri = (tmp_path / "mlruns").as_uri()

    result: TrainingResult = train_fraud_model(
        df=train_df,
        target_column="is_fraud_label",
        tracking_uri=tracking_uri,
        n_trials=2,
        test_size=0.25,
        random_state=7,
    )
    assert result.roc_auc >= 0 and result.roc_auc <= 1
    assert result.run_id


def test_e2e_serving_fraud_and_pricing(minimal_local_env) -> None:
    """Run fraud and pricing prediction after pipeline."""
    batch_fraud = fraud_service.FraudBatchRequest(
        requests=[fraud_service.FraudRequest(user_id=1, event_id=2, amount=100.0)]
    )
    resp_fraud = fraud_service.handle_predict(batch_fraud)
    assert len(resp_fraud.predictions) == 1
    assert 0 <= resp_fraud.predictions[0].fraud_score <= 1

    batch_pricing = pricing_service.PricingBatchRequest(
        requests=[pricing_service.PricingRequest(event_id=1, current_price=100.0)]
    )
    resp_pricing = pricing_service.handle_pricing(batch_pricing)
    assert len(resp_pricing.predictions) == 1
    assert resp_pricing.predictions[0].recommended_price > 0


def test_e2e_monitoring_drift_and_alerts(minimal_local_env) -> None:
    """Generate drift report and check alert thresholds."""
    _, user_metrics = generate_synthetic_data(
        n_events=15, n_users=50, n_transactions=200, seed=11
    )
    ref_df = user_metrics[["lifetime_purchases", "fraud_risk_score"]].head(30)
    cur_df = user_metrics[["lifetime_purchases", "fraud_risk_score"]].tail(20)

    drift_result = generate_drift_report(ref_df, cur_df)
    assert hasattr(drift_result, "drift_score")
    assert 0 <= drift_result.drift_score <= 1

    alerted = check_alert_thresholds(drift_result, drift_threshold=0.30)
    assert isinstance(alerted, bool)
