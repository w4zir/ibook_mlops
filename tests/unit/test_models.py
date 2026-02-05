from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")

from common.model_utils import (  # noqa: E402
    TrainingResult,
    build_fraud_training_dataframe,
    train_fraud_model,
)
from scripts.seed_data import generate_synthetic_data  # noqa: E402


@pytest.fixture
def small_training_df() -> "tuple[object, object]":
    # Use a tiny synthetic dataset so tests remain fast.
    event_metrics, user_metrics = generate_synthetic_data(
        n_events=20, n_users=50, n_transactions=200, seed=123
    )
    train_df = build_fraud_training_dataframe(
        user_metrics,
        fraud_threshold=0.08,
    )
    return event_metrics, train_df


def test_build_fraud_training_dataframe_creates_label_column(small_training_df):
    _, train_df = small_training_df
    assert "is_fraud_label" in train_df.columns
    # Label should be binary.
    unique_labels = sorted(train_df["is_fraud_label"].unique().tolist())
    assert unique_labels[0] in (0, 1)
    assert unique_labels[-1] in (0, 1)


def test_train_fraud_model_logs_to_file_based_mlflow(tmp_path: Path, small_training_df):
    _, train_df = small_training_df

    tracking_dir = (tmp_path / "mlruns").resolve()
    tracking_uri = tracking_dir.as_uri()

    result: TrainingResult = train_fraud_model(
        df=train_df,
        target_column="is_fraud_label",
        tracking_uri=tracking_uri,
        n_trials=2,  # keep tests fast
        test_size=0.25,
        random_state=7,
    )

    assert isinstance(result.model, object)
    assert isinstance(result.feature_names, list)
    assert result.roc_auc >= 0.0 and result.roc_auc <= 1.0
    assert result.accuracy >= 0.0 and result.accuracy <= 1.0
    assert isinstance(result.run_id, str) and result.run_id

    # Inspect the tracking directory to make sure runs and SHAP artifacts exist.
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiments = client.search_experiments()
    assert any(exp.name == "fraud_detection" for exp in experiments)

    fraud_experiment = next(exp for exp in experiments if exp.name == "fraud_detection")
    runs = client.search_runs(experiment_ids=[fraud_experiment.experiment_id])
    assert runs, "Expected at least one MLflow run for fraud_detection"

    # Look for shap artifacts on the best run.
    best_run = max(runs, key=lambda r: r.data.metrics.get("roc_auc", 0.0))
    shap_artifacts = client.list_artifacts(best_run.info.run_id, path="shap")
    shap_filenames = {p.path for p in shap_artifacts}
    assert any(name.endswith("shap_values.npy") for name in shap_filenames)
    assert any(name.endswith("feature_names.txt") for name in shap_filenames)


