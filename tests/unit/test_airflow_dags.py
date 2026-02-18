from __future__ import annotations

import pytest


airflow = pytest.importorskip("airflow")
from services.airflow.dags.feature_engineering_pipeline import dag as feature_dag  # noqa: E402
from services.airflow.dags.model_training_pipeline import dag as training_dag  # noqa: E402
from services.airflow.dags.ml_monitoring_pipeline import dag as monitoring_dag  # noqa: E402


def test_feature_engineering_dag_shape() -> None:
    dag = feature_dag
    assert dag.schedule_interval == "@hourly"
    assert dag.catchup is False

    task_ids = {t.task_id for t in dag.tasks}
    for expected in [
        "start",
        "extract_realtime_events",
        "compute_batch_features",
        "validate_features",
        "materialize_to_feast",
        "check_for_drift",
        "maybe_trigger_training",
        "end",
    ]:
        assert expected in task_ids, f"Missing task_id={expected!r} in feature DAG"


def test_model_training_dag_shape() -> None:
    dag = training_dag
    assert dag.schedule_interval == "@weekly"
    assert dag.catchup is False

    task_ids = {t.task_id for t in dag.tasks}
    for expected in [
        "start",
        "build_training_dataset",
        "train_model",
        "evaluate_against_baseline",
        "register_and_mark_candidate",
        "deploy_canary",
        "monitor_canary",
        "finalize_promotion",
        "end",
    ]:
        assert expected in task_ids, f"Missing task_id={expected!r} in training DAG"


def test_ml_monitoring_dag_shape() -> None:
    dag = monitoring_dag
    assert dag.schedule_interval == "@daily"
    assert dag.catchup is False

    task_ids = {t.task_id for t in dag.tasks}
    for expected in [
        "start",
        "collect_production_metrics",
        "compute_drift_reports",
        "check_thresholds",
        "send_alerts",
        "branch_trigger_retraining",
        "trigger_retraining",
        "skip_retraining",
        "end",
    ]:
        assert expected in task_ids, f"Missing task_id={expected!r} in monitoring DAG"


def test_ml_monitoring_branch_returns_trigger_or_skip() -> None:
    """Branch chooses trigger_retraining when needs_retrain else skip_retraining."""
    from services.airflow.dags.ml_monitoring_pipeline import _branch_trigger_retraining

    class FakeTI:
        def __init__(self, decision: dict) -> None:
            self._decision = decision
        def xcom_pull(self, task_ids: str, key: str | None = None) -> dict:
            return self._decision

    out = _branch_trigger_retraining(ti=FakeTI({"needs_retrain": True}))
    assert out == "trigger_retraining"
    out = _branch_trigger_retraining(ti=FakeTI({"needs_retrain": False}))
    assert out == "skip_retraining"


def test_feature_pipeline_drift_uses_seed_reference() -> None:
    """Drift check uses seed-derived reference (no reference path; _build_seed_reference_features)."""
    from services.airflow.dags.feature_engineering_pipeline import (
        _get_feature_paths,
        _build_seed_reference_features,
    )

    paths = _get_feature_paths()
    assert len(paths) == 2, "Drift uses current + summary paths only; reference is regenerated from seed"
    current_path, summary_path = paths
    assert "user_realtime_features" in str(current_path)
    assert "drift_summary" in str(summary_path)

    ref_df = _build_seed_reference_features()
    assert ref_df is not None and not ref_df.empty
    for col in ("user_txn_count_1h", "user_txn_amount_1h", "user_distinct_events_1h", "user_avg_amount_24h"):
        assert col in ref_df.columns, f"Seed reference must include {col}"

