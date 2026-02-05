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
        "trigger_retraining",
        "end",
    ]:
        assert expected in task_ids, f"Missing task_id={expected!r} in monitoring DAG"

