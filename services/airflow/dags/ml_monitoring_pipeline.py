from __future__ import annotations

"""
Airflow DAG for Phase 5: ML monitoring pipeline.

This DAG provides a production-shaped skeleton for daily monitoring of model
performance and data drift. For Phase 5, it operates entirely on small,
synthetic in-memory data and writes lightweight summary artifacts so it can be
run safely in local environments without external systems.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import json
import logging

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

try:
    from .utils import get_retries_for_dag, get_workspace_data_path
except ImportError:
    from utils import get_retries_for_dag, get_workspace_data_path


DAG_ID = "ml_monitoring_pipeline"
MONITORING_DIR = get_workspace_data_path("monitoring")
# When workspace is mounted read-only (e.g. Docker), write to this path instead.
MONITORING_DIR_FALLBACK = Path("/opt/airflow/data/monitoring")


def _collect_production_metrics(**_: Any) -> Dict[str, float]:
    """
    Collect a small synthetic sample of production predictions and actuals.

    In a future phase this will read from real logs or a warehouse. For now we
    generate two distributions with a small configurable shift to mimic drift.
    """
    rng = np.random.default_rng(seed=123)
    n = 512

    # Synthetic \"actual\" labels and predicted scores.
    y_true = rng.integers(0, 2, size=n)
    y_pred = np.clip(
        y_true * 0.7 + rng.normal(0.0, 0.2, size=n),
        0.0,
        1.0,
    )

    # Compute a few summary statistics that will stand in for a full Evidently report.
    mean_pred = float(np.mean(y_pred))
    mean_label = float(np.mean(y_true))

    logging.info(
        "Collected synthetic monitoring metrics: mean_pred=%.4f mean_label=%.4f",
        mean_pred,
        mean_label,
    )
    return {
        "mean_prediction": mean_pred,
        "mean_label": mean_label,
    }


def _compute_drift_reports(**context: Any) -> Dict[str, float]:
    """
    Compute a minimal \"drift\" signal from the summary metrics.

    For Phase 5 we simply compute the absolute difference between the mean
    prediction and the mean label as a proxy for calibration / drift.
    """
    ti = context["ti"]
    metrics: Dict[str, float] = ti.xcom_pull(task_ids="collect_production_metrics") or {}

    mean_pred = float(metrics.get("mean_prediction", 0.0))
    mean_label = float(metrics.get("mean_label", 0.0))
    drift_score = abs(mean_pred - mean_label)

    report = {
        "mean_prediction": mean_pred,
        "mean_label": mean_label,
        "drift_score": drift_score,
    }
    out_dir = MONITORING_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "daily_drift_summary.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as e:
        # Workspace is often mounted read-only in Docker; write to writable path.
        out_dir = MONITORING_DIR_FALLBACK
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "daily_drift_summary.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logging.warning(
            "Could not write to %s (%s); wrote to fallback %s",
            MONITORING_DIR,
            e,
            out_path,
        )
    logging.info("Wrote synthetic drift summary to %s", out_path)

    return report


def _check_thresholds(**context: Any) -> Dict[str, Any]:
    """
    Apply simple thresholds to decide whether alerts and retraining are needed.
    """
    ti = context["ti"]
    report: Dict[str, float] = ti.xcom_pull(task_ids="compute_drift_reports") or {}
    drift_score = float(report.get("drift_score", 0.0))

    # Thresholds chosen to roughly mirror the intent from PLAN.md while working
    # on our synthetic metric.
    drift_threshold = 0.30
    needs_retrain = drift_score >= drift_threshold

    logging.info(
        "Drift evaluation: drift_score=%.4f (>= %.2f?) -> %s",
        drift_score,
        drift_threshold,
        "RETRAIN" if needs_retrain else "OK",
    )

    return {
        "drift_score": drift_score,
        "drift_threshold": drift_threshold,
        "needs_retrain": needs_retrain,
    }


def _send_alerts(**context: Any) -> None:
    """
    Stub for sending alerts to Slack/PagerDuty.
    """
    ti = context["ti"]
    decision: Dict[str, Any] = ti.xcom_pull(task_ids="check_thresholds") or {}
    if decision.get("needs_retrain"):
        logging.info(
            "Would send alert to Slack/PagerDuty: drift_score=%.4f exceeded threshold=%.2f.",
            decision.get("drift_score"),
            decision.get("drift_threshold"),
        )
    else:
        logging.info("No alert sent; drift within acceptable bounds.")


def _trigger_retraining(**context: Any) -> None:
    """
    Stub for triggering the model_training_pipeline DAG.

    A future phase can replace this with Airflow's TriggerDagRunOperator.
    """
    ti = context["ti"]
    decision: Dict[str, Any] = ti.xcom_pull(task_ids="check_thresholds") or {}
    if decision.get("needs_retrain"):
        logging.info(
            "Would trigger model_training_pipeline DAG due to drift_score=%.4f.",
            decision.get("drift_score"),
        )
    else:
        logging.info("No retraining trigger; drift below threshold.")


default_args = {
    "owner": "ml-platform",
    "retries": get_retries_for_dag(DAG_ID, 1),
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id=DAG_ID,
    description="Daily ML monitoring and drift evaluation pipeline (Phase 5).",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "monitoring", "phase5"],
) as dag:
    start = EmptyOperator(task_id="start")

    collect_production_metrics = PythonOperator(
        task_id="collect_production_metrics",
        python_callable=_collect_production_metrics,
    )

    compute_drift_reports = PythonOperator(
        task_id="compute_drift_reports",
        python_callable=_compute_drift_reports,
        provide_context=True,
    )

    check_thresholds = PythonOperator(
        task_id="check_thresholds",
        python_callable=_check_thresholds,
        provide_context=True,
    )

    send_alerts = PythonOperator(
        task_id="send_alerts",
        python_callable=_send_alerts,
        provide_context=True,
    )

    trigger_retraining = PythonOperator(
        task_id="trigger_retraining",
        python_callable=_trigger_retraining,
        provide_context=True,
    )

    end = EmptyOperator(task_id="end")

    start >> collect_production_metrics >> compute_drift_reports >> check_thresholds
    check_thresholds >> [send_alerts, trigger_retraining] >> end


__all__ = ["dag"]

