from __future__ import annotations

"""
ML monitoring pipeline: data-driven drift from feature pipeline artifacts,
threshold checks, and conditional trigger of model_training_pipeline.

Reads drift_summary.json (and optional feature outputs) produced by
feature_engineering_pipeline; computes threshold decision; triggers
model_training_pipeline with drift context when drift_score >= threshold.
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
from airflow.operators.python import BranchPythonOperator, PythonOperator

try:
    from .utils import get_retries_for_dag, get_workspace_data_path
except ImportError:
    from utils import get_retries_for_dag, get_workspace_data_path


DAG_ID = "ml_monitoring_pipeline"
MONITORING_DIR = get_workspace_data_path("monitoring")
MONITORING_DIR_FALLBACK = Path("/opt/airflow/data/monitoring")
# Feature pipeline writes drift_summary.json here (processed/feast or fallback).
FEAST_DATA_DIR = get_workspace_data_path("processed", "feast")
FEAST_FALLBACK_DIR = Path("/opt/airflow/data/processed/feast")
DRIFT_THRESHOLD = 0.30


def _find_drift_summary_path() -> Path | None:
    """Return path to feature pipeline drift_summary.json if it exists."""
    for d in (FEAST_DATA_DIR, FEAST_FALLBACK_DIR):
        p = d / "drift_summary.json"
        if p.exists():
            return p
    return None


def _collect_production_metrics(**_: Any) -> Dict[str, Any]:
    """
    Collect drift/metrics from feature pipeline artifact (drift_summary.json).
    If not present, fall back to a small synthetic sample for local runs.
    """
    summary_path = _find_drift_summary_path()
    if summary_path is not None:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            drift_score = float(data.get("drift_score", 0.0))
            drift_detected = bool(data.get("drift_detected", False))
            column_scores = data.get("column_scores") or {}
            reference_source = data.get("reference_source", "unknown")
            target_window_hours = data.get("target_window_hours")
            logging.info(
                "Collected drift from feature pipeline: score=%.4f detected=%s reference=%s target_window_hours=%s",
                drift_score,
                drift_detected,
                reference_source,
                target_window_hours,
            )
            return {
                "drift_score": drift_score,
                "drift_detected": drift_detected,
                "column_scores": column_scores,
                "source": "feature_pipeline",
                "run_ts": data.get("run_ts"),
                "reference_source": reference_source,
                "target_window_hours": target_window_hours,
            }
        except Exception as e:
            logging.warning("Failed to read drift_summary.json: %s; using synthetic.", e)

    # Fallback: synthetic proxy for drift when no feature artifact exists
    rng = np.random.default_rng(seed=123)
    n = 512
    y_true = rng.integers(0, 2, size=n)
    y_pred = np.clip(y_true * 0.7 + rng.normal(0.0, 0.2, size=n), 0.0, 1.0)
    mean_pred = float(np.mean(y_pred))
    mean_label = float(np.mean(y_true))
    drift_score = abs(mean_pred - mean_label)
    logging.info(
        "Collected synthetic monitoring metrics: mean_pred=%.4f mean_label=%.4f drift_score=%.4f",
        mean_pred,
        mean_label,
        drift_score,
    )
    return {
        "mean_prediction": mean_pred,
        "mean_label": mean_label,
        "drift_score": drift_score,
        "drift_detected": drift_score >= DRIFT_THRESHOLD,
        "source": "synthetic",
    }


def _compute_drift_reports(**context: Any) -> Dict[str, Any]:
    """
    Build drift report from collected metrics and persist daily_drift_summary.json.
    """
    ti = context["ti"]
    metrics: Dict[str, Any] = ti.xcom_pull(task_ids="collect_production_metrics") or {}
    drift_score = float(metrics.get("drift_score", 0.0))
    column_scores = metrics.get("column_scores") or {}

    report = {
        "drift_score": drift_score,
        "column_scores": column_scores,
        "source": metrics.get("source", "unknown"),
        "run_ts": datetime.utcnow().isoformat() + "Z",
    }
    if metrics.get("reference_source") is not None:
        report["reference_source"] = metrics["reference_source"]
    if metrics.get("target_window_hours") is not None:
        report["target_window_hours"] = metrics["target_window_hours"]
    if "mean_prediction" in metrics:
        report["mean_prediction"] = metrics["mean_prediction"]
        report["mean_label"] = metrics["mean_label"]

    out_dir = MONITORING_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "daily_drift_summary.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as e:
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
    logging.info("Wrote drift summary to %s", out_path)
    return report


def _check_thresholds(**context: Any) -> Dict[str, Any]:
    """
    Apply drift threshold to decide whether alerts and retraining are needed.
    """
    ti = context["ti"]
    report: Dict[str, Any] = ti.xcom_pull(task_ids="compute_drift_reports") or {}
    drift_score = float(report.get("drift_score", 0.0))
    column_scores = report.get("column_scores") or {}
    needs_retrain = drift_score >= DRIFT_THRESHOLD

    logging.info(
        "Drift evaluation: drift_score=%.4f (>= %.2f?) -> %s",
        drift_score,
        DRIFT_THRESHOLD,
        "RETRAIN" if needs_retrain else "OK",
    )

    return {
        "drift_score": drift_score,
        "drift_threshold": DRIFT_THRESHOLD,
        "needs_retrain": needs_retrain,
        "column_scores": column_scores,
        "source_window": report.get("run_ts"),
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


def _branch_trigger_retraining(**context: Any) -> str:
    """Return task_id for either trigger_retraining or skip_retraining."""
    ti = context["ti"]
    decision: Dict[str, Any] = ti.xcom_pull(task_ids="check_thresholds") or {}
    if decision.get("needs_retrain"):
        return "trigger_retraining"
    return "skip_retraining"


def _trigger_retraining(**context: Any) -> None:
    """
    Trigger model_training_pipeline DAG with drift context as conf.
    Only runs when branch_trigger_retraining chose this branch (needs_retrain=True).
    """
    ti = context["ti"]
    decision: Dict[str, Any] = ti.xcom_pull(task_ids="check_thresholds") or {}
    if not decision.get("needs_retrain"):
        logging.info("No retraining trigger; drift below threshold.")
        return

    conf = {
        "drift_score": decision.get("drift_score"),
        "drift_threshold": decision.get("drift_threshold"),
        "column_scores": decision.get("column_scores"),
        "source_window": decision.get("source_window"),
        "triggered_by": DAG_ID,
        "trigger_run_id": ti.dag_run.run_id if ti.dag_run else None,
    }
    try:
        from airflow.api.common.experimental.trigger_dag import trigger_dag
        run = trigger_dag(dag_id="model_training_pipeline", conf=conf)
        logging.info(
            "Triggered model_training_pipeline DAG (run_id=%s) due to drift_score=%.4f.",
            run.run_id if run else None,
            decision.get("drift_score"),
        )
    except Exception as e:
        logging.exception("Failed to trigger model_training_pipeline: %s", e)
        raise


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

    branch_trigger_retraining = BranchPythonOperator(
        task_id="branch_trigger_retraining",
        python_callable=_branch_trigger_retraining,
        provide_context=True,
    )

    trigger_retraining = PythonOperator(
        task_id="trigger_retraining",
        python_callable=_trigger_retraining,
        provide_context=True,
    )

    skip_retraining = EmptyOperator(task_id="skip_retraining")

    end = EmptyOperator(task_id="end")

    start >> collect_production_metrics >> compute_drift_reports >> check_thresholds
    check_thresholds >> [send_alerts, branch_trigger_retraining]
    branch_trigger_retraining >> [trigger_retraining, skip_retraining]
    send_alerts >> end
    trigger_retraining >> end
    skip_retraining >> end


__all__ = ["dag"]

