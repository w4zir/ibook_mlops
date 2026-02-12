from __future__ import annotations

"""
Airflow DAG: auto-training on fraud detection failure rate.

Runs every N seconds (AUTO_TRAINING_DAG_INTERVAL_SECONDS, default 60). Fetches
failure rate from the BentoML fraud service (/admin/stats). If the rate meets
or exceeds the configured threshold and there are enough samples, triggers
model retraining, registers in MLflow as Production, and calls /admin/reload
on the BentoML service to hot-reload the new model.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from common.auto_training import _build_synthetic_dataset, _register_model_in_mlflow
from common.config import get_config
from common.model_utils import TrainingResult, build_fraud_training_dataframe, train_fraud_model

try:
    from .utils import get_retries_for_dag, get_workspace_data_path
except ImportError:
    from utils import get_retries_for_dag, get_workspace_data_path


DAG_ID = "auto_training_on_fraud_rate"
REALTIME_FEATURES_PATH = get_workspace_data_path("training", "realtime_features.parquet")
# When workspace is mounted read-only (e.g. Docker), use fallback.
REALTIME_FEATURES_FALLBACK = Path("/opt/airflow/data/training/realtime_features.parquet")
# Consider logged features "recent" if modified within this many seconds.
REALTIME_FEATURES_MAX_AGE_SECONDS = 86400  # 24 hours


def _bentoml_base_url() -> str:
    return (
        os.environ.get("BENTOML_BASE_URL")
        or os.environ.get("FRAUD_API_BASE_URL")
        or "http://localhost:7001"
    ).rstrip("/")


def _check_fraud_rate(**_: Any) -> Dict[str, Any]:
    """Fetch failure rate and sample count from BentoML /admin/stats."""
    import urllib.request

    url = f"{_bentoml_base_url()}/admin_stats"
    req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            data = json.loads(body) if body else {}
    except Exception as e:
        logging.warning("Failed to fetch admin_stats from %s: %s", url, e)
        # No stats: treat as no breach so we do not trigger training blindly.
        return {
            "failure_rate": 0.0,
            "window_samples": 0,
            "threshold": 0.4,
            "training_in_progress": False,
        }
    return {
        "failure_rate": float(data.get("failure_rate", 0.0)),
        "window_samples": int(data.get("window_samples", 0)),
        "threshold": float(data.get("threshold", 0.4)),
        "training_in_progress": bool(data.get("training_in_progress", False)),
    }


def _evaluate_threshold(**context: Any) -> bool:
    """
    Return True if we should proceed with retraining (threshold breached and enough samples).
    Return False to short-circuit downstream tasks.
    """
    ti = context["ti"]
    stats: Dict[str, Any] = ti.xcom_pull(task_ids="check_fraud_rate") or {}
    cfg = get_config().auto_training
    if not cfg.enabled:
        logging.info("Auto-training disabled in config; skipping.")
        return False

    failure_rate = float(stats.get("failure_rate", 0.0))
    window_samples = int(stats.get("window_samples", 0))
    threshold = cfg.failure_rate_threshold
    min_samples = cfg.min_samples

    if stats.get("training_in_progress"):
        logging.info("Training already in progress; skipping this run.")
        return False
    if window_samples < min_samples:
        logging.info(
            "Insufficient samples (window_samples=%d < min_samples=%d); skipping.",
            window_samples,
            min_samples,
        )
        return False
    if failure_rate < threshold:
        logging.info(
            "Failure rate %.2f%% below threshold %.2f%%; skipping.",
            failure_rate * 100,
            threshold * 100,
        )
        return False

    logging.info(
        "Threshold breached: failure_rate=%.2f%% (>= %.2f%%), samples=%d; triggering retrain.",
        failure_rate * 100,
        threshold * 100,
        window_samples,
    )
    return True


def _build_training_dataset(**_: Any) -> pd.DataFrame:
    """
    Build training DataFrame. Use logged realtime features if present and recent;
    otherwise use synthetic dataset from common.auto_training.
    """
    import time

    cfg = get_config().auto_training
    for path in (REALTIME_FEATURES_PATH, REALTIME_FEATURES_FALLBACK):
        if not path.exists():
            continue
        try:
            mtime = path.stat().st_mtime
            if (time.time() - mtime) > REALTIME_FEATURES_MAX_AGE_SECONDS:
                logging.info(
                    "Realtime features at %s are older than %ds; using synthetic.",
                    path,
                    REALTIME_FEATURES_MAX_AGE_SECONDS,
                )
                break
            df = pd.read_parquet(path)
            # Label: actual_fraud or is_fraud_label from simulator log.
            label_col = "actual_fraud" if "actual_fraud" in df.columns else "is_fraud_label"
            if label_col not in df.columns:
                logging.warning("Logged features missing label column; using synthetic.")
                break
            # Features expected by fraud model: lifetime_purchases, fraud_risk_score.
            if "lifetime_purchases" not in df.columns:
                df["lifetime_purchases"] = 0
            if "fraud_risk_score" not in df.columns:
                df["fraud_risk_score"] = 0.5
            df["is_fraud_label"] = df[label_col].astype(int)
            out = df[["lifetime_purchases", "fraud_risk_score", "is_fraud_label"]].dropna()
            if len(out) < 64:
                logging.warning("Logged features have too few rows (%d); using synthetic.", len(out))
                break
            logging.info("Using %d rows from %s for training.", len(out), path)
            return out
        except Exception as e:
            logging.warning("Could not use realtime features from %s: %s", path, e)
            break

    df = _build_synthetic_dataset(n_rows=cfg.training_dataset_size, seed=None)
    logging.info("Built synthetic training DataFrame with shape %s", df.shape)
    return df


def _train_model(**context: Any) -> Dict[str, Any]:
    """Train fraud model and log to MLflow."""
    ti = context["ti"]
    df: pd.DataFrame = ti.xcom_pull(task_ids="build_training_dataset")
    if df is None or df.empty:
        raise ValueError("Training dataset is missing or empty.")

    result: TrainingResult = train_fraud_model(
        df,
        target_column="is_fraud_label",
        experiment_name="fraud_detection_auto_retrain",
        n_trials=5,
    )
    logging.info(
        "Trained fraud model: run_id=%s roc_auc=%.4f accuracy=%.4f",
        result.run_id,
        result.roc_auc,
        result.accuracy,
    )
    return {
        "run_id": result.run_id,
        "roc_auc": result.roc_auc,
        "accuracy": result.accuracy,
    }


def _evaluate_quality(**context: Any) -> Dict[str, Any]:
    """Apply quality gates; downstream tasks check accepted."""
    ti = context["ti"]
    metrics: Dict[str, Any] = ti.xcom_pull(task_ids="train_model") or {}
    min_roc_auc = 0.60
    min_accuracy = 0.65
    roc_auc = float(metrics.get("roc_auc", 0.0))
    accuracy = float(metrics.get("accuracy", 0.0))
    accepted = roc_auc >= min_roc_auc and accuracy >= min_accuracy
    logging.info(
        "Quality gates: roc_auc=%.4f (>= %.2f?) accuracy=%.4f (>= %.2f?) -> %s",
        roc_auc,
        min_roc_auc,
        accuracy,
        min_accuracy,
        "ACCEPT" if accepted else "REJECT",
    )
    return {
        "run_id": metrics.get("run_id"),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "accepted": accepted,
    }


def _register_model(**context: Any) -> None:
    """Register and promote to Production in MLflow; skip if quality rejected."""
    ti = context["ti"]
    evaluation: Dict[str, Any] = ti.xcom_pull(task_ids="evaluate_quality") or {}
    if not evaluation.get("accepted"):
        raise AirflowSkipException("Model did not pass quality gates; skipping registration.")

    run_id = evaluation.get("run_id")
    if not run_id:
        raise ValueError("Missing run_id from train_model.")
    model_version = _register_model_in_mlflow(run_id=run_id, model_name="fraud_detection")
    if model_version is None:
        raise RuntimeError("Failed to register model in MLflow.")
    ti.xcom_push(key="model_version", value=model_version)


def _finalize_promotion(**context: Any) -> None:
    """Log promotion details."""
    ti = context["ti"]
    evaluation: Dict[str, Any] = ti.xcom_pull(task_ids="evaluate_quality") or {}
    model_version = ti.xcom_pull(task_ids="register_model", key="model_version")
    logging.info(
        "Promotion finalized: run_id=%s model_version=%s",
        evaluation.get("run_id"),
        model_version,
    )


def _notify_reload(**context: Any) -> None:
    """POST to BentoML /admin_reload to hot-reload the new model."""
    import urllib.request

    ti = context["ti"]
    evaluation: Dict[str, Any] = ti.xcom_pull(task_ids="evaluate_quality") or {}
    if not evaluation.get("accepted"):
        raise AirflowSkipException("Model not accepted; skipping reload.")

    url = f"{_bentoml_base_url()}/admin_reload"
    req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            data = json.loads(body) if body else {}
        status = data.get("status", "")
        if status == "reloaded":
            logging.info("BentoML reload succeeded: %s", data)
        else:
            logging.warning("BentoML reload returned: %s", data)
    except Exception as e:
        logging.exception("Failed to call admin_reload at %s: %s", url, e)
        raise


def _dag_interval_seconds() -> int:
    return int(os.environ.get("AUTO_TRAINING_DAG_INTERVAL_SECONDS", "60"))


default_args = {
    "owner": "ml-platform",
    "retries": get_retries_for_dag(DAG_ID, 1),
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    description="Auto-train fraud model when failure rate exceeds threshold; promote and reload BentoML.",
    default_args=default_args,
    schedule=timedelta(seconds=_dag_interval_seconds()),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "auto-training", "fraud-rate"],
) as dag:
    start = EmptyOperator(task_id="start")

    check_fraud_rate = PythonOperator(
        task_id="check_fraud_rate",
        python_callable=_check_fraud_rate,
    )

    evaluate_threshold = ShortCircuitOperator(
        task_id="evaluate_threshold",
        python_callable=_evaluate_threshold,
    )

    build_training_dataset = PythonOperator(
        task_id="build_training_dataset",
        python_callable=_build_training_dataset,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    evaluate_quality = PythonOperator(
        task_id="evaluate_quality",
        python_callable=_evaluate_quality,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
    )

    finalize_promotion = PythonOperator(
        task_id="finalize_promotion",
        python_callable=_finalize_promotion,
    )

    notify_reload = PythonOperator(
        task_id="notify_reload",
        python_callable=_notify_reload,
    )

    end = EmptyOperator(task_id="end")

    start >> check_fraud_rate >> evaluate_threshold >> build_training_dataset >> train_model >> evaluate_quality
    evaluate_quality >> register_model >> finalize_promotion >> notify_reload >> end


__all__ = ["dag"]
