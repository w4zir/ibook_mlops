from __future__ import annotations

"""
Airflow DAG for Phase 5: feature engineering pipeline.

This DAG is intentionally lightweight but shaped like a real production workflow:

- Runs hourly.
- Stubs out Kafka ingestion and drift detection.
- Uses simple pandas-based aggregation on the local synthetic Feast data, when present.
- Touches the Feast repo via `common.feature_utils` to keep the contract realistic.

All external integrations are safe no-ops or best‑effort operations so the DAG can
be parsed and exercised in unit tests without requiring running services.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import logging

import pandas as pd
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from common.feature_utils import feature_store_healthcheck

try:
    from .utils import get_retries_for_dag, get_workspace_data_path
except ImportError:
    from utils import get_retries_for_dag, get_workspace_data_path


DAG_ID = "feature_engineering_pipeline"
DATA_DIR = get_workspace_data_path("processed", "feast")
logger = logging.getLogger(__name__)


def _extract_realtime_events(**_: Any) -> None:
    """
    Stub for Kafka/event-stream ingestion.

    In a real deployment this would consume from Kafka and write raw events into
    a landing table or object storage. For Phase 5 we simply log that the step
    ran so that the DAG topology is realistic without external dependencies.
    """
    logging.info("Extracting real-time events from Kafka (stub only for Phase 5).")


def _compute_batch_features(**_: Any) -> None:
    """
    Compute simple batch features from synthetic Feast data, when available.

    We keep the logic intentionally small and defensive so the DAG can run
    even if the sample data has not been generated yet.
    """
    import traceback

    event_path = DATA_DIR / "event_metrics.parquet"
    user_path = DATA_DIR / "user_metrics.parquet"

    logger.info(
        "compute_batch_features: DATA_DIR=%s (resolved=%s)",
        DATA_DIR,
        DATA_DIR.resolve(),
    )
    logger.info(
        "compute_batch_features: event_path.exists=%s, user_path.exists=%s; listing dir: %s",
        event_path.exists(),
        user_path.exists(),
        list(DATA_DIR.iterdir()) if DATA_DIR.exists() else "DIR_NOT_EXIST",
    )

    if not event_path.exists() or not user_path.exists():
        logger.warning(
            "Feast synthetic data not found under %s; skipping batch feature computation.",
            DATA_DIR,
        )
        return

    try:
        logger.info("Loading synthetic Feast data from %s", DATA_DIR)
        events = pd.read_parquet(event_path)
        users = pd.read_parquet(user_path)
        logger.info(
            "Loaded events shape=%s columns=%s, users shape=%s columns=%s",
            events.shape,
            list(events.columns),
            users.shape,
            list(users.columns),
        )

        if events.empty or users.empty:
            raise AirflowException(
                "Synthetic Feast datasets are empty; cannot compute features."
            )

        if "event_id" not in events.columns:
            raise AirflowException(
                "event_metrics.parquet missing 'event_id' column; got %s"
                % (list(events.columns),)
            )
        if "avg_ticket_price" not in events.columns:
            raise AirflowException(
                "event_metrics.parquet missing 'avg_ticket_price' column; got %s"
                % (list(events.columns),)
            )

        logger.info("Computing aggregation groupby event_id, avg_ticket_price")
        agg = (
            events.groupby("event_id")["avg_ticket_price"]
            .mean()
            .rename("avg_ticket_price_mean")
            .reset_index()
        )

        out_path = DATA_DIR / "event_aggregates.parquet"
        logger.info(
            "Writing aggregated features to %s (parent exists=%s)",
            out_path,
            out_path.parent.exists(),
        )
        try:
            agg.to_parquet(out_path, index=False)
            logger.info("Wrote aggregated event features to %s", out_path)
        except OSError as write_err:
            # Workspace is often mounted read-only in Docker; write to writable path.
            fallback_dir = Path("/opt/airflow/data/processed/feast")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_dir / "event_aggregates.parquet"
            logger.warning(
                "Could not write to %s (%s); writing to fallback %s",
                out_path,
                write_err,
                fallback_path,
            )
            agg.to_parquet(fallback_path, index=False)
            logger.info("Wrote aggregated event features to fallback %s", fallback_path)
    except Exception as e:
        logger.error(
            "compute_batch_features failed: %s\n%s",
            e,
            traceback.format_exc(),
            exc_info=True,
        )
        raise


def _validate_features(**_: Any) -> None:
    """
    Lightweight validation step standing in for Great Expectations.

    If the aggregate file is missing (e.g. compute_batch_features skipped because
    synthetic data was not present), we skip validation so the pipeline does not
    get stuck in retries.
    """
    agg_path = DATA_DIR / "event_aggregates.parquet"
    fallback_agg_path = Path("/opt/airflow/data/processed/feast/event_aggregates.parquet")
    if agg_path.exists():
        path_to_validate = agg_path
    elif fallback_agg_path.exists():
        path_to_validate = fallback_agg_path
        logger.info("Validating aggregate file from fallback path %s", path_to_validate)
    else:
        logger.warning(
            "Aggregated features at %s (or %s) not found (upstream may have skipped); skipping validation.",
            agg_path,
            fallback_agg_path,
        )
        return

    df = pd.read_parquet(path_to_validate)
    if df.empty:
        raise AirflowException("Aggregated feature file is empty.")

    if "event_id" not in df.columns:
        raise AirflowException("Aggregated feature file missing 'event_id' column.")

    if df["avg_ticket_price_mean"].isnull().any():
        raise AirflowException("Null values found in avg_ticket_price_mean.")

    logger.info("Basic feature validation passed for %s", path_to_validate)


def _materialize_to_feast(**_: Any) -> None:
    """
    Touch the Feast feature store to ensure configuration is healthy.

    We intentionally keep this to a metadata-level healthcheck so that the task
    is fast and safe in local environments.
    """
    info = feature_store_healthcheck()
    logging.info(
        "Feast healthcheck: repo_path=%s feature_views=%d (%s)",
        info["repo_path"],
        info["feature_view_count"],
        ", ".join(info["feature_view_names"]),
    )


def _check_for_drift(**_: Any) -> bool:
    """
    Stub for drift detection.

    A future phase can wire this into Evidently or other monitoring utilities.
    For now we simply return False (no drift) and log a message.
    """
    logging.info("Drift detection stub executed; returning no-drift for Phase 5.")
    return False


def _maybe_trigger_training(**context: Any) -> None:
    """
    Log whether model retraining would be triggered.

    We read the drift flag from XCom if present; otherwise we default to
    \"no drift\". This keeps the implementation simple while preserving the
    intent of conditional retraining.
    """
    ti = context["ti"]
    drift_flag = ti.xcom_pull(task_ids="check_for_drift")
    if drift_flag:
        logging.info("Drift detected (stub); would trigger model_training_pipeline DAG.")
    else:
        logging.info("No drift detected; skipping retraining trigger.")


default_args = {
    "owner": "ml-platform",
    "retries": get_retries_for_dag(DAG_ID, 3),
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id=DAG_ID,
    description="Hourly feature engineering pipeline for ticketing platform (Phase 5).",
    default_args=default_args,
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "features", "phase5"],
) as dag:
    start = EmptyOperator(task_id="start")

    extract_realtime_events = PythonOperator(
        task_id="extract_realtime_events",
        python_callable=_extract_realtime_events,
    )

    compute_batch_features = PythonOperator(
        task_id="compute_batch_features",
        python_callable=_compute_batch_features,
    )

    validate_features = PythonOperator(
        task_id="validate_features",
        python_callable=_validate_features,
    )

    materialize_to_feast = PythonOperator(
        task_id="materialize_to_feast",
        python_callable=_materialize_to_feast,
    )

    check_for_drift = PythonOperator(
        task_id="check_for_drift",
        python_callable=_check_for_drift,
    )

    maybe_trigger_training = PythonOperator(
        task_id="maybe_trigger_training",
        python_callable=_maybe_trigger_training,
        provide_context=True,
    )

    end = EmptyOperator(task_id="end")

    (
        start
        >> extract_realtime_events
        >> compute_batch_features
        >> validate_features
        >> materialize_to_feast
        >> check_for_drift
        >> maybe_trigger_training
        >> end
    )


__all__ = ["dag"]

