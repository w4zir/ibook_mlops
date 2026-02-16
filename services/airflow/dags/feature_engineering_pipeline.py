from __future__ import annotations

"""
Airflow DAG: feature engineering pipeline.

- Runs hourly.
- Reads raw events from MinIO (s3://raw-events/transactions/) and computes
  user_realtime_fraud_features (same aggregation logic as Faust worker) for
  offline store and training.
- Uses simple pandas-based aggregation on local synthetic Feast data for
  event-level features when present.
- Materializes to Feast (healthcheck + user_realtime_fraud_features when data exists).
- Drift detection and training trigger are stubs.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
    Read raw transaction events from MinIO (raw-events bucket), compute
    per-user aggregates aligned with Faust (user_txn_count_1h, user_txn_amount_1h,
    user_distinct_events_1h, user_avg_amount_24h), write to user_realtime_features.parquet.

    If MinIO is unavailable or the bucket is empty, log and skip (no failure).
    """
    import io

    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not available; skipping MinIO raw-events read.")
        return

    endpoint = os.environ.get("AWS_S3_ENDPOINT_URL") or os.environ.get("MLFLOW_S3_ENDPOINT_URL") or "http://localhost:9000"
    bucket = os.environ.get("RAW_EVENTS_BUCKET", "raw-events")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")

    try:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
        prefix = "transactions/"
        paginator = client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                k = obj.get("Key")
                if k and k.endswith(".parquet"):
                    keys.append(k)
        if not keys:
            logger.info("No Parquet files under s3://%s/%s; skipping user_realtime_features.", bucket, prefix)
            return

        dfs = []
        for key in keys:
            buf = io.BytesIO()
            client.download_fileobj(bucket, key, buf)
            buf.seek(0)
            dfs.append(pd.read_parquet(buf))
        raw = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        logger.warning("Failed to read raw events from MinIO (bucket=%s): %s; skipping.", bucket, e)
        return

    if raw.empty:
        logger.info("Raw events DataFrame is empty; skipping user_realtime_features.")
        return

    # Require columns from simulator transaction schema
    for col in ("user_id", "event_id", "total_amount"):
        if col not in raw.columns:
            logger.warning("Raw events missing column %s; skipping user_realtime_features.", col)
            return

    # Same aggregations as Faust worker (offline/historical version)
    agg = raw.groupby("user_id").agg(
        user_txn_count_1h=("user_id", "count"),
        user_txn_amount_1h=("total_amount", "sum"),
        user_distinct_events_1h=("event_id", "nunique"),
    ).reset_index()
    agg["user_avg_amount_24h"] = agg["user_txn_amount_1h"] / agg["user_txn_count_1h"].clip(lower=1)
    agg["event_timestamp"] = pd.Timestamp.utcnow()
    agg["ingested_at"] = pd.Timestamp.utcnow()

    out_path = DATA_DIR / "user_realtime_features.parquet"
    fallback_dir = Path("/opt/airflow/data/processed/feast")
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out_path, index=False)
        logger.info("Wrote user_realtime_features to %s (%d rows).", out_path, len(agg))
    except OSError as write_err:
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / "user_realtime_features.parquet"
        agg.to_parquet(fallback_path, index=False)
        logger.info("Wrote user_realtime_features to fallback %s (%d rows).", fallback_path, len(agg))


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
    Feast healthcheck and, when user_realtime_features.parquet exists,
    materialize user_realtime_fraud_features from batch source to online store.
    """
    info = feature_store_healthcheck()
    logging.info(
        "Feast healthcheck: repo_path=%s feature_views=%d (%s)",
        info["repo_path"],
        info["feature_view_count"],
        ", ".join(info["feature_view_names"]),
    )

    user_realtime_path = DATA_DIR / "user_realtime_features.parquet"
    fallback_path = Path("/opt/airflow/data/processed/feast/user_realtime_features.parquet")
    path_to_use = user_realtime_path if user_realtime_path.exists() else (fallback_path if fallback_path.exists() else None)
    if path_to_use and "user_realtime_fraud_features" in info["feature_view_names"]:
        try:
            from common.feature_utils import get_feature_store

            fs = get_feature_store()
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=1)
            fs.materialize(feature_views=["user_realtime_fraud_features"], start_date=start, end_date=end)
            logging.info("Materialized user_realtime_fraud_features to online store.")
        except Exception as e:
            logging.warning("Materialize user_realtime_fraud_features failed (non-fatal): %s", e)


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

