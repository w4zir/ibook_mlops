from __future__ import annotations

"""
Airflow DAG: feature engineering pipeline.

- Runs hourly.
- Reads raw events from MinIO (s3://raw-events/transactions/) within a
  configurable time window and computes user_realtime_fraud_features
  (same aggregation logic as Faust worker) for offline store and training.
- Uses simple pandas-based aggregation on local synthetic Feast data for
  event-level features when present.
- Materializes to Feast (healthcheck + user_realtime_fraud_features when data exists).
- Drift detection: reference = seed-derived features (regenerated each run); target = last 24h from MinIO. Reference is never updated from target.
"""

import json
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
FALLBACK_DIR = Path("/opt/airflow/data/processed/feast")
# Time window (hours) for raw events read from MinIO; env FEATURE_RAW_EVENTS_HOURS.
DEFAULT_RAW_EVENTS_HOURS = 24
# Seed reference for drift: env DRIFT_REFERENCE_SEED, DRIFT_REFERENCE_*.
DEFAULT_DRIFT_REFERENCE_SEED = 42
DEFAULT_DRIFT_REFERENCE_EVENTS = 50
DEFAULT_DRIFT_REFERENCE_USERS = 500
DEFAULT_DRIFT_REFERENCE_TRANSACTIONS = 5000
logger = logging.getLogger(__name__)


def _extract_realtime_events(**_: Any) -> dict:
    """
    Read raw transaction events from MinIO (raw-events bucket) within a
    configurable time window (FEATURE_RAW_EVENTS_HOURS, default 24), compute
    per-user aggregates aligned with Faust, write user_realtime_features.parquet.

    Returns run metadata (window_hours, window_end_utc, rows_processed) for XCom.
    If MinIO is unavailable or the bucket is empty, log and skip (no failure).
    """
    import io

    try:
        import boto3
    except ImportError:
        logger.warning("boto3 not available; skipping MinIO raw-events read.")
        return {"window_hours": 0, "window_end_utc": None, "rows_processed": 0}

    endpoint = os.environ.get("AWS_S3_ENDPOINT_URL") or os.environ.get("MLFLOW_S3_ENDPOINT_URL") or "http://localhost:9000"
    bucket = os.environ.get("RAW_EVENTS_BUCKET", "raw-events")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
    window_hours = int(os.environ.get("FEATURE_RAW_EVENTS_HOURS", str(DEFAULT_RAW_EVENTS_HOURS)))

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
            return {"window_hours": window_hours, "window_end_utc": datetime.now(timezone.utc).isoformat(), "rows_processed": 0}

        dfs = []
        for key in keys:
            buf = io.BytesIO()
            client.download_fileobj(bucket, key, buf)
            buf.seek(0)
            dfs.append(pd.read_parquet(buf))
        raw = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        logger.warning("Failed to read raw events from MinIO (bucket=%s): %s; skipping.", bucket, e)
        return {"window_hours": window_hours, "window_end_utc": datetime.now(timezone.utc).isoformat(), "rows_processed": 0}

    if raw.empty:
        logger.info("Raw events DataFrame is empty; skipping user_realtime_features.")
        return {"window_hours": window_hours, "window_end_utc": datetime.now(timezone.utc).isoformat(), "rows_processed": 0}

    # Filter to recent window if timestamp column present
    if "timestamp" in raw.columns and window_hours > 0:
        try:
            raw["_ts"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
            cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
            raw = raw.loc[raw["_ts"] >= cutoff].drop(columns=["_ts"])
        except Exception as e:
            logger.warning("Could not filter by timestamp window: %s; using full data.", e)
    rows_processed = len(raw)

    # Require columns from simulator transaction schema
    for col in ("user_id", "event_id", "total_amount"):
        if col not in raw.columns:
            logger.warning("Raw events missing column %s; skipping user_realtime_features.", col)
            return {"window_hours": window_hours, "window_end_utc": datetime.now(timezone.utc).isoformat(), "rows_processed": 0}

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
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(out_path, index=False)
        logger.info("Wrote user_realtime_features to %s (%d rows, window_hours=%s).", out_path, len(agg), window_hours)
    except OSError as write_err:
        FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        fallback_path = FALLBACK_DIR / "user_realtime_features.parquet"
        agg.to_parquet(fallback_path, index=False)
        logger.info("Wrote user_realtime_features to fallback %s (%d rows).", fallback_path, len(agg))

    return {
        "window_hours": window_hours,
        "window_end_utc": datetime.now(timezone.utc).isoformat(),
        "rows_processed": rows_processed,
    }


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
            FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
            fallback_path = FALLBACK_DIR / "event_aggregates.parquet"
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
    fallback_agg_path = FALLBACK_DIR / "event_aggregates.parquet"
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
    fallback_path = FALLBACK_DIR / "user_realtime_features.parquet"
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


def _get_feature_paths() -> tuple[Path, Path]:
    """Return (current_path, drift_summary_path); use DATA_DIR or FALLBACK_DIR."""
    current = DATA_DIR / "user_realtime_features.parquet"
    summary = DATA_DIR / "drift_summary.json"
    if not current.exists():
        current = FALLBACK_DIR / "user_realtime_features.parquet"
        summary = FALLBACK_DIR / "drift_summary.json"
    return current, summary


def _build_seed_reference_features() -> pd.DataFrame | None:
    """
    Regenerate reference features from deterministic seed transactions each run.
    Returns DataFrame with columns user_txn_count_1h, user_txn_amount_1h,
    user_distinct_events_1h, user_avg_amount_24h (same schema as target features).
    """
    try:
        from common.seed_transactions import generate_seed_transactions
    except ImportError as e:
        logger.warning("common.seed_transactions not available for drift reference: %s", e)
        return None
    seed = int(os.environ.get("DRIFT_REFERENCE_SEED", str(DEFAULT_DRIFT_REFERENCE_SEED)))
    n_events = int(os.environ.get("DRIFT_REFERENCE_EVENTS", str(DEFAULT_DRIFT_REFERENCE_EVENTS)))
    n_users = int(os.environ.get("DRIFT_REFERENCE_USERS", str(DEFAULT_DRIFT_REFERENCE_USERS)))
    n_transactions = int(os.environ.get("DRIFT_REFERENCE_TRANSACTIONS", str(DEFAULT_DRIFT_REFERENCE_TRANSACTIONS)))
    txns = generate_seed_transactions(
        seed=seed,
        n_events=n_events,
        n_users=n_users,
        n_transactions=n_transactions,
    )
    if not txns:
        return None
    raw = pd.DataFrame(txns)
    agg = raw.groupby("user_id").agg(
        user_txn_count_1h=("user_id", "count"),
        user_txn_amount_1h=("total_amount", "sum"),
        user_distinct_events_1h=("event_id", "nunique"),
    ).reset_index()
    agg["user_avg_amount_24h"] = agg["user_txn_amount_1h"] / agg["user_txn_count_1h"].clip(lower=1)
    return agg


def _check_for_drift(**_: Any) -> bool:
    """
    Compare current user_realtime_features (last 24h from MinIO) to seed-derived
    reference (regenerated each run). Persist drift summary JSON. Reference is
    never updated from target.
    """
    from common.monitoring_utils import generate_drift_report

    window_hours = int(os.environ.get("FEATURE_RAW_EVENTS_HOURS", str(DEFAULT_RAW_EVENTS_HOURS)))
    current_path, summary_path = _get_feature_paths()
    if not current_path.exists():
        logger.info("No current user_realtime_features at %s; skipping drift check.", current_path)
        return False

    current_df = pd.read_parquet(current_path)
    numeric_cols = [c for c in current_df.columns if c in (
        "user_txn_count_1h", "user_txn_amount_1h", "user_distinct_events_1h", "user_avg_amount_24h"
    )]
    if not numeric_cols:
        logger.warning("No numeric feature columns for drift; skipping.")
        return False
    current_sub = current_df[numeric_cols]

    reference_df = _build_seed_reference_features()
    if reference_df is None or reference_df.empty:
        logger.warning("Could not build seed reference features; skipping drift check.")
        summary = {
            "drift_score": 0.0,
            "drift_detected": False,
            "column_scores": {},
            "run_ts": datetime.now(timezone.utc).isoformat(),
            "reference_source": "seed_regenerated",
            "target_window_hours": window_hours,
            "note": "reference build failed or empty",
        }
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not write drift summary: %s", e)
        return False

    ref_sub = reference_df[numeric_cols].reindex(columns=numeric_cols).dropna(how="all")
    if ref_sub.empty or current_sub.empty:
        logger.info("Reference or current feature subset empty; skipping drift.")
        return False

    result = generate_drift_report(ref_sub, current_sub, include_html=False)
    summary = {
        "drift_score": result.drift_score,
        "drift_detected": result.drift_detected,
        "column_scores": result.column_scores,
        "run_ts": datetime.now(timezone.utc).isoformat(),
        "reference_source": "seed_regenerated",
        "target_window_hours": window_hours,
    }
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not write drift summary: %s", e)

    logger.info(
        "Drift check: score=%.4f detected=%s (reference=seed, target=last %sh)",
        result.drift_score,
        result.drift_detected,
        window_hours,
    )
    return result.drift_detected


def _maybe_trigger_training(**context: Any) -> None:
    """
    Log whether model retraining would be triggered.

    Reads the drift flag from check_for_drift (XCom). When drift is detected,
    the ml_monitoring_pipeline (or this DAG in a future variant) can trigger
    model_training_pipeline; here we only log the intent.
    """
    ti = context["ti"]
    drift_flag = ti.xcom_pull(task_ids="check_for_drift")
    if drift_flag:
        logging.info("Drift detected; would trigger model_training_pipeline DAG.")
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

