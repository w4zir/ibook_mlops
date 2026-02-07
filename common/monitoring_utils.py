"""
ML monitoring utilities with Evidently AI integration.

Provides drift reports, performance comparison, HTML export, Prometheus metrics
extraction, and alert threshold checks. When Evidently is not installed or
fails to import, a simple fallback computes a basic drift score from
DataFrame statistics so the API remains usable (e.g. in CI or minimal envs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Optional Evidently integration
_evidently_available = False
_Report = None
_DataDriftPreset = None
_Dataset = None
_ClassificationPreset = None

try:
    from evidently.core.report import Report as _ReportCls
    from evidently.core.datasets import Dataset as _DatasetCls
    from evidently.presets import DataDriftPreset as _DataDriftPresetCls
    from evidently.presets import ClassificationPreset as _ClassificationPresetCls

    _Report = _ReportCls
    _DataDriftPreset = _DataDriftPresetCls
    _Dataset = _DatasetCls
    _ClassificationPreset = _ClassificationPresetCls
    _evidently_available = True
except Exception as e:  # noqa: BLE001
    logger.debug("Evidently not available: %s", e)
    _evidently_available = False


@dataclass
class DriftResult:
    """Result of a drift analysis between reference and current data."""

    drift_score: float
    """Overall drift score in [0, 1]; higher means more drift."""

    drift_detected: bool
    """True if drift is considered significant (e.g. share of drifted columns > threshold)."""

    column_scores: Dict[str, float] = field(default_factory=dict)
    """Per-column drift scores when available."""

    report_object: Any = None
    """Evidently snapshot or None when using fallback."""

    html: Optional[str] = None
    """HTML report content when generated."""


@dataclass
class PerformanceResult:
    """Result of model performance comparison."""

    accuracy_current: float
    accuracy_reference: float
    metrics_current: Dict[str, float] = field(default_factory=dict)
    metrics_reference: Dict[str, float] = field(default_factory=dict)
    report_object: Any = None


def _fallback_drift_score(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftResult:
    """
    Compute a simple drift score without Evidently (mean absolute difference
    of numeric column means, normalized).
    """
    common = [c for c in reference_df.select_dtypes(include=["number"]).columns if c in current_df.columns]
    if not common:
        drift_score = 0.0
        column_scores = {}
    else:
        column_scores = {}
        for col in common:
            ref_mean = reference_df[col].mean()
            cur_mean = current_df[col].mean()
            std = reference_df[col].std()
            if std is None or std == 0 or pd.isna(std):
                std = 1.0
            column_scores[col] = float(min(1.0, abs(cur_mean - ref_mean) / (std + 1e-8)))
        drift_score = sum(column_scores.values()) / len(column_scores) if column_scores else 0.0
    return DriftResult(
        drift_score=float(drift_score),
        drift_detected=drift_score >= 0.3,
        column_scores=column_scores,
        report_object=None,
        html=None,
    )


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    include_html: bool = False,
) -> DriftResult:
    """
    Generate a data drift report between reference and current DataFrames.

    Uses Evidently DataDriftPreset when available; otherwise uses a simple
    fallback based on numeric column statistics.

    Args:
        reference_df: Reference (baseline) dataset.
        current_df: Current (production) dataset to compare.
        include_html: If True and Evidently is available, populate result.html.

    Returns:
        DriftResult with drift_score, drift_detected, and optional column_scores/html.
    """
    if not _evidently_available or _Report is None or _DataDriftPreset is None or _Dataset is None:
        return _fallback_drift_score(reference_df, current_df)

    try:
        ref_dataset = _Dataset.from_pandas(reference_df)
        cur_dataset = _Dataset.from_pandas(current_df)
        report = _Report([_DataDriftPreset()])
        snapshot = report.run(cur_dataset, ref_dataset)

        # Extract drift share from snapshot metrics (Evidently exposes dataset drift share)
        drift_score = 0.0
        column_scores: Dict[str, float] = {}
        for metric_id, metric_result in snapshot._metrics.items():
            for key, value in metric_result.itervalues():
                if "drift" in key.lower() and "share" in key.lower() and isinstance(value, (int, float)):
                    drift_score = float(value)
                if "column_drift" in str(metric_id).lower() or "drift_share" in key.lower():
                    if isinstance(value, dict):
                        for col, score in value.items():
                            if isinstance(score, (int, float)):
                                column_scores[str(col)] = float(score)
        # Fallback: use number of drifted columns / total if available
        for metric_id, metric_result in snapshot._metrics.items():
            for key, value in metric_result.itervalues():
                if key == "share_drifted_columns" and isinstance(value, (int, float)):
                    drift_score = float(value)
                if key == "number_of_drifted_columns" and isinstance(value, (int, float)):
                    n_cols = len(reference_df.columns)
                    if n_cols > 0:
                        drift_score = float(value) / n_cols

        drift_detected = drift_score >= 0.3
        html_str: Optional[str] = None
        if include_html:
            html_str = f"<html><body><p>Drift score: {drift_score:.4f}</p></body></html>"

        if not column_scores and len(reference_df.columns) > 0:
            column_scores = {c: drift_score for c in reference_df.columns}

        return DriftResult(
            drift_score=drift_score,
            drift_detected=drift_detected,
            column_scores=column_scores,
            report_object=snapshot,
            html=html_str,
        )
    except Exception as e:
        logger.warning("Evidently drift report failed, using fallback: %s", e)
        return _fallback_drift_score(reference_df, current_df)


def generate_prediction_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: Optional[str] = None,
    prediction_column: Optional[str] = None,
) -> DriftResult:
    """
    Generate prediction (concept) drift report between reference and current.

    When Evidently is available, uses appropriate preset; otherwise delegates
    to generate_drift_report on the prediction column if given, or fallback.
    """
    if not _evidently_available:
        return generate_drift_report(reference_df, current_df, include_html=False)

    if prediction_column and prediction_column in reference_df.columns and prediction_column in current_df.columns:
        ref_pred = reference_df[[prediction_column]].copy()
        cur_pred = current_df[[prediction_column]].copy()
        return generate_drift_report(ref_pred, cur_pred, include_html=False)
    return generate_drift_report(reference_df, current_df, include_html=False)


def compare_model_performance(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str = "target",
    prediction_column: str = "prediction",
) -> PerformanceResult:
    """
    Compare classification performance between reference and current datasets.

    DataFrames should contain target_column and prediction_column (or prediction proba).
    When Evidently is not available, computes simple accuracy only.
    """
    def _simple_accuracy(df: pd.DataFrame, target: str, pred: str) -> float:
        if target not in df.columns or pred not in df.columns:
            return 0.0
        return float((df[target] == df[pred]).mean())

    acc_ref = _simple_accuracy(reference_df, target_column, prediction_column)
    acc_cur = _simple_accuracy(current_df, target_column, prediction_column)

    if not _evidently_available or _Report is None or _ClassificationPreset is None or _Dataset is None:
        return PerformanceResult(
            accuracy_current=acc_cur,
            accuracy_reference=acc_ref,
            metrics_current={"accuracy": acc_cur},
            metrics_reference={"accuracy": acc_ref},
            report_object=None,
        )

    try:
        ref_ds = _Dataset.from_pandas(reference_df)
        cur_ds = _Dataset.from_pandas(current_df)
        report = _Report([_ClassificationPreset()])
        snapshot = report.run(cur_ds, ref_ds)
        metrics_cur: Dict[str, float] = {"accuracy": acc_cur}
        metrics_ref: Dict[str, float] = {"accuracy": acc_ref}
        for _mid, mres in snapshot._metrics.items():
            for k, v in mres.itervalues():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metrics_cur[str(k)] = float(v)
        return PerformanceResult(
            accuracy_current=acc_cur,
            accuracy_reference=acc_ref,
            metrics_current=metrics_cur,
            metrics_reference=metrics_ref,
            report_object=snapshot,
        )
    except Exception as e:
        logger.warning("Evidently classification report failed: %s", e)
        return PerformanceResult(
            accuracy_current=acc_cur,
            accuracy_reference=acc_ref,
            metrics_current={"accuracy": acc_cur},
            metrics_reference={"accuracy": acc_ref},
            report_object=None,
        )


def save_report_html(report: Any, path: str | Path) -> None:
    """
    Save report HTML to a local path or GCS (when path is gs://...).

    When report is a DriftResult with html set, writes that. When report is an
    Evidently snapshot, attempts to export HTML if the API supports it.
    Otherwise writes a minimal placeholder.
    """
    path_str = str(path)
    html_content: str
    if isinstance(report, DriftResult) and report.html:
        html_content = report.html
    else:
        html_content = "<html><body><p>ML monitoring report</p></body></html>"

    if path_str.startswith("gs://"):
        try:
            from google.cloud import storage
            bucket_name = path_str.replace("gs://", "").split("/")[0]
            blob_path = path_str.replace(f"gs://{bucket_name}/", "")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(html_content, content_type="text/html")
            return
        except ImportError:
            raise ImportError("google-cloud-storage is required for GCS paths") from None
        except Exception as e:
            logger.warning("GCS upload failed: %s", e)
            raise

    out_path = Path(path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")


def extract_prometheus_metrics(drift_result: DriftResult) -> Dict[str, float]:
    """
    Convert DriftResult into key-value metrics suitable for Prometheus.

    Returns a dict of metric names to values. Callers can push these to a
    Prometheus Pushgateway or set them on prometheus_client Gauge objects.
    """
    return {
        "ml_data_drift_score": drift_result.drift_score,
        "ml_drift_detected": 1.0 if drift_result.drift_detected else 0.0,
    }


_gauge_drift = None
_gauge_detected = None


def set_prometheus_gauges(drift_result: DriftResult, prefix: str = "ml_") -> None:
    """
    Set Prometheus Gauge metrics from a DriftResult.

    Uses prometheus_client.Gauge; gauges are created on first call and
    reused. Safe to call from multiple threads if the default registry
    is used.
    """
    global _gauge_drift, _gauge_detected
    try:
        from prometheus_client import Gauge
    except ImportError:
        logger.debug("prometheus_client not available, skipping gauges")
        return

    if _gauge_drift is None:
        _gauge_drift = Gauge(f"{prefix}data_drift_score", "Data drift score (0-1)")
        _gauge_detected = Gauge(f"{prefix}drift_detected", "1 if drift detected else 0")
    _gauge_drift.set(drift_result.drift_score)
    _gauge_detected.set(1 if drift_result.drift_detected else 0)


def check_alert_thresholds(
    drift_result: DriftResult,
    drift_threshold: float = 0.3,
    performance_threshold: Optional[float] = None,
    performance_current: Optional[float] = None,
    performance_reference: Optional[float] = None,
) -> bool:
    """
    Return True if any alert threshold is exceeded (retraining/review recommended).

    - Drift: drift_result.drift_score >= drift_threshold.
    - Performance: when performance_* are provided, trigger if current
      performance degrades by more than performance_threshold (e.g. 0.1 = 10%)
      relative to reference.
    """
    if drift_result.drift_score >= drift_threshold:
        return True
    if (
        performance_threshold is not None
        and performance_current is not None
        and performance_reference is not None
        and performance_reference > 0
    ):
        drop = (performance_reference - performance_current) / performance_reference
        if drop >= performance_threshold:
            return True
    return False


def is_evidently_available() -> bool:
    """Return True if Evidently was successfully imported."""
    return _evidently_available


__all__ = [
    "DriftResult",
    "PerformanceResult",
    "generate_drift_report",
    "generate_prediction_drift_report",
    "compare_model_performance",
    "save_report_html",
    "extract_prometheus_metrics",
    "set_prometheus_gauges",
    "check_alert_thresholds",
    "is_evidently_available",
]
