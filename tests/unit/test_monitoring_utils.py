"""Unit tests for common.monitoring_utils (Phase 6)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.monitoring_utils import (
    DriftResult,
    PerformanceResult,
    check_alert_thresholds,
    compare_model_performance,
    extract_prometheus_metrics,
    generate_drift_report,
    generate_prediction_drift_report,
    save_report_html,
    set_prometheus_gauges,
)


def test_generate_drift_report_returns_result():
    """Synthetic reference/current DataFrames yield a DriftResult with expected fields."""
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    cur = pd.DataFrame({"a": [1.1, 2.1, 3.0], "b": [11.0, 19.0, 31.0]})
    result = generate_drift_report(ref, cur, include_html=False)
    assert isinstance(result, DriftResult)
    assert hasattr(result, "drift_score")
    assert hasattr(result, "drift_detected")
    assert hasattr(result, "column_scores")
    assert 0 <= result.drift_score <= 1.0
    assert isinstance(result.column_scores, dict)


def test_drift_report_detects_shift():
    """Obvious distribution shift yields high drift score or drift_detected True."""
    ref = pd.DataFrame({"x": np.random.randn(200) * 1.0 + 0.0})
    cur = pd.DataFrame({"x": np.random.randn(200) * 1.0 + 5.0})  # mean shift
    result = generate_drift_report(ref, cur, include_html=False)
    assert result.drift_score > 0.1 or result.drift_detected


def test_no_drift_on_identical_data():
    """Identical reference and current yield low drift score and drift_detected False."""
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    cur = ref.copy()
    result = generate_drift_report(ref, cur, include_html=False)
    assert result.drift_score == 0.0
    assert result.drift_detected is False


def test_compare_model_performance():
    """Synthetic predictions and labels yield PerformanceResult with accuracy metrics."""
    ref = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "prediction": [0, 1, 0, 1],
    })
    cur = pd.DataFrame({
        "target": [0, 1, 1, 0],
        "prediction": [0, 1, 0, 1],
    })
    result = compare_model_performance(
        ref, cur,
        target_column="target",
        prediction_column="prediction",
    )
    assert isinstance(result, PerformanceResult)
    assert result.accuracy_reference == 1.0
    assert 0 <= result.accuracy_current <= 1.0
    assert "accuracy" in result.metrics_current
    assert "accuracy" in result.metrics_reference


def test_check_alert_thresholds_triggers():
    """When drift_score >= threshold, check_alert_thresholds returns True."""
    result = DriftResult(drift_score=0.5, drift_detected=True, column_scores={})
    assert check_alert_thresholds(result, drift_threshold=0.3) is True
    assert check_alert_thresholds(result, drift_threshold=0.6) is False


def test_check_alert_thresholds_no_trigger():
    """When drift_score < threshold, check_alert_thresholds returns False."""
    result = DriftResult(drift_score=0.1, drift_detected=False, column_scores={})
    assert check_alert_thresholds(result, drift_threshold=0.3) is False
    # Performance degradation trigger
    assert check_alert_thresholds(
        result,
        drift_threshold=0.5,
        performance_threshold=0.2,
        performance_current=0.7,
        performance_reference=1.0,
    ) is True


def test_extract_prometheus_metrics():
    """DriftResult is converted to dict with ml_data_drift_score and ml_drift_detected."""
    result = DriftResult(drift_score=0.25, drift_detected=False, column_scores={})
    metrics = extract_prometheus_metrics(result)
    assert "ml_data_drift_score" in metrics
    assert "ml_drift_detected" in metrics
    assert metrics["ml_data_drift_score"] == 0.25
    assert metrics["ml_drift_detected"] == 0.0
    result2 = DriftResult(drift_score=0.4, drift_detected=True, column_scores={})
    metrics2 = extract_prometheus_metrics(result2)
    assert metrics2["ml_drift_detected"] == 1.0


def test_save_report_html_to_file(tmp_path: Path):
    """HTML from DriftResult is written to the given path."""
    result = DriftResult(
        drift_score=0.2,
        drift_detected=False,
        column_scores={},
        html="<html><body><p>Test report</p></body></html>",
    )
    out = tmp_path / "report.html"
    save_report_html(result, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "Test report" in content
    assert "<html>" in content


def test_set_prometheus_gauges_no_raise():
    """set_prometheus_gauges runs without error (may be no-op if prometheus_client missing)."""
    result = DriftResult(drift_score=0.15, drift_detected=False, column_scores={})
    set_prometheus_gauges(result, prefix="test_ml_")


def test_generate_prediction_drift_report():
    """Prediction drift report returns DriftResult (fallback or Evidently)."""
    ref = pd.DataFrame({"pred": [0.1, 0.9, 0.2]})
    cur = pd.DataFrame({"pred": [0.8, 0.2, 0.7]})
    result = generate_prediction_drift_report(
        ref, cur,
        prediction_column="pred",
    )
    assert isinstance(result, DriftResult)
    assert 0 <= result.drift_score <= 1.0
