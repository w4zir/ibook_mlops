"""Tests for simulator validators - latency, accuracy, drift, business."""

from __future__ import annotations

import pytest

from simulator.validators.accuracy_validator import AccuracyValidator
from simulator.validators.business_validator import BusinessValidator
from simulator.validators.drift_validator import DriftValidator
from simulator.validators.latency_validator import LatencyValidator


def test_latency_validator_passes_under_sla() -> None:
    validator = LatencyValidator(p50_max_ms=50, p95_max_ms=150, p99_max_ms=200)
    responses = [{"latency_ms": 30 + (i % 20)} for i in range(100)]
    result = validator.validate(responses)
    assert result["passed"] is True
    assert result["p50_ms"] is not None
    assert result["p99_ms"] is not None


def test_latency_validator_fails_over_sla() -> None:
    validator = LatencyValidator(p50_max_ms=10, p95_max_ms=10, p99_max_ms=10)
    responses = [{"latency_ms": 100} for _ in range(50)]
    result = validator.validate(responses)
    assert result["passed"] is False


def test_latency_validator_empty_responses() -> None:
    validator = LatencyValidator()
    result = validator.validate([])
    assert result["passed"] is False
    assert "reason" in result


def test_accuracy_validator_computes_metrics() -> None:
    validator = AccuracyValidator()
    responses = [
        {"is_fraud": True, "blocked": True},
        {"is_fraud": True, "blocked": False},
        {"is_fraud": False, "blocked": False},
        {"is_fraud": False, "blocked": True},
    ]
    result = validator.validate(responses)
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1


def test_accuracy_validator_empty_responses() -> None:
    validator = AccuracyValidator()
    result = validator.validate([])
    assert result["passed"] is False


def test_drift_validator_returns_drift_score() -> None:
    validator = DriftValidator()
    ref = [{"total_amount": 100, "num_tickets": 2}, {"total_amount": 150, "num_tickets": 3}]
    cur = [{"total_amount": 110, "num_tickets": 2}, {"total_amount": 160, "num_tickets": 3}]
    result = validator.validate(ref, cur)
    assert "drift_score" in result
    assert "passed" in result
    assert 0 <= result["drift_score"] <= 1


def test_drift_validator_insufficient_data() -> None:
    validator = DriftValidator()
    result = validator.validate([], [])
    assert result["passed"] is True
    assert result["drift_score"] == 0


def test_business_validator_fraud_metrics() -> None:
    validator = BusinessValidator(min_fraud_recall=0.9)
    responses = [
        {"is_fraud": True, "blocked": True},
        {"is_fraud": True, "blocked": True},
        {"is_fraud": False, "blocked": False},
    ]
    result = validator.validate(responses)
    assert "fraud_recall" in result
    assert "fraud_blocked_pct" in result
    assert result["fraud_recall"] == 1.0
