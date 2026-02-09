"""Tests for RealtimeRunner - short duration and output structure."""

from __future__ import annotations

import pytest

from simulator.runners.realtime_runner import RealtimeRunner
from simulator.scenarios.normal_traffic import NormalTrafficScenario


def test_realtime_runner_short_duration() -> None:
    """Run for a few seconds and verify transactions are generated and metrics returned."""
    runner = RealtimeRunner(
        scenario_class=NormalTrafficScenario,
        duration_seconds=2,
        rps=5,
    )
    result = runner.run()
    assert "responses" in result
    assert "duration_seconds" in result
    assert len(result["responses"]) >= 1
    assert result["duration_seconds"] >= 1.0
    assert "peak_rps" in result
    assert "error_rate" in result


def test_realtime_runner_returns_metrics() -> None:
    """Runner returns dict with latency and throughput metrics after run."""
    runner = RealtimeRunner(
        scenario_class=NormalTrafficScenario,
        duration_seconds=1,
        rps=10,
    )
    result = runner.run()
    assert "p99_latency_ms" in result or "responses" in result
    assert "peak_rps" in result
    assert "error_rate" in result
    assert isinstance(result["responses"], list)
