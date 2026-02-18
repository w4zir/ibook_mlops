"""Tests for simulator scenarios - setup, lifecycle, validators."""

from __future__ import annotations

import pytest

from simulator.config import UserPersona
from simulator.core.event_generator import EventGenerator
from simulator.core.transaction_generator import TransactionGenerator
from simulator.core.user_generator import UserGenerator
from simulator.scenarios.base_scenario import BaseScenario
from simulator.scenarios.black_friday import BlackFridayScenario
from simulator.scenarios.flash_sale import FlashSaleScenario
from simulator.scenarios.fraud_attack import FraudAttackScenario
from simulator.scenarios.drift import DriftScenario
from simulator.scenarios.mixed import MixedScenario
from simulator.scenarios.normal_traffic import NormalTrafficScenario
from simulator.scenarios.system_degradation import SystemDegradationScenario


def test_event_generator_produces_valid_events() -> None:
    gen = EventGenerator()
    event = gen.generate_event()
    assert "event_id" in event
    assert "name" in event
    assert "category" in event
    assert "capacity" in event
    assert "pricing_tiers" in event
    assert len(event["pricing_tiers"]) == 3
    assert event["pricing_tiers"][0]["tier"] == "VIP"


def test_user_generator_produces_valid_users() -> None:
    gen = UserGenerator()
    user = gen.generate_user()
    assert "user_id" in user
    assert "persona" in user
    assert user["persona"] in ("casual", "enthusiast", "vip", "scalper", "fraudster")


def test_transaction_generator_produces_valid_transactions() -> None:
    event_gen = EventGenerator()
    user_gen = UserGenerator()
    txn_gen = TransactionGenerator()
    event = event_gen.generate_event()
    user = user_gen.generate_user()
    txn = txn_gen.generate_transaction(event, user, UserPersona.CASUAL)
    assert "transaction_id" in txn
    assert "event_id" in txn
    assert "user_id" in txn
    assert "total_amount" in txn
    assert "fraud_indicators" in txn


@pytest.mark.parametrize(
    "scenario_class",
    [
        NormalTrafficScenario,
        FlashSaleScenario,
        FraudAttackScenario,
        DriftScenario,
        SystemDegradationScenario,
        BlackFridayScenario,
    ],
)
def test_scenario_setup_runs_without_error(scenario_class: type) -> None:
    if scenario_class is DriftScenario:
        scenario = scenario_class(drift_level=0.5)
    else:
        scenario = scenario_class()
    scenario.setup()
    assert scenario.results is not None or hasattr(scenario, "users") or hasattr(scenario, "events")


def test_base_scenario_validate_returns_structure() -> None:
    """BaseScenario.validate expects results to contain expected_metrics keys."""
    class ConcreteScenario(BaseScenario):
        def setup(self) -> None:
            self.results["peak_rps"] = 100
            self.results["p99_latency_ms"] = 50
            self.results["error_rate"] = 0.01

        def run(self) -> None:
            pass

        def teardown(self) -> None:
            pass

    scenario = ConcreteScenario(
        name="Concrete",
        description="Test",
        duration_minutes=1,
        expected_metrics={"peak_rps": 100, "p99_latency_ms": 50, "error_rate": 0.01},
    )
    scenario.results = {"peak_rps": 100, "p99_latency_ms": 50, "error_rate": 0.01}
    validation = scenario.validate()
    assert "passed" in validation
    assert "failures" in validation
    assert "metrics" in validation
    assert validation["passed"] is True


def test_normal_traffic_execute_completes() -> None:
    scenario = NormalTrafficScenario()
    result = scenario.execute()
    assert "passed" in result
    assert "duration_seconds" in scenario.results or "peak_rps" in scenario.results


def test_scenario_duration_override() -> None:
    """Execute with duration_override_minutes scales workload."""
    scenario = NormalTrafficScenario()
    result = scenario.execute(duration_override_minutes=2)
    assert "passed" in result
    assert scenario.get_effective_duration_minutes() == 2
    assert "duration_seconds" in scenario.results
    assert "responses" in scenario.results
    assert len(scenario.results["responses"]) >= 50


def test_mixed_scenario_setup_runs_without_error() -> None:
    scenario = MixedScenario(
        scenario_weights={"normal-traffic": 0.5, "flash-sale": 0.5},
        duration_minutes=2,
    )
    scenario.setup()
    assert len(scenario._scenario_instances) >= 1
    assert scenario.results is not None or hasattr(scenario, "_scenario_instances")


def test_mixed_scenario_execute_completes() -> None:
    scenario = MixedScenario(
        scenario_weights={"normal-traffic": 1.0},
        duration_minutes=1,
    )
    result = scenario.execute(duration_override_minutes=1)
    assert "passed" in result
    assert "duration_seconds" in scenario.results or "peak_rps" in scenario.results
    assert "responses" in scenario.results
    assert len(scenario.results["responses"]) >= 1


def test_mixed_scenario_custom_weights() -> None:
    scenario = MixedScenario(
        scenario_weights={"normal-traffic": 70, "fraud-attack": 30},
        duration_minutes=1,
    )
    scenario.setup()
    assert "normal-traffic" in scenario._scenario_instances
    assert "fraud-attack" in scenario._scenario_instances
    scenario.run()
    assert len(scenario.results.get("responses", [])) >= 1


def test_drift_scenario_drift_level_zero_no_drift() -> None:
    """Drift level 0 yields no drift (score ~0)."""
    from simulator.validators.drift_validator import DriftValidator

    scenario = DriftScenario(drift_level=0.0)
    scenario.setup()
    scenario.run()
    scenario.teardown()
    ref = scenario.results.get("reference_data", [])
    cur = scenario.results.get("current_data", [])
    assert len(ref) >= 1 and len(cur) >= 1
    result = DriftValidator().validate(ref, cur)
    assert result["drift_score"] < 0.15, "drift_level=0 should yield near-zero drift score"
    assert result["passed"] is True


def test_drift_scenario_drift_level_high_triggers_validator() -> None:
    """Drift level near 1 yields high drift (score >= 0.30) so pipeline can trigger retrain."""
    from simulator.validators.drift_validator import DriftValidator

    scenario = DriftScenario(drift_level=0.9)
    scenario.setup()
    scenario.run()
    scenario.teardown()
    ref = scenario.results.get("reference_data", [])
    cur = scenario.results.get("current_data", [])
    assert len(ref) >= 1 and len(cur) >= 1
    result = DriftValidator().validate(ref, cur)
    assert result["drift_score"] >= 0.30
    assert result["passed"] is False


def test_drift_scenario_produces_kafka_compatible_payloads() -> None:
    """Drift scenario emits transactions with schema required by sink/feature pipeline."""
    from simulator.core.kafka_emitter import prepare_transaction_for_kafka

    scenario = DriftScenario(drift_level=0.5)
    scenario.setup()
    scenario.run()
    assert "kafka_current_sent" in scenario.results
    cur = scenario.results.get("current_data", [])
    ref = scenario.results.get("reference_data", [])
    assert len(ref) >= 1 and len(cur) >= 1
    for txn in (ref[0], cur[0]):
        payload = prepare_transaction_for_kafka(txn, scenario_tag="test")
        assert "user_id" in payload
        assert "event_id" in payload
        assert "total_amount" in payload
        assert "timestamp" in payload
        assert payload["_scenario"] == "test"


def test_drift_scenario_higher_level_gives_higher_score() -> None:
    """Monotonicity: higher drift_level should not reduce drift score (same seed)."""
    scenario_low = DriftScenario(drift_level=0.3)
    scenario_low.setup()
    scenario_low.run()
    scenario_low.teardown()
    scenario_high = DriftScenario(drift_level=0.8)
    scenario_high.setup()
    scenario_high.run()
    scenario_high.teardown()
    score_low = scenario_low.results.get("drift_score_detected", 0.0)
    score_high = scenario_high.results.get("drift_score_detected", 0.0)
    assert score_high >= score_low - 0.05, "Higher drift_level should yield at least as high drift score"

