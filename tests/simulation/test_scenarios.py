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
from simulator.scenarios.gradual_drift import GradualDriftScenario
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
        GradualDriftScenario,
        SystemDegradationScenario,
        BlackFridayScenario,
    ],
)
def test_scenario_setup_runs_without_error(scenario_class: type) -> None:
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
