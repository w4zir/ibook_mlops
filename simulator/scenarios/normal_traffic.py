"""Normal traffic scenario - typical daily operations, 100-500 RPS, low fraud."""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from simulator.config import UserPersona, config
from simulator.core.event_generator import EventGenerator
from simulator.core.transaction_generator import TransactionGenerator
from simulator.core.user_generator import UserGenerator
from simulator.scenarios.base_scenario import BaseScenario

logger = logging.getLogger(__name__)


def _synthetic_response(transaction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": 200,
        "latency_ms": random.uniform(10, 80),
        "fraud_score": 0.2 if not transaction.get("is_fraud") else 0.9,
        "blocked": transaction.get("is_fraud", False),
        "is_fraud": transaction.get("is_fraud", False),
    }


class NormalTrafficScenario(BaseScenario):
    """Simulate typical daily operations - 100-500 RPS, ~3% fraud."""

    def __init__(self) -> None:
        super().__init__(
            name="Normal Traffic",
            description="Simulate typical daily operations with baseline load",
            duration_minutes=5,
            expected_metrics={
                "peak_rps": 300,
                "p99_latency_ms": 200,
                "error_rate": 0.01,
            },
        )
        self.events: List[Dict[str, Any]] = []
        self.users: List[Dict[str, Any]] = []
        self.event_gen = EventGenerator()
        self.user_gen = UserGenerator()
        self.txn_gen = TransactionGenerator()

    def setup(self) -> None:
        logger.info("Setting up normal traffic scenario...")
        self.events = self.event_gen.generate_batch(count=20)
        self.users = self.user_gen.generate_batch(count=2000)
        logger.info("Generated %d events, %d users", len(self.events), len(self.users))

    def run(self) -> None:
        logger.info("Generating normal traffic...")
        transactions = self.txn_gen.generate_batch(
            self.events, self.users, count=1500, time_range_hours=1
        )
        responses = [_synthetic_response(t) for t in transactions]
        for i, t in enumerate(transactions):
            responses[i]["is_fraud"] = t.get("is_fraud", False)
        self.results["responses"] = responses
        self.results["transactions"] = transactions

    def teardown(self) -> None:
        responses = self.results.get("responses", [])
        if not responses:
            return
        latencies = [r["latency_ms"] for r in responses if "latency_ms" in r]
        if latencies:
            self.results["p99_latency_ms"] = float(np.percentile(latencies, 99))
        else:
            self.results["p99_latency_ms"] = 0
        errors = [r for r in responses if r.get("status", 200) >= 400]
        self.results["error_rate"] = len(errors) / len(responses)
        duration = self.results.get("duration_seconds", 1)
        self.results["peak_rps"] = len(responses) / duration
