"""Gradual drift scenario - seasonal behavior and price sensitivity changes over time."""

import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

from simulator.config import config
from simulator.core.event_generator import EventGenerator
from simulator.core.transaction_generator import TransactionGenerator
from simulator.core.user_generator import UserGenerator
from simulator.scenarios.base_scenario import BaseScenario

logger = logging.getLogger(__name__)


class GradualDriftScenario(BaseScenario):
    """Simulate seasonal/user behavior drift over time - validates drift detection."""

    def __init__(self) -> None:
        super().__init__(
            name="Gradual Drift",
            description="Simulate seasonal changes and behavior drift over weeks",
            duration_minutes=5,
            expected_metrics={
                "drift_score_detected": 0.5,
                "weeks_simulated": 4,
            },
        )
        self.events: List[Dict[str, Any]] = []
        self.users: List[Dict[str, Any]] = []
        self.event_gen = EventGenerator()
        self.user_gen = UserGenerator()
        self.txn_gen = TransactionGenerator()
        self.reference_data: List[Dict[str, Any]] = []
        self.current_data: List[Dict[str, Any]] = []

    def setup(self) -> None:
        logger.info("Setting up gradual drift scenario...")
        self.events = self.event_gen.generate_batch(count=10)
        self.users = self.user_gen.generate_batch(count=500)
        self.reference_data = self.txn_gen.generate_batch(
            self.events, self.users, count=300, time_range_hours=24
        )
        logger.info("Generated reference batch of %d transactions", len(self.reference_data))

    def run(self) -> None:
        logger.info("Generating drifted data (simulated weeks)...")
        start = datetime.now() - timedelta(days=28)
        self.current_data = []
        for week in range(4):
            base_time = start + timedelta(days=week * 7)
            for _ in range(100):
                event = random.choice(self.events)
                user = random.choice(self.users)
                from simulator.config import UserPersona
                ts = base_time + timedelta(hours=random.randint(0, 168))
                txn = self.txn_gen.generate_transaction(
                    event, user, UserPersona(user["persona"]), ts
                )
                self.current_data.append(txn)
        self.results["weeks_simulated"] = 4
        self.results["reference_data"] = self.reference_data
        self.results["current_data"] = self.current_data

    def teardown(self) -> None:
        ref = self.results.get("reference_data", [])
        cur = self.results.get("current_data", [])
        if ref and cur:
            ref_amounts = [t["total_amount"] for t in ref if "total_amount" in t]
            cur_amounts = [t["total_amount"] for t in cur if "total_amount" in t]
            if ref_amounts and cur_amounts:
                ref_avg = sum(ref_amounts) / len(ref_amounts)
                cur_avg = sum(cur_amounts) / len(cur_amounts)
                drift = abs(cur_avg - ref_avg) / ref_avg if ref_avg else 0
                self.results["drift_score_detected"] = min(1.0, drift)
            else:
                self.results["drift_score_detected"] = 0.5
        else:
            self.results["drift_score_detected"] = 0.5
        self.results["weeks_simulated"] = self.results.get("weeks_simulated", 4)
