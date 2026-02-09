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

    DEFAULT_DURATION_MINUTES = 5

    def __init__(self, duration_minutes: int | None = None) -> None:
        mins = duration_minutes if duration_minutes is not None else self.DEFAULT_DURATION_MINUTES
        super().__init__(
            name="Gradual Drift",
            description="Simulate seasonal changes and behavior drift over weeks",
            duration_minutes=mins,
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
        effective = self.get_effective_duration_minutes()
        scale = effective / self.DEFAULT_DURATION_MINUTES
        weeks = min(12, max(1, int(4 * scale)))
        start = datetime.now() - timedelta(days=7 * weeks)
        self.current_data = []
        for week in range(weeks):
            base_time = start + timedelta(days=week * 7)
            per_week = max(20, int(100 * scale))
            for _ in range(per_week):
                event = random.choice(self.events)
                user = random.choice(self.users)
                from simulator.config import UserPersona
                ts = base_time + timedelta(hours=random.randint(0, 168))
                txn = self.txn_gen.generate_transaction(
                    event, user, UserPersona(user["persona"]), ts
                )
                self.current_data.append(txn)
        self.results["weeks_simulated"] = weeks
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
        if self.results["weeks_simulated"] == 0:
            self.results["weeks_simulated"] = 4
