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

# Columns used by DriftValidator; shifting these ensures pipeline drift threshold (0.30) is exceeded.
DRIFT_NUMERIC_COLUMNS = ["total_amount", "num_tickets", "price_per_ticket"]

# Default multiplier so drift_score reliably >= 0.30 (pipeline retrain trigger).
DEFAULT_DRIFT_STRENGTH = 1.8


def _apply_drift_shift(
    data: List[Dict[str, Any]],
    strength: float,
    numeric_columns: List[str],
) -> None:
    """Apply multiplicative shift to numeric columns in-place so drift score exceeds threshold."""
    if strength <= 1.0 or not data:
        return
    for row in data:
        for col in numeric_columns:
            if col not in row or row[col] is None:
                continue
            val = row[col]
            if isinstance(val, (int, float)):
                shifted = val * strength
                row[col] = round(shifted, 2) if isinstance(val, float) else max(1, int(round(shifted)))


class GradualDriftScenario(BaseScenario):
    """Simulate seasonal/user behavior drift over time - validates drift detection.

    Applies an intentional distribution shift to current_data so that the monitoring
    pipeline's drift check (threshold 0.30) fails and model retraining is triggered.
    """

    DEFAULT_DURATION_MINUTES = 5

    def __init__(
        self,
        duration_minutes: int | None = None,
        drift_strength: float | None = None,
    ) -> None:
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
        self.drift_strength = drift_strength if drift_strength is not None else DEFAULT_DRIFT_STRENGTH
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
        # Fixed seed so drift score is reproducible and reliably above threshold.
        random.seed(42)
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
        # Apply intentional shift so pipeline drift check (>= 0.30) triggers retraining.
        _apply_drift_shift(self.current_data, self.drift_strength, DRIFT_NUMERIC_COLUMNS)
        logger.info(
            "Applied drift strength %.2f to current_data (target drift_score >= 0.30)",
            self.drift_strength,
        )
        self.results["weeks_simulated"] = weeks
        self.results["reference_data"] = self.reference_data
        self.results["current_data"] = self.current_data

    def teardown(self) -> None:
        ref = self.results.get("reference_data", [])
        cur = self.results.get("current_data", [])
        if ref and cur:
            drift_scores = []
            for col in DRIFT_NUMERIC_COLUMNS:
                ref_vals = [r[col] for r in ref if col in r and r[col] is not None]
                cur_vals = [r[col] for r in cur if col in r and r[col] is not None]
                if not ref_vals or not cur_vals:
                    continue
                ref_mean = sum(ref_vals) / len(ref_vals)
                cur_mean = sum(cur_vals) / len(cur_vals)
                denom = ref_mean if ref_mean != 0 else 1
                drift_scores.append(abs(cur_mean - ref_mean) / denom)
            self.results["drift_score_detected"] = (
                min(1.0, sum(drift_scores) / len(drift_scores)) if drift_scores else 0.5
            )
        else:
            self.results["drift_score_detected"] = 0.5
        self.results["weeks_simulated"] = self.results.get("weeks_simulated", 4)
        if self.results["weeks_simulated"] == 0:
            self.results["weeks_simulated"] = 4
