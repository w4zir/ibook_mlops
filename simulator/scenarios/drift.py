"""
Unified drift scenario: single configurable drift_level in [0, 1].

Generates seed-compatible data; drift_level 0 = no drift (same as seed),
drift_level 1 = shift so drift score approaches 1. Emits only the drifted
batch to Kafka so the feature pipeline's last-24h target sees it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from simulator.core.kafka_emitter import emit_transactions_to_kafka
from simulator.scenarios.base_scenario import BaseScenario

logger = logging.getLogger(__name__)

# Columns used by DriftValidator and for applying drift.
DRIFT_NUMERIC_COLUMNS = ["total_amount", "num_tickets", "price_per_ticket"]

# Default sizes for deterministic generation (single data type).
DEFAULT_SEED = 42
DEFAULT_N_EVENTS = 50
DEFAULT_N_USERS = 500
DEFAULT_N_TRANSACTIONS = 2000
# Align with feature pipeline's FEATURE_RAW_EVENTS_HOURS (default 24).
DEFAULT_WINDOW_HOURS = 24


def _apply_drift_level(
    data: List[Dict[str, Any]],
    drift_level: float,
    numeric_columns: List[str],
) -> None:
    """
    Apply multiplicative shift so that drift_level 0 -> no change, drift_level 1 -> high drift.
    multiplier = 1 + drift_level so 0 -> 1.0, 1 -> 2.0 (drift score ~1).
    """
    if drift_level <= 0 or not data:
        return
    multiplier = 1.0 + float(drift_level)
    for row in data:
        for col in numeric_columns:
            if col not in row or row[col] is None:
                continue
            val = row[col]
            if isinstance(val, (int, float)):
                shifted = val * multiplier
                row[col] = round(shifted, 2) if isinstance(val, float) else max(1, int(round(shifted)))


def _compute_drift_score(
    reference_data: List[Dict[str, Any]],
    current_data: List[Dict[str, Any]],
    numeric_columns: List[str],
) -> float:
    """Same formula as DriftValidator: mean shift per column, averaged, capped at 1."""
    if not reference_data or not current_data:
        return 0.0
    drift_scores: List[float] = []
    for col in numeric_columns:
        ref_vals = [r[col] for r in reference_data if col in r and r[col] is not None]
        cur_vals = [r[col] for r in current_data if col in r and r[col] is not None]
        if not ref_vals or not cur_vals:
            continue
        ref_mean = sum(ref_vals) / len(ref_vals)
        cur_mean = sum(cur_vals) / len(cur_vals)
        denom = ref_mean if ref_mean != 0 else 1.0
        drift_scores.append(abs(cur_mean - ref_mean) / denom)
    return min(1.0, sum(drift_scores) / len(drift_scores)) if drift_scores else 0.0


class DriftScenario(BaseScenario):
    """
    Single drift scenario with configurable drift_level in [0, 1].

    - drift_level 0: no drift (same distribution as seed).
    - drift_level 1: shift so drift score is close to 1.
    Generates one batch of data (seed-like + optional shift) and emits to Kafka.
    """

    DEFAULT_DURATION_MINUTES = 2

    def __init__(
        self,
        drift_level: float = 0.5,
        duration_minutes: int | None = None,
        seed: int = DEFAULT_SEED,
        n_events: int = DEFAULT_N_EVENTS,
        n_users: int = DEFAULT_N_USERS,
        n_transactions: int = DEFAULT_N_TRANSACTIONS,
        window_hours: int | None = DEFAULT_WINDOW_HOURS,
    ) -> None:
        drift_level = max(0.0, min(1.0, float(drift_level)))
        mins = duration_minutes if duration_minutes is not None else self.DEFAULT_DURATION_MINUTES
        super().__init__(
            name="Drift",
            description="Configurable data drift (0=no drift, 1=maximum drift); seed-compatible distribution.",
            duration_minutes=mins,
            expected_metrics={"drift_score_detected": drift_level},
        )
        self.drift_level = drift_level
        self.seed = seed
        self.n_events = n_events
        self.n_users = n_users
        self.n_transactions = n_transactions
        self.window_hours = window_hours
        self.reference_data: List[Dict[str, Any]] = []
        self.current_data: List[Dict[str, Any]] = []

    def setup(self) -> None:
        logger.info(
            "Drift scenario setup: drift_level=%.2f seed=%s n_transactions=%s window_hours=%s",
            self.drift_level,
            self.seed,
            self.n_transactions,
            self.window_hours,
        )
        self.reference_data = []
        self.current_data = []

    def run(self) -> None:
        try:
            from common.seed_transactions import generate_seed_transactions
        except ImportError:
            logger.warning("common.seed_transactions not available; using empty data.")
            self.results["kafka_current_sent"] = 0
            self.results["drift_score_detected"] = 0.0
            return

        # One batch: seed-compatible transactions (reference = no drift).
        # window_hours so timestamps fall in feature pipeline's last-N-h window.
        self.reference_data = generate_seed_transactions(
            seed=self.seed,
            n_events=self.n_events,
            n_users=self.n_users,
            n_transactions=self.n_transactions,
            window_hours=self.window_hours,
        )
        if not self.reference_data:
            logger.warning("No seed transactions generated.")
            self.results["kafka_current_sent"] = 0
            self.results["drift_score_detected"] = 0.0
            return

        # Current = copy of reference then apply drift (in-place on copy).
        self.current_data = [dict(r) for r in self.reference_data]
        _apply_drift_level(
            self.current_data,
            self.drift_level,
            DRIFT_NUMERIC_COLUMNS,
        )
        self.results["reference_data"] = self.reference_data
        self.results["current_data"] = self.current_data

        # Emit only current (drifted) batch to Kafka so last-24h target sees it.
        cur_sent = emit_transactions_to_kafka(self.current_data, scenario_tag="drift")
        self.results["kafka_current_sent"] = cur_sent
        logger.info(
            "Emitted %d drifted transactions to Kafka (drift_level=%.2f).",
            cur_sent,
            self.drift_level,
        )

    def teardown(self) -> None:
        ref = self.results.get("reference_data", [])
        cur = self.results.get("current_data", [])
        score = _compute_drift_score(ref, cur, DRIFT_NUMERIC_COLUMNS)
        self.results["drift_score_detected"] = score
        logger.info("Drift score detected: %.4f (drift_level=%.2f).", score, self.drift_level)
