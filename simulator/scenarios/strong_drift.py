"""Strong drift scenario - guarantees drift above threshold to trigger model retraining."""

import logging
from typing import Any, Dict, List

from simulator.scenarios.gradual_drift import GradualDriftScenario

logger = logging.getLogger(__name__)

# Drift strength chosen so drift_score is clearly >= 0.30 (pipeline retrain threshold).
STRONG_DRIFT_STRENGTH = 2.0


class StrongDriftScenario(GradualDriftScenario):
    """Simulate strong data drift so the next pipeline run will trigger model training.

    Uses the same setup as GradualDriftScenario but applies a larger distribution
    shift (e.g. 1.6x) to current_data so that:
    - DriftValidator reports drift_score >= 0.30 (failed check).
    - ml_monitoring_pipeline / feature_engineering_pipeline will trigger retraining.
    """

    DEFAULT_DURATION_MINUTES = 3

    def __init__(self, duration_minutes: int | None = None) -> None:
        super().__init__(
            duration_minutes=duration_minutes or self.DEFAULT_DURATION_MINUTES,
            drift_strength=STRONG_DRIFT_STRENGTH,
        )
        self.name = "Strong Drift"
        self.description = (
            "Strong distribution shift so drift score >= 0.30 and pipeline triggers model retraining"
        )
        self.expected_metrics = {
            "drift_score_detected": 0.5,
            "weeks_simulated": 4,
        }
