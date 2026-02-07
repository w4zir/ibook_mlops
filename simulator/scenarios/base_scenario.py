"""Abstract base class for test scenarios with setup/run/teardown/validate lifecycle."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import logging

logger = logging.getLogger(__name__)


class BaseScenario(ABC):
    """Abstract base class for test scenarios."""

    def __init__(
        self,
        name: str,
        description: str,
        duration_minutes: int,
        expected_metrics: Dict[str, float],
    ) -> None:
        self.name = name
        self.description = description
        self.duration_minutes = duration_minutes
        self.expected_metrics = expected_metrics
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def setup(self) -> None:
        """Setup phase - prepare data, initialize state."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Main execution phase - generate load, simulate scenario."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup phase - collect metrics, generate reports."""
        pass

    def validate(self) -> Dict[str, Any]:
        """Validate results against expected metrics."""
        validation_results: Dict[str, Any] = {
            "scenario": self.name,
            "passed": True,
            "failures": [],
            "metrics": {},
        }
        for metric_name, expected_value in self.expected_metrics.items():
            actual_value = self.results.get(metric_name)
            if actual_value is None:
                validation_results["failures"].append(
                    f"Metric '{metric_name}' not found in results"
                )
                validation_results["passed"] = False
                continue
            denom = expected_value if expected_value != 0 else 1.0
            tolerance = 0.10
            if abs(actual_value - expected_value) / denom > tolerance:
                validation_results["failures"].append(
                    f"{metric_name}: expected {expected_value}, got {actual_value}"
                )
                validation_results["passed"] = False
            validation_results["metrics"][metric_name] = {
                "expected": expected_value,
                "actual": actual_value,
                "passed": abs(actual_value - expected_value) / denom <= tolerance,
            }
        return validation_results

    def execute(self) -> Dict[str, Any]:
        """Execute full scenario lifecycle."""
        logger.info("Starting scenario: %s", self.name)
        self.start_time = datetime.now()
        try:
            self.setup()
            logger.info("Setup complete")
            self.run()
            logger.info("Run complete")
            self.teardown()
            logger.info("Teardown complete")
        except Exception as e:
            logger.error("Scenario failed: %s", e)
            self.results["error"] = str(e)
        finally:
            self.end_time = datetime.now()
            self.results["duration_seconds"] = (
                self.end_time - self.start_time
            ).total_seconds()
        return self.validate()
