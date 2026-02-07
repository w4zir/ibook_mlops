"""Run scenarios locally (in-process or via HTTP)."""

from typing import Any, Dict, Type

from simulator.scenarios.base_scenario import BaseScenario


class LocalRunner:
    """Execute scenarios locally - in-process or against a live API."""

    def __init__(self, scenario_class: Type[BaseScenario]) -> None:
        self.scenario_class = scenario_class

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Instantiate scenario, execute, return validation result."""
        scenario = self.scenario_class()
        return scenario.execute()

    def run_setup_only(self) -> None:
        """Run only setup (e.g. for dry-run)."""
        scenario = self.scenario_class()
        scenario.setup()
