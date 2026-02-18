"""Command-line interface for Ibook MLOps Simulator."""

import logging
from pathlib import Path

import click

from simulator.scenarios.black_friday import BlackFridayScenario
from simulator.scenarios.drift import DriftScenario
from simulator.scenarios.flash_sale import FlashSaleScenario
from simulator.scenarios.fraud_attack import FraudAttackScenario
from simulator.scenarios.mixed import MixedScenario
from simulator.scenarios.normal_traffic import NormalTrafficScenario
from simulator.scenarios.system_degradation import SystemDegradationScenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENARIOS = {
    "flash-sale": FlashSaleScenario,
    "normal-traffic": NormalTrafficScenario,
    "fraud-attack": FraudAttackScenario,
    "drift": DriftScenario,
    "system-degradation": SystemDegradationScenario,
    "black-friday": BlackFridayScenario,
    "mix": MixedScenario,
}


@click.group()
def cli() -> None:
    """Ibook MLOps Platform Simulator."""
    pass


@cli.command("list-scenarios")
def list_scenarios() -> None:
    """List all available scenarios."""
    click.echo("\nAvailable Scenarios:")
    click.echo("=" * 50)
    for name, scenario_class in SCENARIOS.items():
        instance = scenario_class()
        click.echo(f"\n{name}:")
        click.echo(f"  Description: {instance.description}")
        click.echo(f"  Duration: {instance.duration_minutes} minutes")


def _make_scenario(scenario_name: str, drift_level: float | None, drift_seed: int | None,
                   drift_events: int | None, drift_users: int | None, drift_transactions: int | None,
                   drift_window_hours: int | None = None):
    """Build scenario instance; for drift, pass CLI drift options."""
    scenario_class = SCENARIOS[scenario_name]
    if scenario_name == "drift":
        kwargs = {}
        if drift_level is not None:
            kwargs["drift_level"] = drift_level
        if drift_seed is not None:
            kwargs["seed"] = drift_seed
        if drift_events is not None:
            kwargs["n_events"] = drift_events
        if drift_users is not None:
            kwargs["n_users"] = drift_users
        if drift_transactions is not None:
            kwargs["n_transactions"] = drift_transactions
        if drift_window_hours is not None:
            kwargs["window_hours"] = drift_window_hours
        return scenario_class(**kwargs)
    return scenario_class()


@cli.command()
@click.argument("scenario_name", type=click.Choice(list(SCENARIOS.keys())))
@click.option("--output", "-o", default="report.html", help="Output report file")
@click.option("--dry-run", is_flag=True, help="Validate setup without running")
@click.option("--duration", "-d", type=int, default=None, help="Override duration in minutes")
@click.option("--drift-level", type=float, default=None, help="[drift only] Drift level 0-1 (0=no drift, 1=max). Default 0.5")
@click.option("--drift-seed", type=int, default=None, help="[drift only] Random seed for reproducible data. Default 42")
@click.option("--drift-events", type=int, default=None, help="[drift only] Number of events. Default 50")
@click.option("--drift-users", type=int, default=None, help="[drift only] Number of users. Default 500")
@click.option("--drift-transactions", type=int, default=None, help="[drift only] Number of transactions. Default 2000")
@click.option("--drift-window-hours", type=int, default=None, help="[drift only] Timestamp window hours (e.g. 24). Default 24 to match feature pipeline.")
def run(
    scenario_name: str,
    output: str,
    dry_run: bool,
    duration: int | None,
    drift_level: float | None,
    drift_seed: int | None,
    drift_events: int | None,
    drift_users: int | None,
    drift_transactions: int | None,
    drift_window_hours: int | None,
) -> None:
    """Run a specific scenario. For 'drift', use --drift-level 0-1 and optional --drift-* options."""
    click.echo(f"\nRunning scenario: {scenario_name}")
    scenario = _make_scenario(
        scenario_name, drift_level, drift_seed, drift_events, drift_users, drift_transactions, drift_window_hours
    )
    if dry_run:
        click.echo("DRY RUN: Setup only")
        scenario.setup()
        click.echo("Setup successful")
        return
    with click.progressbar(length=100, label="Running") as bar:
        results = scenario.execute(duration_override_minutes=duration)
        bar.update(100)
    if scenario_name == "drift" and isinstance(scenario, DriftScenario):
        score = scenario.results.get("drift_score_detected")
        if score is not None:
            click.echo(f"Drift score detected: {score:.4f}")
    click.echo("\nResults:")
    click.echo("=" * 50)
    if results["passed"]:
        click.echo(click.style("PASSED", fg="green"))
    else:
        click.echo(click.style("FAILED", fg="red"))
        for failure in results["failures"]:
            click.echo(f"  - {failure}")
    try:
        from simulator.visualizers.report_generator import ReportGenerator
        report_gen = ReportGenerator()
        report_gen.generate_html(scenario, results, output)
        click.echo(f"\nReport saved: {output}")
    except Exception as e:
        logger.warning("Could not generate report: %s", e)


@cli.command("run-all")
@click.option("--output-dir", "-o", default="reports/", help="Output directory")
@click.option("--duration", "-d", type=int, default=None, help="Override duration in minutes for each scenario")
def run_all(output_dir: str, duration: int | None) -> None:
    """Run all scenarios (test suite)."""
    click.echo("Running full test suite...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for name, scenario_class in SCENARIOS.items():
        click.echo(f"\n{'='*60}")
        click.echo(f"Scenario: {name}")
        click.echo("=" * 60)
        scenario = scenario_class()
        result = scenario.execute(duration_override_minutes=duration)
        results[name] = result
        status = "PASSED" if result["passed"] else "FAILED"
        click.echo(f"Result: {status}")
    click.echo("\n" + "=" * 60)
    click.echo("SUMMARY")
    click.echo("=" * 60)
    passed = sum(1 for r in results.values() if r["passed"])
    total = len(results)
    click.echo(f"Passed: {passed}/{total}")
    if passed == total:
        click.echo(click.style("\nALL SCENARIOS PASSED", fg="green", bold=True))
    else:
        click.echo(click.style(f"\n{total - passed} SCENARIOS FAILED", fg="red", bold=True))


DEFAULT_MIX_WEIGHTS = {
    "normal-traffic": 0.40,
    "flash-sale": 0.20,
    "fraud-attack": 0.20,
    "system-degradation": 0.10,
    "black-friday": 0.10,
}


def _parse_scenario_weights(scenarios_str: str) -> dict[str, float]:
    """Parse 'name1:40,name2:20' into {'name1': 0.4, 'name2': 0.2} (normalized)."""
    out: dict[str, float] = {}
    for part in scenarios_str.split(","):
        part = part.strip()
        if ":" in part:
            name, w = part.rsplit(":", 1)
            name, w = name.strip(), w.strip()
            try:
                out[name] = float(w)
            except ValueError:
                logger.warning("Invalid weight for %s: %s", name, w)
        elif part:
            out[part] = 1.0
    total = sum(out.values())
    if total > 0:
        return {k: v / total for k, v in out.items()}
    return out


@cli.command()
@click.option("--duration", "-d", type=int, default=30, help="Duration in minutes")
@click.option(
    "--scenarios",
    "-s",
    default="normal-traffic:40,flash-sale:20,fraud-attack:20,system-degradation:10,black-friday:10",
    help="Comma-separated name:weight (e.g. normal-traffic:40,flash-sale:20)",
)
@click.option("--output", "-o", default="reports/mix-report.html", help="Output report file")
def mix(duration: int, scenarios: str, output: str) -> None:
    """Run mixed scenarios with configurable weights and duration."""
    weights = _parse_scenario_weights(scenarios)
    if not weights:
        click.echo("No valid scenario weights; using default mix.")
        weights = dict(DEFAULT_MIX_WEIGHTS)
    click.echo(f"\nRunning mix scenario (duration={duration} min)")
    scenario = MixedScenario(scenario_weights=weights, duration_minutes=duration)
    with click.progressbar(length=100, label="Running") as bar:
        results = scenario.execute(duration_override_minutes=duration)
        bar.update(100)
    click.echo("\nResults:")
    click.echo("=" * 50)
    if results["passed"]:
        click.echo(click.style("PASSED", fg="green"))
    else:
        click.echo(click.style("FAILED", fg="red"))
        for failure in results["failures"]:
            click.echo(f"  - {failure}")
    try:
        from simulator.visualizers.report_generator import ReportGenerator
        ReportGenerator().generate_html(scenario, results, output)
        click.echo(f"\nReport saved: {output}")
    except Exception as e:
        logger.warning("Could not generate report: %s", e)


@cli.command()
@click.argument("scenario_name", type=click.Choice(list(SCENARIOS.keys())))
@click.option("--duration", "-d", type=int, default=60, help="Duration in seconds (wall-clock)")
@click.option("--rps", "-r", type=float, default=100.0, help="Target requests per second")
@click.option("--output", "-o", default="reports/realtime-report.html", help="Output report file (metrics only)")
def realtime(scenario_name: str, duration: int, rps: float, output: str) -> None:
    """Run real-time traffic for a given duration and RPS."""
    from simulator.runners.realtime_runner import RealtimeRunner
    scenario_class = SCENARIOS[scenario_name]
    if scenario_name == "mix":
        click.echo("Realtime mode uses normal-traffic generators for mix; use normal-traffic for mixed-style data.")
        scenario_class = SCENARIOS["normal-traffic"]
    click.echo(f"\nRealtime: {scenario_name} (duration={duration}s, rps={rps})")
    runner = RealtimeRunner(
        scenario_class=scenario_class,
        duration_seconds=duration,
        rps=rps,
    )
    results = runner.run()
    click.echo(f"Completed: {len(results.get('responses', []))} transactions, peak_rps={results.get('peak_rps', 0):.0f}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    try:
        from simulator.visualizers.report_generator import ReportGenerator
        class FakeScenario:
            def __init__(self, metrics: dict) -> None:
                self.name = "Realtime"
                self.description = f"Realtime {duration}s @ {rps} rps"
                self.results = metrics

        fake_scenario = FakeScenario(results)
        ReportGenerator().generate_html(fake_scenario, {"passed": True, "failures": [], "metrics": {}, **results}, output)
        click.echo(f"Report saved: {output}")
    except Exception as e:
        logger.warning("Could not generate report: %s", e)


if __name__ == "__main__":
    cli()
