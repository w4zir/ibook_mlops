"""Command-line interface for Ibook MLOps Simulator."""

import logging
from pathlib import Path

import click

from simulator.scenarios.black_friday import BlackFridayScenario
from simulator.scenarios.flash_sale import FlashSaleScenario
from simulator.scenarios.fraud_attack import FraudAttackScenario
from simulator.scenarios.gradual_drift import GradualDriftScenario
from simulator.scenarios.normal_traffic import NormalTrafficScenario
from simulator.scenarios.system_degradation import SystemDegradationScenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCENARIOS = {
    "flash-sale": FlashSaleScenario,
    "normal-traffic": NormalTrafficScenario,
    "fraud-attack": FraudAttackScenario,
    "gradual-drift": GradualDriftScenario,
    "system-degradation": SystemDegradationScenario,
    "black-friday": BlackFridayScenario,
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


@cli.command()
@click.argument("scenario_name", type=click.Choice(list(SCENARIOS.keys())))
@click.option("--output", "-o", default="report.html", help="Output report file")
@click.option("--dry-run", is_flag=True, help="Validate setup without running")
def run(scenario_name: str, output: str, dry_run: bool) -> None:
    """Run a specific scenario."""
    click.echo(f"\nRunning scenario: {scenario_name}")
    scenario_class = SCENARIOS[scenario_name]
    scenario = scenario_class()
    if dry_run:
        click.echo("DRY RUN: Setup only")
        scenario.setup()
        click.echo("Setup successful")
        return
    with click.progressbar(length=100, label="Running") as bar:
        results = scenario.execute()
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
        report_gen = ReportGenerator()
        report_gen.generate_html(scenario, results, output)
        click.echo(f"\nReport saved: {output}")
    except Exception as e:
        logger.warning("Could not generate report: %s", e)


@cli.command("run-all")
@click.option("--output-dir", "-o", default="reports/", help="Output directory")
def run_all(output_dir: str) -> None:
    """Run all scenarios (test suite)."""
    click.echo("Running full test suite...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for name, scenario_class in SCENARIOS.items():
        click.echo(f"\n{'='*60}")
        click.echo(f"Scenario: {name}")
        click.echo("=" * 60)
        scenario = scenario_class()
        result = scenario.execute()
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


if __name__ == "__main__":
    cli()
