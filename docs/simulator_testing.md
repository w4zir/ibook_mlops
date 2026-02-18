# Simulator Testing Guide

How to test simulator scenarios using CLI, pytest, Docker Compose, Makefile, and Locust, with expected results and validation criteria.

---

## Overview

The **Ibook Event Simulator** runs predefined scenarios to stress-test the MLOps platform. Each scenario exercises different conditions (normal load, flash sale, fraud, drift, degradation). This document describes how to run and validate them across tools.

| Scenario | What it tests | Expected outcome |
|----------|----------------|------------------|
| **normal-traffic** | Baseline daily operations, 100–500 RPS, ~3% fraud | p99 &lt; 200 ms, error rate &lt; 1% |
| **flash-sale** | Mega-event launch, traffic spike, bot traffic | Peak RPS, fraud detection ~90%, latency within SLA |
| **fraud-attack** | Coordinated fraud (credential stuffing, card testing, scalping) | Fraud recall ≥ 90%, precision ≥ 85% |
| **drift** | Configurable drift level (0 = no drift, 1 = max); seed-compatible data | Drift score scales with --drift-level; ≥ 0.30 triggers retrain |
| **system-degradation** | Partial failures, circuit breakers, fallback | Error rate ~5%, fallback usage measurable |
| **black-friday** | Extreme sustained load | Peak RPS, p99 &lt; 200 ms, error rate &lt; 2% |
| **mix** | Weighted combination of the above | Combined metrics within relaxed thresholds |

---

## Testing with CLI

From repo root (with `pip install -e .` and venv active):

### List scenarios

```bash
python -m simulator.cli list-scenarios
```

Expected: Table of scenario names, descriptions, and default duration (minutes).

### Run a single scenario

```bash
python -m simulator.cli run normal-traffic -o reports/normal-traffic-report.html
python -m simulator.cli run flash-sale -o reports/flash-sale-report.html
```

Optional `--duration` (minutes) scales the workload:

```bash
python -m simulator.cli run normal-traffic --duration 10 -o reports/normal-10min.html
```

Expected: Progress bar, then "PASSED" or "FAILED" with failure lines; HTML report written.

### Dry run (setup only)

```bash
python -m simulator.cli run normal-traffic --dry-run
```

Expected: "DRY RUN: Setup only", "Setup successful", no traffic.

### Run all scenarios (test suite)

```bash
python -m simulator.cli run-all -o reports/
```

Optional: `--duration 5` to use 5 minutes for each scenario.

Expected: One block per scenario, then "SUMMARY" with "Passed: X/Y" and "ALL SCENARIOS PASSED" or "N SCENARIOS FAILED".

### Mix mode

Run a weighted mix of scenarios for a configurable duration:

```bash
python -m simulator.cli mix --duration 30 --scenarios "normal-traffic:40,flash-sale:20,fraud-attack:20,system-degradation:10,black-friday:10" -o reports/mix-report.html
```

Weights are normalized. Default is 40% normal, 20% flash-sale, 20% fraud-attack, 10% system-degradation, 10% black-friday.

Expected: Progress bar, PASSED/FAILED, report at `reports/mix-report.html`.

### Realtime mode

Stream traffic at a fixed RPS for a wall-clock duration (seconds):

```bash
python -m simulator.cli realtime normal-traffic --duration 60 --rps 100 -o reports/realtime-report.html
```

Expected: Live line updates (elapsed, txns, rps, errors); final summary and report. Use Ctrl+C for graceful stop.

---

## Testing with pytest

Simulation tests live under `tests/simulation/`.

### Run all simulation tests

```bash
python -m pytest tests/simulation/ -v --tb=short
```

Or from repo root:

```bash
make test
```

### What each test validates

- **test_scenarios.py**
  - `test_event_generator_produces_valid_events`: Event has event_id, name, category, capacity, 3 pricing tiers.
  - `test_user_generator_produces_valid_users`: User has user_id, persona in allowed set.
  - `test_transaction_generator_produces_valid_transactions`: Transaction has required fields and fraud_indicators.
  - `test_scenario_setup_runs_without_error`: Each scenario’s `setup()` runs (parametrized over all scenario classes).
  - `test_base_scenario_validate_returns_structure`: Validation returns passed/failures/metrics; matching metrics pass.
  - `test_normal_traffic_execute_completes`: Full execute() completes and returns passed/duration_seconds or peak_rps.
- **test_validators.py**
  - Latency validator: passes under SLA, fails over SLA, handles empty responses.
  - Accuracy validator: computes precision/recall/F1; handles empty.
  - Drift validator: returns drift_score and passed; insufficient data returns 0.
  - Business validator: fraud recall and blocked %.

Expected: All tests pass (green). No Docker or live API required; scenarios use synthetic responses when needed.

---

## Testing with Docker Compose

Run the simulator against the live stack (BentoML fraud API, Airflow, etc.):

### Start stack and simulator overlay

```bash
docker compose -f docker-compose.yml -f docker-compose.simulator.yml up -d
```

### Run a scenario inside the simulator container

```bash
docker compose -f docker-compose.yml -f docker-compose.simulator.yml run --rm simulator python -m simulator.cli run normal-traffic -o /app/reports/out.html
```

Reports are in `./reports/` (mounted from host).

### Point simulator at fraud API

Set `API_BASE_URL` for the simulator service (e.g. `http://bentoml-fraud:7001`). Then run scenarios as above; they will hit the live `/predict` endpoint instead of synthetic responses.

Expected: Same CLI output as local; latency and fraud metrics reflect real API behavior.

---

## Testing with Makefile

Requires `make` and `pip install -e .`:

| Command | Description | Example |
|---------|-------------|--------|
| `make sim-list` | List scenarios | - |
| `make sim-run scenario=normal-traffic` | Run one scenario | `scenario=flash-sale` |
| `make sim-run-all` | Run all scenarios | - |
| `make sim-mix` | Run mix (default 30 min) | `make sim-mix duration=10` |
| `make sim-realtime` | Realtime (default 60s, 100 rps) | `make sim-realtime scenario=normal-traffic duration=120 rps=200` |

Override variables: `duration`, `scenario`, `rps` (for sim-realtime).

---

## Expected results matrix

Validation uses **10% tolerance** on expected metrics.

| Scenario | Metric | Expected | Tolerance | Failure means |
|----------|--------|----------|-----------|----------------|
| normal-traffic | peak_rps | 300 | 10% | Throughput too far from baseline |
| normal-traffic | p99_latency_ms | 200 | 10% | Latency SLA breach |
| normal-traffic | error_rate | 0.01 | 10% | Error rate too high |
| flash-sale | peak_rps | 10000 | 10% | Peak load not reached |
| flash-sale | p99_latency_ms | 200 | 10% | Latency SLA breach |
| flash-sale | error_rate | 0.01 | 10% | Errors during peak |
| flash-sale | fraud_detected_pct | 90 | 10% | Fraud detection under target |
| fraud-attack | fraud_recall | 0.90 | 10% | Too many frauds missed |
| fraud-attack | fraud_precision | 0.85 | 10% | Too many false blocks |
| fraud-attack | p99_latency_ms | 200 | 10% | Latency SLA breach |
| drift | drift_score_detected | 0.5 | 15% | Drift level vs expected (use --drift-level) |
| system-degradation | error_rate | 0.05 | 10% | Error rate out of range |
| system-degradation | fallback_used_pct | 50 | 10% | Fallback behavior unexpected |
| black-friday | peak_rps | 5000 | 10% | Load not sustained |
| black-friday | p99_latency_ms | 200 | 10% | Latency SLA breach |
| black-friday | error_rate | 0.02 | 10% | Error rate too high |
| mix | peak_rps | 500 | 10% | Combined throughput low |
| mix | p99_latency_ms | 200 | 10% | Latency SLA breach |
| mix | error_rate | 0.05 | 10% | Combined error rate high |

---

## Mix mode testing

- **Configure weights**: `--scenarios "normal-traffic:50,fraud-attack:50"` for 50/50.
- **Duration**: `--duration 15` runs 15 minutes of time-sliced rounds.
- **What to check**: Report and validation for combined peak_rps, p99_latency_ms, error_rate; optional fraud metrics if present.

---

## Realtime mode testing

- **Duration**: Wall-clock seconds (`--duration 60` = 1 minute).
- **RPS**: Target requests per second (`--rps 100`).
- **Progress**: One line per second: elapsed, txns, current rps, errors.
- **Graceful stop**: Ctrl+C sets a flag and loop exits after current request; results still aggregated and reported.

---

## Locust integration

The simulator provides `LoadTestRunner` in `simulator.runners.load_test_runner` for building Locust tasks (e.g. fraud and pricing payloads). For full load tests, use `tests/e2e/test_load_performance.py` (Locust-based) or run scenarios with `--duration` and optional live API to stress the stack.

Expected: Locust reports request count, failure rate, latency percentiles; align with the expected results matrix above for SLA checks.

---

## CI/CD integration

To run simulation tests in CI (e.g. GitHub Actions):

1. **Unit/simulation tests** (no stack): `pytest tests/simulation/ -v`
2. **Optional**: Start stack, then run `simulator.cli run-all --duration 2` (short duration) and assert exit code 0 and "ALL SCENARIOS PASSED" in output.
3. **Artifacts**: Upload `reports/` as workflow artifacts for inspection.

Simulation tests are fast when using synthetic responses; with a live API they depend on stack startup and scenario duration.
