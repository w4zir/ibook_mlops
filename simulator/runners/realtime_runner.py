"""Realtime runner - stream traffic over wall-clock time with live progress and configurable RPS."""

import logging
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from simulator.config import UserPersona, config
from simulator.scenarios.base_scenario import BaseScenario
from simulator.scenarios.normal_traffic import NormalTrafficScenario

from services.kafka.producer import send_raw_transaction

logger = logging.getLogger(__name__)

_should_stop = False


def _synthetic_response(transaction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": 200,
        "latency_ms": random.uniform(10, 120),
        "fraud_score": 0.2 if not transaction.get("is_fraud") else 0.9,
        "blocked": transaction.get("is_fraud", False),
        "is_fraud": transaction.get("is_fraud", False),
    }


def _handler(_signum: int, _frame: Any) -> None:
    global _should_stop
    _should_stop = True


class RealtimeRunner:
    """Run scenario-style traffic in real time over wall-clock duration at configurable RPS."""

    def __init__(
        self,
        scenario_class: Type[BaseScenario] | None = None,
        duration_seconds: int = 60,
        rps: float = 100,
        api_base_url: str | None = None,
        fraud_api_base_url: str | None = None,
        log_features: bool = False,
    ) -> None:
        self.scenario_class = scenario_class or NormalTrafficScenario
        self.duration_seconds = duration_seconds
        self.rps = max(0.1, rps)
        # Keep api_base_url for backwards compatibility / future use.
        self.api_base_url = (api_base_url or config.api_base_url).rstrip("/")
        # Dedicated fraud API base URL pointing at BentoML fraud detection.
        self.fraud_api_base_url = (fraud_api_base_url or config.fraud_api_base_url).rstrip("/")
        self.log_features = log_features
        self.results: Dict[str, Any] = {}
        self._scenario: BaseScenario | None = None
        # Feedback batch for auto-training metrics; flushed every progress interval.
        self._feedback_batch: List[Dict[str, Any]] = []
        self._last_feedback_stats: Optional[Dict[str, Any]] = None
        # Feature log for auto model training when log_features=True.
        self._feature_log: List[Dict[str, Any]] = []

    def _get_events_users_txn_gen(self) -> tuple[List[Dict], List[Dict], Any]:
        """Return (events or [single event], users, txn_gen) from the scenario."""
        s = self._scenario
        if s is None:
            raise RuntimeError("Scenario not setup")
        if hasattr(s, "events") and s.events and hasattr(s, "users") and s.users and hasattr(s, "txn_gen"):
            return s.events, s.users, s.txn_gen
        if hasattr(s, "event") and s.event and hasattr(s, "users") and s.users and hasattr(s, "txn_gen"):
            return [s.event], s.users, s.txn_gen
        raise RuntimeError("Scenario has no events/users/txn_gen for realtime run")

    def run(self) -> Dict[str, Any]:
        """Run real-time traffic for duration_seconds at rps; return validation-like result."""
        global _should_stop
        _should_stop = False
        signal.signal(signal.SIGINT, _handler)
        if sys.platform != "win32":
            try:
                signal.signal(signal.SIGTERM, _handler)
            except Exception:
                pass

        self._scenario = self.scenario_class()
        self._scenario.setup()
        self._feedback_batch = []
        self._feature_log = []
        self._last_feedback_stats = None
        events, users, txn_gen = self._get_events_users_txn_gen()
        responses: List[Dict[str, Any]] = []
        start = time.monotonic()
        last_print = start
        interval = 1.0 / self.rps if self.rps >= 1 else 1.0
        next_send = start
        errors = 0

        while (time.monotonic() - start) < self.duration_seconds and not _should_stop:
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(next_send - now, 0.1))
                continue
            event = random.choice(events)
            user = random.choice(users)
            persona = UserPersona(user["persona"]) if isinstance(user.get("persona"), str) else UserPersona.CASUAL
            txn = txn_gen.generate_transaction(event, user, persona, datetime.now())
            try:
                # Always emit the raw transaction to Kafka as the primary event log.
                # Failures here are logged inside the producer wrapper and do not
                # prevent us from calling the fraud API.
                send_raw_transaction(txn)
            except Exception as e:  # pragma: no cover - best-effort logging
                logger.debug("Failed to send transaction to Kafka: %s", e)
            try:
                resp = self._send_request(txn)
            except Exception as e:
                logger.debug("Request failed: %s", e)
                resp = _synthetic_response(txn)
                # Use a synthetic status code to indicate network/transport failure.
                resp["status"] = 599
                resp["error"] = "transport_error"
                resp["timed_out"] = True
                errors += 1
            resp["is_fraud"] = txn.get("is_fraud", False)
            responses.append(resp)
            status = int(resp.get("status", 200))
            # Treat any non-2xx status as an error (including timeouts/transport failures).
            if status < 200 or status >= 300:
                errors += 1
            # Queue feedback for auto-training metrics (only for successful predict).
            if 200 <= status < 300 and txn.get("user_id") is not None and txn.get("event_id") is not None:
                self._feedback_batch.append({
                    "user_id": txn["user_id"],
                    "event_id": txn["event_id"],
                    "predicted_fraud": resp.get("predicted_is_fraud", resp.get("blocked", False)),
                    "actual_fraud": txn.get("is_fraud", False),
                })
                # Log features for auto model training when flag is set.
                if self.log_features:
                    overrides = txn.get("feature_overrides") or {}
                    self._feature_log.append({
                        "user_id": txn["user_id"],
                        "event_id": txn["event_id"],
                        "amount": float(txn.get("total_amount", 0.0)),
                        "lifetime_purchases": overrides.get("user_purchase_behavior__lifetime_purchases", overrides.get("lifetime_purchases", 0)),
                        "fraud_risk_score": overrides.get("user_purchase_behavior__fraud_risk_score", overrides.get("fraud_risk_score", 0.5)),
                        "fraud_score": resp.get("fraud_score", 0.0),
                        "predicted_is_fraud": resp.get("predicted_is_fraud", False),
                        "actual_fraud": txn.get("is_fraud", False),
                        "timestamp": time.time(),
                    })
            next_send += interval
            if now - last_print >= 1.0:
                elapsed = now - start
                current_rps = len(responses) / elapsed if elapsed > 0 else 0
                # Flush feedback batch to service and get current failure-rate stats for display.
                if self._feedback_batch:
                    stats = self._send_feedback_batch(self._feedback_batch)
                    if stats is not None:
                        self._last_feedback_stats = stats
                    self._feedback_batch.clear()
                # Build progress line: txns/errors plus auto-training metrics when available.
                line = (
                    f"\r  elapsed={elapsed:.0f}s txns={len(responses)} rps={current_rps:.0f} errors={errors}"
                )
                if self._last_feedback_stats is not None:
                    fr = self._last_feedback_stats.get("failure_rate", 0.0)
                    n = self._last_feedback_stats.get("window_samples", 0)
                    line += f" | failure_rate={fr * 100:.1f}% n={n}"
                    if self._last_feedback_stats.get("training_triggered"):
                        line += " [AUTO_TRAINING TRIGGERED]"
                line += "   "
                print(line, end="", flush=True)
                last_print = now

        elapsed_total = time.monotonic() - start
        print()
        # Write feature log to Parquet for auto model training if enabled.
        if self.log_features and self._feature_log:
            self._write_feature_log()
        self.results["responses"] = responses
        self.results["duration_seconds"] = elapsed_total
        if responses:
            import numpy as np
            latencies = [r["latency_ms"] for r in responses if "latency_ms" in r]
            if latencies:
                self.results["p99_latency_ms"] = float(np.percentile(latencies, 99))
                self.results["p50_latency_ms"] = float(np.percentile(latencies, 50))
            else:
                self.results["p99_latency_ms"] = 0
            self.results["error_rate"] = errors / len(responses)
            self.results["peak_rps"] = len(responses) / elapsed_total
        return self.results

    def _write_feature_log(self) -> None:
        """Write accumulated feature log to data/training/realtime_features.parquet (append if exists)."""
        try:
            import pandas as pd

            out_path = Path("data/training/realtime_features.parquet")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            new_df = pd.DataFrame(self._feature_log)
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df
            combined.to_parquet(out_path, index=False)
            logger.info("Wrote %d feature rows to %s (total %d).", len(new_df), out_path, len(combined))
        except Exception as e:
            logger.warning("Failed to write feature log to parquet: %s", e)

    def _build_fraud_payload(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a FraudBatchRequest-compatible payload from a simulator transaction.

        The BentoML fraud detection service expects:
            {
                "requests": [
                    {
                        "user_id": int,
                        "event_id": int,
                        "amount": float,
                        "feature_overrides": { ... } | null
                    }
                ]
            }
        """
        user_id = transaction.get("user_id")
        event_id = transaction.get("event_id")
        # Use total_amount as the transaction amount sent to the fraud model.
        amount = float(transaction.get("total_amount", 0.0))
        feature_overrides = transaction.get("feature_overrides")
        request = {
            "user_id": user_id,
            "event_id": event_id,
            "amount": amount,
        }
        if feature_overrides is not None:
            request["feature_overrides"] = feature_overrides
        return {"requests": [request]}

    def _send_request(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Send one transaction to the BentoML fraud API or return synthetic response."""
        try:
            import json
            import socket
            import urllib.error
            import urllib.request

            payload = self._build_fraud_payload(transaction)
            req = urllib.request.Request(
                f"{self.fraud_api_base_url}/predict",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    body = resp.read().decode()
                    status = getattr(resp, "status", 200)
                latency_ms = (time.perf_counter() - t0) * 1000
                data = json.loads(body) if body else {}
                # Expect a FraudBatchResponse-like payload: {"predictions": [{...}]}
                predictions = data.get("predictions") or []
                first = predictions[0] if predictions else {}
                fraud_score = float(first.get("fraud_score", 0.0))
                predicted_is_fraud = bool(first.get("is_fraud", fraud_score >= 0.7))
                return {
                    "status": status,
                    "latency_ms": latency_ms,
                    "fraud_score": fraud_score,
                    "blocked": fraud_score > 0.7,
                    # Ground-truth label is attached in run(); keep prediction as separate flag.
                    "predicted_is_fraud": predicted_is_fraud,
                }
            except (socket.timeout, urllib.error.URLError) as _e:
                # Treat network and timeout failures as transport-level errors.
                resp = _synthetic_response(transaction)
                resp["status"] = 599
                resp["error"] = "transport_error"
                resp["timed_out"] = True
                return resp
        except Exception:
            # On unexpected exceptions, fall back to a synthetic non-error response so that
            # the simulator can still make progress; the caller will treat non-2xx as errors.
            return _synthetic_response(transaction)

    def _send_feedback_batch(self, items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Send a batch of (predicted, actual) feedback to the fraud service.
        Returns dict with failure_rate, window_samples, training_triggered, or None if request fails.
        """
        if not items:
            return None
        try:
            import json
            import urllib.error
            import urllib.request

            payload = {
                "feedbacks": [
                    {
                        "user_id": it["user_id"],
                        "event_id": it["event_id"],
                        "predicted_fraud": it["predicted_fraud"],
                        "actual_fraud": it["actual_fraud"],
                    }
                    for it in items
                ]
            }
            req = urllib.request.Request(
                f"{self.fraud_api_base_url}/feedback",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
                data = json.loads(body) if body else {}
                return {
                    "failure_rate": float(data.get("failure_rate", 0.0)),
                    "window_samples": int(data.get("window_samples", 0)),
                    "training_triggered": bool(data.get("training_triggered", False)),
                }
        except Exception as e:
            logger.debug("Feedback request failed: %s", e)
            return None
