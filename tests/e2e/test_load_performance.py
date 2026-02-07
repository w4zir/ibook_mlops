"""
Locust load test definitions for BentoML services.

Run headless:
  locust -f tests/e2e/test_load_performance.py --headless -u 50 -r 10 -t 30s --host http://localhost:7001

Run with UI:
  locust -f tests/e2e/test_load_performance.py --host http://localhost:7001
"""

from __future__ import annotations

import os

try:
    from locust import HttpUser, task
except ImportError:
    HttpUser = object  # type: ignore[misc, assignment]
    def task(weight=None):  # noqa: D103
        def deco(f):
            return f
        return deco

HOST_FRAUD = os.environ.get("LOCUST_FRAUD_HOST", "http://localhost:7001")
HOST_PRICING = os.environ.get("LOCUST_PRICING_HOST", "http://localhost:7002")


class FraudDetectionUser(HttpUser):  # type: ignore[valid-type,misc]
    """Locust user that hits the fraud detection API."""

    host = HOST_FRAUD

    @task(3)
    def predict(self) -> None:
        self.client.post(
            "/predict",
            json={
                "requests": [
                    {"user_id": 1, "event_id": 2, "amount": 100.0},
                ]
            },
            name="/predict",
        )


class DynamicPricingUser(HttpUser):  # type: ignore[valid-type,misc]
    """Locust user that hits the dynamic pricing API."""

    host = HOST_PRICING

    @task(2)
    def recommend(self) -> None:
        self.client.post(
            "/recommend",
            json={
                "requests": [
                    {"event_id": 1, "current_price": 100.0},
                ]
            },
            name="/recommend",
        )


def test_load_performance_module_defines_users() -> None:
    """Ensure Locust user classes are defined for use with locust -f."""
    assert FraudDetectionUser.host == HOST_FRAUD
    assert DynamicPricingUser.host == HOST_PRICING
