from __future__ import annotations

"""
Prometheus metrics helpers for BentoML services.

These are intentionally thin so they can be imported from any service module
without creating circular dependencies. BentoML will expose the metrics via
its own `/metrics` endpoint as long as these collectors are imported.
"""

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Histogram


REQUEST_COUNTER = Counter(
    "bentoml_requests_total",
    "Total number of requests handled by BentoML services.",
    labelnames=("service", "endpoint", "http_status"),
)

REQUEST_LATENCY_SECONDS = Histogram(
    "bentoml_request_latency_seconds",
    "Latency of BentoML service requests in seconds.",
    labelnames=("service", "endpoint"),
)

ERROR_COUNTER = Counter(
    "bentoml_request_errors_total",
    "Total number of errors raised by BentoML services.",
    labelnames=("service", "endpoint"),
)


def record_request(service: str, endpoint: str, http_status: int) -> None:
    REQUEST_COUNTER.labels(service=service, endpoint=endpoint, http_status=str(http_status)).inc()


def record_error(service: str, endpoint: str) -> None:
    ERROR_COUNTER.labels(service=service, endpoint=endpoint).inc()


def observe_latency(service: str, endpoint: str, duration_seconds: float) -> None:
    REQUEST_LATENCY_SECONDS.labels(service=service, endpoint=endpoint).observe(duration_seconds)


@contextmanager
def track_latency(service: str, endpoint: str) -> Iterator[None]:
    """
    Context manager to conveniently measure request latency:

    with track_latency(\"fraud\", \"/predict\"):
        ...
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        observe_latency(service, endpoint, duration)


__all__ = [
    "REQUEST_COUNTER",
    "REQUEST_LATENCY_SECONDS",
    "ERROR_COUNTER",
    "record_request",
    "record_error",
    "observe_latency",
    "track_latency",
]

