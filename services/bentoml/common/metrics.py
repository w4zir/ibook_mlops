from __future__ import annotations

"""
Prometheus metrics helpers for BentoML services.

These are intentionally thin so they can be imported from any service module
without creating circular dependencies.  BentoML will expose the metrics via
its own ``/metrics`` endpoint as long as these collectors are imported.

**Important**: ``prometheus_client`` must NOT be imported at module level.
BentoML 1.x sets up its own multiprocessing-safe Prometheus registry and
asserts that ``prometheus_client`` has not yet been loaded when it does so.
All metric objects are therefore created lazily on first use.
"""

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover – type hints only
    from prometheus_client import Counter as _Counter, Histogram as _Histogram

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_REQUEST_COUNTER: _Counter | None = None
_REQUEST_LATENCY_SECONDS: _Histogram | None = None
_ERROR_COUNTER: _Counter | None = None


def _get_request_counter() -> _Counter:
    global _REQUEST_COUNTER
    if _REQUEST_COUNTER is None:
        from prometheus_client import Counter

        _REQUEST_COUNTER = Counter(
            "bentoml_requests_total",
            "Total number of requests handled by BentoML services.",
            labelnames=("service", "endpoint", "http_status"),
        )
    return _REQUEST_COUNTER


def _get_request_latency() -> _Histogram:
    global _REQUEST_LATENCY_SECONDS
    if _REQUEST_LATENCY_SECONDS is None:
        from prometheus_client import Histogram

        _REQUEST_LATENCY_SECONDS = Histogram(
            "bentoml_request_latency_seconds",
            "Latency of BentoML service requests in seconds.",
            labelnames=("service", "endpoint"),
        )
    return _REQUEST_LATENCY_SECONDS


def _get_error_counter() -> _Counter:
    global _ERROR_COUNTER
    if _ERROR_COUNTER is None:
        from prometheus_client import Counter

        _ERROR_COUNTER = Counter(
            "bentoml_request_errors_total",
            "Total number of errors raised by BentoML services.",
            labelnames=("service", "endpoint"),
        )
    return _ERROR_COUNTER


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def record_request(service: str, endpoint: str, http_status: int) -> None:
    _get_request_counter().labels(service=service, endpoint=endpoint, http_status=str(http_status)).inc()


def record_error(service: str, endpoint: str) -> None:
    _get_error_counter().labels(service=service, endpoint=endpoint).inc()


def observe_latency(service: str, endpoint: str, duration_seconds: float) -> None:
    _get_request_latency().labels(service=service, endpoint=endpoint).observe(duration_seconds)


@contextmanager
def track_latency(service: str, endpoint: str) -> Iterator[None]:
    """
    Context manager to conveniently measure request latency:

    with track_latency("fraud", "/predict"):
        ...
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        observe_latency(service, endpoint, duration)


__all__ = [
    "record_request",
    "record_error",
    "observe_latency",
    "track_latency",
]

