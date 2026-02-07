"""Validate latency SLAs (p50, p95, p99)."""

from typing import Any, Dict, List


class LatencyValidator:
    """Check response latencies against SLA targets."""

    def __init__(
        self,
        p50_max_ms: float = 50.0,
        p95_max_ms: float = 150.0,
        p99_max_ms: float = 200.0,
    ) -> None:
        self.p50_max_ms = p50_max_ms
        self.p95_max_ms = p95_max_ms
        self.p99_max_ms = p99_max_ms

    def validate(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate latency metrics from a list of responses with 'latency_ms'."""
        latencies = [r["latency_ms"] for r in responses if "latency_ms" in r]
        if not latencies:
            return {
                "passed": False,
                "reason": "no latency data",
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
            }
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        p50 = sorted_lat[int(0.50 * (n - 1))] if n else 0
        p95 = sorted_lat[int(0.95 * (n - 1))] if n else 0
        p99 = sorted_lat[int(0.99 * (n - 1))] if n else 0
        passed = p50 <= self.p50_max_ms and p95 <= self.p95_max_ms and p99 <= self.p99_max_ms
        return {
            "passed": passed,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "p50_max_ms": self.p50_max_ms,
            "p95_max_ms": self.p95_max_ms,
            "p99_max_ms": self.p99_max_ms,
        }
