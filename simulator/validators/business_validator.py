"""Validate business metrics - revenue, fraud savings, etc."""

from typing import Any, Dict, List


class BusinessValidator:
    """Check revenue and fraud-related business metrics."""

    def __init__(
        self,
        min_fraud_recall: float = 0.95,
        min_revenue_uplift: float = 0.0,
    ) -> None:
        self.min_fraud_recall = min_fraud_recall
        self.min_revenue_uplift = min_revenue_uplift

    def validate(
        self,
        responses: List[Dict[str, Any]],
        transactions: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Validate business metrics from responses and optional transactions."""
        if not responses:
            return {
                "passed": False,
                "reason": "no responses",
                "fraud_blocked_pct": 0,
                "revenue_impact": 0,
            }
        blocked = [r for r in responses if r.get("blocked")]
        actual_fraud = [r for r in responses if r.get("is_fraud")]
        true_positives = len([r for r in blocked if r.get("is_fraud")])
        fraud_recall = true_positives / len(actual_fraud) if actual_fraud else 0
        fraud_blocked_pct = (len(blocked) / len(responses)) * 100
        revenue_impact = 0.0
        if transactions:
            completed = [t for t in transactions if t.get("payment_status") == "completed"]
            total = sum(t.get("total_amount", 0) for t in completed)
            revenue_impact = total / len(completed) if completed else 0
        passed = fraud_recall >= self.min_fraud_recall
        return {
            "passed": passed,
            "fraud_recall": fraud_recall,
            "fraud_blocked_pct": fraud_blocked_pct,
            "revenue_impact": revenue_impact,
        }
