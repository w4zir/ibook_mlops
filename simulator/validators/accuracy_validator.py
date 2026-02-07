"""Validate model performance - precision, recall, F1."""

from typing import Any, Dict, List


class AccuracyValidator:
    """Check fraud detection (or other) accuracy metrics."""

    def validate(
        self,
        responses: List[Dict[str, Any]],
        actual_key: str = "is_fraud",
        predicted_key: str = "blocked",
        score_key: str = "fraud_score",
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Compute precision, recall, F1 from responses."""
        if not responses:
            return {
                "passed": False,
                "reason": "no responses",
                "precision": 0,
                "recall": 0,
                "f1": 0,
            }
        y_true = [1 if r.get(actual_key) else 0 for r in responses]
        y_pred = [
            1 if (r.get(predicted_key) or (r.get(score_key, 0) or 0) > threshold)
            else 0
            for r in responses
        ]
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        passed = recall >= 0.90 and precision >= 0.85
        return {
            "passed": passed,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
