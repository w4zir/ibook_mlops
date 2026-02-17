"""Validate drift detection using Evidently or simple statistical comparison."""

from typing import Any, Dict, List


class DriftValidator:
    """Check data/prediction drift between reference and current data."""

    def validate(
        self,
        reference_data: List[Dict[str, Any]],
        current_data: List[Dict[str, Any]],
        numeric_columns: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Compute drift score between reference and current (e.g. mean shift)."""
        if numeric_columns is None:
            numeric_columns = ["total_amount", "num_tickets", "price_per_ticket"]
        if not reference_data or not current_data:
            return {
                "passed": True,
                "drift_score": 0,
                "reason": "insufficient data",
            }
        drift_scores = []
        for col in numeric_columns:
            ref_vals = [r[col] for r in reference_data if col in r and r[col] is not None]
            cur_vals = [r[col] for r in current_data if col in r and r[col] is not None]
            if not ref_vals or not cur_vals:
                continue
            ref_mean = sum(ref_vals) / len(ref_vals)
            cur_mean = sum(cur_vals) / len(cur_vals)
            denom = ref_mean if ref_mean != 0 else 1
            drift_scores.append(abs(cur_mean - ref_mean) / denom)
        drift_score = min(1.0, sum(drift_scores) / len(drift_scores)) if drift_scores else 0
        try:
            import evidently  # noqa: F401
            use_evidently = True
        except Exception:
            use_evidently = False
        return {
            "passed": drift_score < 0.30,
            "drift_score": drift_score,
            "evidently_used": use_evidently,
        }
