from __future__ import annotations

"""
Models and bandit logic for the dynamic pricing BentoML service.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field


class PricingRequest(BaseModel):
    event_id: int = Field(..., description="Event identifier.")
    current_price: float = Field(..., ge=0.0, description="Current ticket price.")
    arm_hint: str | None = Field(
        default=None,
        description="Optional hint for which pricing strategy arm to evaluate.",
    )


class PricingResponse(BaseModel):
    recommended_price: float = Field(..., ge=0.0)
    arm: str = Field(..., description="Chosen pricing strategy arm.")
    lower_confidence: float = Field(..., ge=0.0)
    upper_confidence: float = Field(..., ge=0.0)


class PricingBatchRequest(BaseModel):
    requests: List[PricingRequest]


class PricingBatchResponse(BaseModel):
    predictions: List[PricingResponse]


@dataclass
class ThompsonSamplingBandit:
    """
    Simple multi-armed bandit using Thompson Sampling with Beta priors.

    Each arm maintains success/failure counts based on observed rewards.
    """

    arms: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {"baseline": (1.0, 1.0), "aggressive": (1.0, 1.0)})

    def choose_arm(self) -> str:
        samples = {
            arm: np.random.beta(alpha, beta)
            for arm, (alpha, beta) in self.arms.items()
        }
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float) -> None:
        alpha, beta = self.arms.get(arm, (1.0, 1.0))
        # Interpret reward in [0, 1] as success probability.
        reward_clamped = float(np.clip(reward, 0.0, 1.0))
        alpha += reward_clamped
        beta += 1.0 - reward_clamped
        self.arms[arm] = (alpha, beta)


def rule_based_fallback(current_price: float) -> float:
    """
    Simple deterministic fallback pricing strategy.
    """
    # For now, keep it extremely simple: small uplift capped to +10%.
    return float(current_price * 1.05)


__all__ = [
    "PricingRequest",
    "PricingResponse",
    "PricingBatchRequest",
    "PricingBatchResponse",
    "ThompsonSamplingBandit",
    "rule_based_fallback",
]

