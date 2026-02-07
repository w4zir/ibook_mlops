"""Generate realistic ticket purchase transactions with fraud indicators."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

from simulator.config import UserPersona, config

try:
    from faker import Faker
    _fake = Faker()
except Exception:
    _fake = None


def _uuid4_short(length: int = 12) -> str:
    if _fake:
        return _fake.uuid4()[:length]
    return f"{random.getrandbits(32):08x}{random.getrandbits(32):08x}"[:length]


def _ipv4() -> str:
    if _fake:
        return _fake.ipv4()
    return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


class TransactionGenerator:
    """Generate realistic ticket purchase transactions."""

    def __init__(self) -> None:
        self.payment_methods = {"card": 0.60, "wallet": 0.25, "bank_transfer": 0.15}
        self.device_types = ["mobile", "desktop", "tablet"]
        self.browsers = ["Chrome", "Safari", "Firefox", "Edge"]

    def generate_transaction(
        self,
        event: Dict[str, Any],
        user: Dict[str, Any],
        persona: UserPersona,
        timestamp: datetime | None = None,
    ) -> Dict[str, Any]:
        """Generate a single transaction."""
        if timestamp is None:
            timestamp = datetime.now()

        behavior = config.user_behaviors[persona]

        num_tickets = max(1, int(np.random.normal(
            behavior.avg_tickets_per_purchase,
            max(0.1, behavior.avg_tickets_per_purchase * 0.3),
        )))

        tier_probs = self._get_tier_probabilities(persona)
        tiers = [t["tier"] for t in event["pricing_tiers"]]
        tier = np.random.choice(tiers, p=tier_probs[: len(tiers)])

        tier_info = next(t for t in event["pricing_tiers"] if t["tier"] == tier)
        price_per_ticket = float(tier_info["price"])

        if random.random() < 0.5:
            price_adjustment = np.random.uniform(-0.15, 0.25)
            price_per_ticket *= 1 + price_adjustment

        total_amount = price_per_ticket * num_tickets
        is_abandoned = random.random() < behavior.cart_abandonment_rate

        methods = list(self.payment_methods.keys())
        probs = list(self.payment_methods.values())
        payment_method = np.random.choice(methods, p=probs)

        device_type = random.choice(self.device_types)
        browser = random.choice(self.browsers)
        user_agent = f"{browser}/{random.randint(90, 120)}.0 ({device_type})"

        if behavior.fraud_probability > 0.5:
            ip_country = random.choice(["SA", "US", "RU", "CN", "NG"])
        else:
            ip_country = "SA"

        fraud_indicators = self._calculate_fraud_indicators(
            user, behavior, timestamp, payment_method, ip_country
        )

        is_fraud = behavior.fraud_probability > random.random()
        transaction: Dict[str, Any] = {
            "transaction_id": f"txn_{_uuid4_short()}",
            "event_id": event["event_id"],
            "user_id": user["user_id"],
            "timestamp": timestamp.isoformat(),
            "num_tickets": num_tickets,
            "tier": tier,
            "price_per_ticket": round(price_per_ticket, 2),
            "total_amount": round(total_amount, 2),
            "payment_method": payment_method,
            "payment_status": "abandoned" if is_abandoned else "pending",
            "device_type": device_type,
            "browser": browser,
            "user_agent": user_agent,
            "ip_address": _ipv4(),
            "ip_country": ip_country,
            "session_id": f"sess_{_uuid4_short(8)}",
            "fraud_indicators": fraud_indicators,
            "is_fraud": is_fraud,
            "persona": persona.value,
            "is_bot": behavior.bot_likelihood > random.random(),
        }

        if not is_abandoned:
            if is_fraud:
                status = np.random.choice(
                    ["failed", "blocked", "completed"],
                    p=[0.5, 0.3, 0.2],
                )
                transaction["payment_status"] = status
            else:
                transaction["payment_status"] = "completed"

        return transaction

    def _get_tier_probabilities(self, persona: UserPersona) -> List[float]:
        if persona == UserPersona.VIP:
            return [0.7, 0.2, 0.1]
        if persona == UserPersona.ENTHUSIAST:
            return [0.2, 0.5, 0.3]
        return [0.05, 0.25, 0.7]

    def _calculate_fraud_indicators(
        self,
        user: Dict[str, Any],
        behavior: Any,
        timestamp: datetime,
        payment_method: str,
        ip_country: str,
    ) -> Dict[str, Any]:
        velocity_score = random.random() * behavior.fraud_probability * 10
        device_mismatch = random.random() < behavior.fraud_probability
        geo_mismatch = ip_country != "SA" and ip_country != user.get("country", "SA")
        hour = timestamp.hour
        time_anomaly = (hour < 4 or hour > 23) and behavior.fraud_probability > 0.5
        risk_score = round(
            (velocity_score / 10 * 0.3 + device_mismatch * 0.2 + geo_mismatch * 0.3 + time_anomaly * 0.2),
            2,
        )
        return {
            "velocity_score": round(velocity_score, 2),
            "device_mismatch": device_mismatch,
            "geo_mismatch": geo_mismatch,
            "time_anomaly": time_anomaly,
            "risk_score": risk_score,
        }

    def generate_batch(
        self,
        events: List[Dict[str, Any]],
        users: List[Dict[str, Any]],
        count: int = 1000,
        time_range_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Generate batch of transactions over time period."""
        transactions = []
        start_time = datetime.now() - timedelta(hours=time_range_hours)
        for _ in range(count):
            event = random.choice(events)
            user = random.choice(users)
            persona = UserPersona(user["persona"])
            random_seconds = random.randint(0, time_range_hours * 3600)
            timestamp = start_time + timedelta(seconds=random_seconds)
            transactions.append(self.generate_transaction(event, user, persona, timestamp))
        return sorted(transactions, key=lambda x: x["timestamp"])
