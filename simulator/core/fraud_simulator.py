"""Simulate coordinated fraud patterns (credential stuffing, card testing, bot scalping)."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

from simulator.config import FraudPattern, UserPersona, config
from simulator.core.transaction_generator import TransactionGenerator


class FraudSimulator:
    """Simulate coordinated fraud attacks using configured fraud patterns."""

    def __init__(self, transaction_generator: TransactionGenerator | None = None) -> None:
        self.txn_gen = transaction_generator or TransactionGenerator()

    def generate_attack_batch(
        self,
        pattern_name: str,
        event: Dict[str, Any],
        users: List[Dict[str, Any]],
        count: int = 100,
        timestamp: datetime | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of transactions following a specific fraud pattern."""
        pattern = next((p for p in config.fraud_patterns if p.name == pattern_name), None)
        if pattern is None:
            pattern = config.fraud_patterns[0]

        if timestamp is None:
            timestamp = datetime.now()

        fraudsters = [u for u in users if u.get("persona") == "fraudster"]
        if not fraudsters:
            fraudsters = users

        transactions = []
        for i in range(count):
            user = random.choice(fraudsters)
            txn = self.txn_gen.generate_transaction(event, user, UserPersona.FRAUDSTER, timestamp)
            txn["fraud_pattern"] = pattern.name
            txn["is_fraud"] = True
            # Simulate success rate: some get blocked
            if random.random() > pattern.success_rate:
                txn["payment_status"] = random.choice(["blocked", "failed"])
            else:
                txn["payment_status"] = "completed"
            transactions.append(txn)
            timestamp = timestamp + timedelta(seconds=random.uniform(0.01, 0.5))
        return transactions

    def generate_mixed_attack(
        self,
        event: Dict[str, Any],
        users: List[Dict[str, Any]],
        duration_seconds: int = 60,
        attacks_per_second: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Generate a mixed fraud attack using all configured patterns over a time window."""
        start = datetime.now()
        all_txns: List[Dict[str, Any]] = []
        pattern_names = [p.name for p in config.fraud_patterns]
        total_attacks = int(duration_seconds * attacks_per_second)
        per_pattern = max(1, total_attacks // len(pattern_names))

        for name in pattern_names:
            batch_start = start + timedelta(seconds=len(all_txns) * 0.1)
            batch = self.generate_attack_batch(name, event, users, count=per_pattern, timestamp=batch_start)
            all_txns.extend(batch)

        return sorted(all_txns, key=lambda x: x["timestamp"])
