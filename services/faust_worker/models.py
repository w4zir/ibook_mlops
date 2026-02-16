"""
Faust Record types for raw.transactions topic (must match simulator transaction schema).
"""

from __future__ import annotations

from faust import Record


class RawTransaction(Record, serializer="json"):
    """One transaction event from the simulator (raw.transactions topic)."""

    transaction_id: str = ""
    event_id: int = 0
    user_id: int = 0
    timestamp: str = ""
    num_tickets: int = 0
    tier: str = ""
    price_per_ticket: float = 0.0
    total_amount: float = 0.0
    payment_method: str = ""
    payment_status: str = ""
    device_type: str = ""
    browser: str = ""
    user_agent: str = ""
    ip_address: str = ""
    ip_country: str = ""
    session_id: str = ""
    is_fraud: bool = False
    persona: str = ""
    is_bot: bool = False
    # Optional nested; we don't need them for aggregates
    fraud_indicators: dict | None = None


class UserAggregate(Record, serializer="json"):
    """Per-user rolling aggregate for real-time features."""

    count: int = 0
    amount_sum: float = 0.0
    event_ids: list = ()  # list of event_id for distinct count; capped in size
