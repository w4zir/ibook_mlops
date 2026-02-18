"""
Deterministic seed transaction generation for drift reference and simulator.

Produces Kafka-compatible transaction records (user_id, event_id, total_amount,
timestamp, num_tickets, price_per_ticket) so both the feature pipeline and
simulator drift scenario use the same distribution.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _init_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _generate_events(rng: np.random.Generator, n_events: int) -> pd.DataFrame:
    categories = ["sports", "concerts", "family", "cultural"]
    base_start = datetime.now(timezone.utc) - timedelta(days=30)

    event_ids = np.arange(1, n_events + 1)
    promoters = rng.integers(1, 21, size=n_events)
    event_categories = rng.choice(categories, size=n_events, replace=True)
    start_times = [
        base_start + timedelta(days=int(offset))
        for offset in rng.integers(-10, 20, size=n_events)
    ]
    capacities = rng.integers(500, 5000, size=n_events)
    base_prices = rng.uniform(50, 500, size=n_events)

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "promoter_id": promoters,
            "category": event_categories,
            "start_time": start_times,
            "capacity": capacities,
            "base_price": base_prices,
        }
    )


def _generate_users(rng: np.random.Generator, n_users: int) -> pd.DataFrame:
    user_ids = np.arange(1, n_users + 1)
    loyalty_tier = rng.choice(["bronze", "silver", "gold", "platinum"], size=n_users)
    signup_offset_days = rng.integers(30, 365 * 3, size=n_users)
    signup_dates = [
        datetime.now(timezone.utc) - timedelta(days=int(d)) for d in signup_offset_days
    ]

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "loyalty_tier": loyalty_tier,
            "signup_date": signup_dates,
        }
    )


def _generate_transactions_df(
    rng: np.random.Generator,
    events: pd.DataFrame,
    users: pd.DataFrame,
    n_transactions: int,
    window_hours: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic transactions (same logic as scripts/seed-data.py).

    If window_hours is set, timestamps are constrained to the last N hours
    (for drift scenario so feature pipeline's last-N-h window includes them).
    If None, timestamps span 0-59 days (legacy behavior).
    """
    event_ids = events["event_id"].to_numpy()
    user_ids = users["user_id"].to_numpy()

    chosen_events = rng.choice(event_ids, size=n_transactions, replace=True)
    chosen_users = rng.choice(user_ids, size=n_transactions, replace=True)

    now = datetime.now(timezone.utc)
    if window_hours is not None and window_hours > 0:
        # Last-N-hours window: random minute offset in [0, window_hours * 60)
        max_minutes = max(1, window_hours * 60)
        minute_offsets = rng.integers(0, max_minutes, size=n_transactions)
        timestamps = [now - timedelta(minutes=int(m)) for m in minute_offsets]
    else:
        timestamps = [
            now - timedelta(days=int(d), minutes=int(m))
            for d, m in zip(
                rng.integers(0, 60, size=n_transactions),
                rng.integers(0, 24 * 60, size=n_transactions),
            )
        ]

    base_price_lookup = events.set_index("event_id")["base_price"]
    base_prices = base_price_lookup.reindex(chosen_events).to_numpy()
    surge = rng.uniform(0.9, 1.5, size=n_transactions)
    ticket_prices = base_prices * surge

    quantities = rng.integers(1, 5, size=n_transactions)

    user_tier_lookup = users.set_index("user_id")["loyalty_tier"]
    user_tiers = user_tier_lookup.reindex(chosen_users).to_numpy()
    fraud_base_prob = 0.03
    fraud_boost = np.where(ticket_prices > 300, 0.05, 0.0)
    fraud_penalty = np.where(user_tiers == "platinum", -0.02, 0.0)
    fraud_prob = np.clip(fraud_base_prob + fraud_boost + fraud_penalty, 0.001, 0.3)
    is_fraud = rng.binomial(1, fraud_prob).astype(bool)

    return pd.DataFrame(
        {
            "event_timestamp": timestamps,
            "event_id": chosen_events,
            "user_id": chosen_users,
            "quantity": quantities,
            "ticket_price": ticket_prices,
            "is_fraud": is_fraud,
        }
    )


def generate_seed_transactions(
    seed: int = 42,
    n_events: int = 100,
    n_users: int = 1000,
    n_transactions: int = 5000,
    window_hours: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Generate deterministic seed transactions in Kafka/sink schema.

    Returns list of dicts with: user_id, event_id, total_amount, timestamp,
    num_tickets, price_per_ticket. Used for drift reference (feature pipeline)
    and drift scenario (simulator).

    window_hours: If set (e.g. 24), all timestamps fall within the last N hours.
      Use this for the drift scenario so the feature pipeline's last-24h window
      includes the emitted data. If None, timestamps span 0-59 days (legacy).
    """
    rng = _init_rng(seed)
    events = _generate_events(rng, n_events=n_events)
    users = _generate_users(rng, n_users=n_users)
    df = _generate_transactions_df(
        rng,
        events=events,
        users=users,
        n_transactions=n_transactions,
        window_hours=window_hours,
    )
    df["total_amount"] = (df["quantity"] * df["ticket_price"]).round(2)
    df["num_tickets"] = df["quantity"]
    df["price_per_ticket"] = df["ticket_price"].round(2)
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        ts = row["event_timestamp"]
        if hasattr(ts, "isoformat"):
            timestamp_str = ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        else:
            timestamp_str = pd.Timestamp(ts).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        out.append({
            "user_id": str(row["user_id"]),
            "event_id": int(row["event_id"]),
            "total_amount": float(row["total_amount"]),
            "timestamp": timestamp_str,
            "num_tickets": int(row["num_tickets"]),
            "price_per_ticket": float(row["price_per_ticket"]),
        })
    return out
