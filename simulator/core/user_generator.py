"""Generate user profiles by persona (casual, enthusiast, VIP, scalper, fraudster)."""

import random
from typing import Any, Dict, List

from simulator.config import UserPersona, config

try:
    from faker import Faker
    _fake = Faker()
except Exception:
    _fake = None


def _uuid4_short() -> str:
    if _fake:
        return _fake.uuid4()[:8]
    return f"{random.getrandbits(32):08x}"


class UserGenerator:
    """Generate realistic user profiles by persona."""

    def __init__(self) -> None:
        self.countries = ["SA", "AE", "EG", "US", "GB"]

    def generate_user(
        self,
        persona: UserPersona | None = None,
        user_id_override: str | None = None,
    ) -> Dict[str, Any]:
        """Generate a single user profile."""
        if persona is None:
            personas = list(config.user_personas_distribution.keys())
            probs = list(config.user_personas_distribution.values())
            persona = random.choices(personas, weights=probs, k=1)[0]

        uid = user_id_override or f"user_{_uuid4_short()}"
        behavior = config.user_behaviors[persona]

        country = "SA" if persona != UserPersona.FRAUDSTER else random.choice(self.countries)
        if _fake:
            name = _fake.name()
            email = _fake.email()
        else:
            name = f"User_{uid}"
            email = f"{uid}@example.com"

        return {
            "user_id": uid,
            "persona": persona.value,
            "name": name,
            "email": email,
            "country": country,
            "lifetime_purchases": max(0, int(random.gauss(behavior.purchase_frequency * 12, 5))),
            "created_at": _fake.date_time_this_year().isoformat() if _fake else "2024-01-01T00:00:00",
        }

    def generate_batch(
        self,
        count: int = 1000,
        persona_distribution: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of users with optional persona distribution."""
        if persona_distribution is None:
            persona_distribution = {p.value: config.user_personas_distribution[p] for p in config.user_personas_distribution}

        personas = list(persona_distribution.keys())
        weights = [persona_distribution[p] for p in personas]
        users = []
        seen_ids: set[str] = set()
        for _ in range(count):
            persona_str = random.choices(personas, weights=weights, k=1)[0]
            persona = UserPersona(persona_str)
            user = self.generate_user(persona=persona)
            while user["user_id"] in seen_ids:
                user = self.generate_user(persona=persona)
            seen_ids.add(user["user_id"])
            users.append(user)
        return users
