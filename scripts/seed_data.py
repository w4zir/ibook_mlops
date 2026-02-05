from __future__ import annotations

"""
Compatibility wrapper for the `seed-data.py` script.

The original generator lives in `seed-data.py` so that it can be invoked as
`python scripts/seed-data.py`. This module makes the same functionality
importable as `scripts.seed_data` for tests and notebooks.
"""

from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING


_LEGACY_PATH = Path(__file__).with_name("seed-data.py")
_loader = SourceFileLoader("scripts_seed_data_legacy", str(_LEGACY_PATH))
_mod: ModuleType = _loader.load_module()  # type: ignore[deprecated]

generate_synthetic_data = _mod.generate_synthetic_data  # type: ignore[attr-defined]

if TYPE_CHECKING:  # pragma: no cover - type checkers only
    from typing import Tuple
    import pandas as pd

    def generate_synthetic_data(
        n_events: int = 100,
        n_users: int = 1000,
        n_transactions: int = 10_000,
        seed: int = 42,
    ) -> "Tuple[pd.DataFrame, pd.DataFrame]":
        ...


__all__ = ["generate_synthetic_data"]

