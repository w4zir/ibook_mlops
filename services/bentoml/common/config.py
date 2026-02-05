from __future__ import annotations

"""
Lightweight configuration helpers for BentoML services.

These helpers intentionally **wrap** the central `common.config` module instead
of introducing a second source of truth. BentoML services should call
`get_bentoml_settings()` to discover how to talk to MLflow, Feast, and
Prometheus.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from common.config import AppConfig, get_config


@dataclass(frozen=True)
class BentoMLSettings:
    """
    Resolved settings for BentoML services.

    All values ultimately come from environment variables consumed by
    `common.config.AppConfig`, with a few BentoML-specific overrides:

    - `BENTOML_MODEL_NAME` / `BENTOML_MODEL_STAGE` control which registered
      MLflow model/version should be served by fraud-related services.
    - `BENTOML_ENABLE_PROMETHEUS` can be set to ``\"0\"`` / ``\"false\"`` to
      disable Prometheus metrics export in lightweight environments.
    """

    app: AppConfig
    model_name: str
    model_stage: str
    enable_prometheus: bool

    @property
    def tracking_uri(self) -> str:
        return self.app.mlflow.tracking_uri

    @property
    def feast_offline_store(self) -> str:
        return self.app.feast.offline_store


@lru_cache(maxsize=1)
def get_bentoml_settings(
    *, model_name_override: Optional[str] = None, model_stage_override: Optional[str] = None
) -> BentoMLSettings:
    """
    Return cached BentoML settings derived from the global `AppConfig`.

    Callers can optionally override the model name or stage (mainly useful in
    tests) without affecting other processes.
    """
    app_cfg = get_config()

    model_name = model_name_override or _env_or_default("BENTOML_MODEL_NAME", "fraud_detection")
    model_stage = model_stage_override or _env_or_default("BENTOML_MODEL_STAGE", "Production")

    enable_prometheus_str = _env_or_default("BENTOML_ENABLE_PROMETHEUS", "1").strip().lower()
    enable_prometheus = enable_prometheus_str not in {"0", "false", "no"}

    return BentoMLSettings(
        app=app_cfg,
        model_name=model_name,
        model_stage=model_stage,
        enable_prometheus=enable_prometheus,
    )


def _env_or_default(key: str, default: str) -> str:
    import os

    return os.getenv(key, default)


__all__ = ["BentoMLSettings", "get_bentoml_settings"]

