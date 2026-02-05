from __future__ import annotations

"""
Helpers for resolving and loading models from the MLflow registry for
use inside BentoML services.
"""

from dataclasses import dataclass
from typing import Any, Optional

try:  # pragma: no cover - import guard for environments without MLflow installed
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    MlflowClient = Any  # type: ignore[misc,assignment]

from services.bentoml.common.config import BentoMLSettings, get_bentoml_settings


@dataclass(frozen=True)
class ResolvedModel:
    """
    Represents a concrete model selection from the MLflow registry.
    """

    name: str
    version: str
    stage: str
    run_id: str
    source: str

    @property
    def model_uri(self) -> str:
        """
        URI that can be passed to `mlflow.pyfunc.load_model` or
        framework-specific loaders.
        """
        # Prefer the canonical `models:/name/version` form; this works even if
        # the stage is not \"Production\".
        return f"models:/{self.name}/{self.version}"


def _get_client(settings: Optional[BentoMLSettings] = None) -> MlflowClient:
    if mlflow is None:  # pragma: no cover - defensive
        raise RuntimeError("MLflow is not installed; cannot create MlflowClient.")
    cfg = settings or get_bentoml_settings()
    mlflow.set_tracking_uri(cfg.tracking_uri)
    return MlflowClient()


def resolve_latest_model(
    *,
    model_name: Optional[str] = None,
    stage: Optional[str] = None,
    settings: Optional[BentoMLSettings] = None,
) -> ResolvedModel:
    """
    Resolve the most appropriate model version from the MLflow registry.

    Preference order:
    1. A version in the requested `stage` (default: settings.model_stage).
    2. The highest numeric version, regardless of stage.
    """
    cfg = settings or get_bentoml_settings()
    name = model_name or cfg.model_name
    desired_stage = (stage or cfg.model_stage).strip()

    client = _get_client(cfg)
    versions = list(client.search_model_versions(f"name = '{name}'"))
    if not versions:
        raise LookupError(f"No MLflow model versions found for registered model '{name}'.")

    chosen = None

    # 1) Prefer the desired stage, if any version is in that stage.
    if desired_stage:
        staged = [v for v in versions if (v.current_stage or "").lower() == desired_stage.lower()]
        if staged:
            # If multiple are in the same stage, use the highest version.
            chosen = max(staged, key=lambda v: int(v.version))

    # 2) Fallback: highest version regardless of stage.
    if chosen is None:
        chosen = max(versions, key=lambda v: int(v.version))

    return ResolvedModel(
        name=name,
        version=str(chosen.version),
        stage=chosen.current_stage or "",
        run_id=chosen.run_id,
        source=chosen.source,
    )


__all__ = ["ResolvedModel", "resolve_latest_model"]

