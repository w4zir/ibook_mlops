"""Shared helpers for Airflow DAGs (workspace paths, retry config)."""

from __future__ import annotations

import os
from pathlib import Path

# Resolve workspace root from common package (PYTHONPATH points at workspace in Docker).
import common  # noqa: E402

WORKSPACE_ROOT = Path(common.__file__).resolve().parents[1]


def get_workspace_data_path(*parts: str) -> Path:
    """Return path under workspace data directory (e.g. data/processed/feast)."""
    return WORKSPACE_ROOT / "data" / Path(*parts)


def get_retries_for_dag(dag_id: str, default: int) -> int:
    """
    Number of task retries for a DAG from env.

    Reads DAG-specific env {DAG_ID}_RETRIES (e.g. FEATURE_ENGINEERING_PIPELINE_RETRIES),
    then AIRFLOW_TASK_RETRIES, then uses default. Set to 0 to fail immediately.
    """
    key = dag_id.upper().replace("-", "_") + "_RETRIES"
    val = (
        (os.environ.get(key) or os.environ.get("AIRFLOW_TASK_RETRIES") or "")
        .strip()
    )
    if not val:
        return default
    return int(val)
