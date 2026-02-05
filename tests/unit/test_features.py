import pytest

pytest.importorskip("feast")

from feast import FileSource

from services.feast.feature_repo import (
    event,
    event_historical_metrics,
    event_realtime_metrics,
    user,
    user_purchase_behavior,
)


def test_entities_and_feature_views_exist(minimal_local_env):
    # Basic smoke checks on entity and feature view names.
    assert event.name == "event"
    assert user.name == "user"

    assert event_realtime_metrics.name == "event_realtime_metrics"
    assert event_historical_metrics.name == "event_historical_metrics"
    assert user_purchase_behavior.name == "user_purchase_behavior"


def test_local_feature_views_use_filesource(minimal_local_env):
    # In local mode, we should be using FileSource-backed sources.
    assert isinstance(event_realtime_metrics.source, FileSource)
    assert isinstance(event_historical_metrics.source, FileSource)
    assert isinstance(user_purchase_behavior.source, FileSource)


