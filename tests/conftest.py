"""Pytest configuration for opportunity_log tests."""

from datetime import datetime, timezone
from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def mock_datetime_for_opportunity_log(request):
    """Freeze UTC clock at 2026-04-17 12:00:00 for opportunity_log tests.

    Patches both `utcnow()` (legacy) and `now(timezone.utc)` paths so tests
    are stable regardless of which call the module uses.
    """
    if "test_opportunity_log" not in request.node.nodeid:
        yield
        return

    fixed_dt_naive = datetime(2026, 4, 17, 12, 0, 0)
    fixed_dt_aware = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)

    patcher = patch("src.opportunity_log.datetime")
    mock_datetime = patcher.start()
    mock_datetime.utcnow.return_value = fixed_dt_naive
    mock_datetime.now.return_value = fixed_dt_aware
    mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
    try:
        yield
    finally:
        patcher.stop()
