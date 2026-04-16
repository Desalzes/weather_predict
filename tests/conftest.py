"""Pytest configuration for opportunity_log tests."""

from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest


# Mock datetime for the opportunity_log tests that expect 2026-04-17
@pytest.fixture(autouse=True)
def mock_datetime_for_opportunity_log(request):
    """Auto-patch datetime.utcnow() to return 2026-04-17 for opportunity_log tests."""
    if "test_opportunity_log" in request.node.nodeid:
        patcher = patch("src.opportunity_log.datetime")
        mock_datetime = patcher.start()

        # Set up the mock to return proper datetime objects
        fixed_dt = datetime(2026, 4, 17, 12, 0, 0)
        mock_datetime.utcnow.return_value = fixed_dt
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        yield
        patcher.stop()
    else:
        yield
