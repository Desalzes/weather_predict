"""Tests for matcher module."""

from datetime import datetime, timezone

import pytest
from src.matcher import match_kalshi_markets


def test_matcher_uses_ngr_when_flag_enabled(monkeypatch):
    """When use_ngr_calibration=True, matcher should call predict_distribution and use its sigma."""

    class StubManager:
        def __init__(self):
            self.calls = []

        def predict_distribution(self, city, market_type, forecast_f, spread_f, lead_h, doy):
            self.calls.append((city, market_type, forecast_f, spread_f, lead_h, doy))
            return forecast_f + 0.5, 2.5, "ngr"  # mu, sigma, source

        def calibrate_probability(self, city, market_type, raw_prob):
            return raw_prob, "raw"

    stub = StubManager()
    forecasts = {
        "Austin": {
            "timezone": "America/Chicago",
            "hourly": {
                "time": ["2026-04-17T00:00"] + [f"2026-04-17T{h:02d}:00" for h in range(1, 24)],
                "temperature_2m": [20.0] + [22.0 + (h % 3) for h in range(1, 24)],
            },
        }
    }
    markets = [{
        "city": "Austin",
        "type": "high",
        "threshold": 75.0,
        "ticker": "KXHIGHTAUS-26APR17-T75",
        "title": "Will the maximum temperature be  >75° on Apr 17, 2026?",
        "yes_sub_title": "75 or above",
        "last_price": 0.50,
        "yes_bid": 0.48,
        "yes_ask": 0.52,
        "close_time": "2026-04-17T23:59:00+00:00",
        "volume_24h": 5000,
    }]

    opps = match_kalshi_markets(
        forecasts,
        markets,
        min_edge=0.0,
        uncertainty_std_f=2.0,
        calibration_manager=stub,
        use_ngr_calibration=True,
        now_utc=datetime(2026, 4, 17, 0, 0, tzinfo=timezone.utc),
    )

    assert len(stub.calls) == 1
    city, mtype, _, _, lead_h, doy = stub.calls[0]
    assert city == "Austin"
    assert mtype == "high"
    assert lead_h > 0  # calculated from close_time
    assert 1 <= doy <= 366
    # Matcher used the sigma from predict_distribution, not ensemble clamp
    assert opps[0]["uncertainty_std_f"] == 2.5
    assert opps[0]["forecast_calibration_source"] == "ngr"
