from datetime import datetime, timezone

import pytest


def _build_hourly(target_date: str, temps_f: list[float]) -> dict:
    return {
        "hourly": {
            "time": [f"{target_date}T{hour:02d}:00" for hour in range(len(temps_f))],
            "temperature_2m": [((t - 32.0) * 5.0 / 9.0) for t in temps_f],
        },
        "timezone": "America/New_York",
    }


def test_matcher_attaches_tail_probability_when_models_exist():
    """When TailCalibrationManager is passed and returns a result, the
    opportunity dict gains our_probability_tail, edge_tail, and
    tail_calibration_source alongside the existing our_probability."""
    from src.matcher import match_kalshi_markets

    class _StubTailManager:
        def calibrate_tail_probability(self, city, market_type, direction, is_bucket, raw_prob):
            # Tail calibration shifts probability by +0.1
            return {
                "calibrated_prob": min(0.999, max(0.001, float(raw_prob) + 0.1)),
                "source": "logistic+isotonic",
            }

    forecasts = {"Austin": _build_hourly("2030-01-01", [75.0] * 24)}
    markets = [{
        "city": "Austin",
        "type": "high",
        "threshold": 80.0,
        "ticker": "KXHIGHTAUS-30JAN01-T80",
        "title": "Will the maximum temperature be >80F on Jan 1, 2030?",
        "yes_sub_title": "80 or above",
        "last_price": 0.50,
        "yes_bid": 0.48,
        "yes_ask": 0.52,
        "close_time": "2030-01-01T23:59:00+00:00",
        "volume_24h": 5000,
    }]
    now = datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc)

    opps = match_kalshi_markets(
        forecasts, markets,
        min_edge=0.0,
        uncertainty_std_f=2.0,
        calibration_manager=None,
        tail_calibration_manager=_StubTailManager(),
        hrrr_blend_horizon_hours=18.0,
        now_utc=now,
    )
    assert len(opps) == 1
    opp = opps[0]
    assert "our_probability_tail" in opp
    assert opp["our_probability_tail"] == pytest.approx(opp["our_probability"] + 0.1, abs=0.01)
    assert opp["tail_calibration_source"] == "logistic+isotonic"
    assert "edge_tail" in opp
    # edge_tail = our_probability_tail - market_price
    assert opp["edge_tail"] == pytest.approx(opp["our_probability_tail"] - opp["market_price"], abs=1e-4)


def test_matcher_omits_tail_fields_when_manager_is_none():
    """Without a TailCalibrationManager, the opportunity dict has no
    tail-related fields (exact backward-compatibility with pre-Task-7)."""
    from src.matcher import match_kalshi_markets

    forecasts = {"Austin": _build_hourly("2030-01-01", [75.0] * 24)}
    markets = [{
        "city": "Austin", "type": "high", "threshold": 80.0,
        "ticker": "KXHIGHTAUS-30JAN01-T80",
        "title": "Will the maximum temperature be >80F on Jan 1, 2030?",
        "yes_sub_title": "80 or above",
        "last_price": 0.50, "yes_bid": 0.48, "yes_ask": 0.52,
        "close_time": "2030-01-01T23:59:00+00:00",
        "volume_24h": 5000,
    }]
    now = datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc)

    opps = match_kalshi_markets(
        forecasts, markets, min_edge=0.0, uncertainty_std_f=2.0,
        calibration_manager=None, tail_calibration_manager=None,
        hrrr_blend_horizon_hours=18.0, now_utc=now,
    )
    assert len(opps) == 1
    assert "our_probability_tail" not in opps[0]
    assert "edge_tail" not in opps[0]
    assert "tail_calibration_source" not in opps[0]


def test_matcher_omits_tail_fields_when_manager_returns_none():
    """When TailCalibrationManager is passed but returns None (no model),
    the opportunity dict has no tail-related fields."""
    from src.matcher import match_kalshi_markets

    class _NullTailManager:
        def calibrate_tail_probability(self, *args, **kwargs):
            return None

    forecasts = {"Austin": _build_hourly("2030-01-01", [75.0] * 24)}
    markets = [{
        "city": "Austin", "type": "high", "threshold": 80.0,
        "ticker": "KXHIGHTAUS-30JAN01-T80",
        "title": "Will the maximum temperature be >80F on Jan 1, 2030?",
        "yes_sub_title": "80 or above",
        "last_price": 0.50, "yes_bid": 0.48, "yes_ask": 0.52,
        "close_time": "2030-01-01T23:59:00+00:00",
        "volume_24h": 5000,
    }]
    now = datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc)

    opps = match_kalshi_markets(
        forecasts, markets, min_edge=0.0, uncertainty_std_f=2.0,
        calibration_manager=None, tail_calibration_manager=_NullTailManager(),
        hrrr_blend_horizon_hours=18.0, now_utc=now,
    )
    assert len(opps) == 1
    assert "our_probability_tail" not in opps[0]
