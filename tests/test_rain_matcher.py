import pytest


def test_parse_rain_outcome_binary_t0_suffix():
    from src.rain_matcher import parse_rain_outcome

    # The -T0 suffix found in the live Kalshi inventory means "binary any-rain".
    # yes_sub_title observed is "Rain in NYC".
    examples = [
        # (outcome_or_yes_sub_title, ticker, expected)
        ("Rain in NYC", "KXRAINNYC-26APR21-T0", {"threshold_in": 0.01, "market_type": "rain_binary"}),
        ("Yes",         "KXRAINNYC-26APR21-T0", {"threshold_in": 0.01, "market_type": "rain_binary"}),
        ("No",          "KXRAINNYC-26APR21-T0", {"threshold_in": 0.01, "market_type": "rain_binary"}),
        (">= 0.01 in",  "KXRAINNYC-26APR21-T0", {"threshold_in": 0.01, "market_type": "rain_binary"}),
    ]
    for text, ticker, expected in examples:
        parsed = parse_rain_outcome(text, ticker=ticker)
        assert parsed == expected, f"{text!r} ticker={ticker}: {parsed}"


def test_parse_rain_outcome_non_binary_returns_none():
    from src.rain_matcher import parse_rain_outcome

    # Non-binary thresholds (hypothetical -T25, -T100 tickers for >=0.25 in, >=1.00 in)
    # are out of scope for P1 and should return None.
    assert parse_rain_outcome("Rain > 0.25 in", ticker="KXRAINNYC-26APR21-T25") is None
    assert parse_rain_outcome(">= 1.0 in", ticker="KXRAINNYC-26APR21-T100") is None
    assert parse_rain_outcome("0.25 to 0.5 in", ticker="KXRAINNYC-26APR21-T25") is None


def test_compute_rain_yes_probability_clips():
    from src.rain_matcher import compute_rain_yes_probability
    assert compute_rain_yes_probability(0.0) == 0.001
    assert compute_rain_yes_probability(1.0) == 0.999
    assert compute_rain_yes_probability(0.5) == 0.5
    # NO side inverts
    assert compute_rain_yes_probability(0.3, position_side="no") == pytest.approx(0.7)


def test_match_kalshi_rain_emits_edge_opportunity():
    from src.rain_matcher import match_kalshi_rain
    from datetime import datetime, timezone

    precip = {
        "New York": {
            "daily": [
                {"date": "2026-04-21", "forecast_prob_any_rain": 0.82, "forecast_amount_in": 0.4},
            ]
        }
    }
    markets = [{
        "ticker": "KXRAINNYC-26APR21-T0",
        "outcome": "Yes",
        "yes_sub_title": "Rain in NYC",
        "city": "New York",
        "market_date": "2026-04-21",
        "yes_ask": 0.55,
        "volume_24h": 1500,
        "close_time": "2026-04-21T23:59:00Z",
    }]
    now = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)

    opps = match_kalshi_rain(precip, markets, now_utc=now, min_edge=0.15)
    assert len(opps) == 1
    opp = opps[0]
    assert opp["market_category"] == "rain"
    assert opp["market_type"] == "rain_binary"
    assert opp["our_probability"] == pytest.approx(0.82)
    assert opp["abs_edge"] == pytest.approx(0.27, rel=1e-2)
    assert opp["edge"] > 0  # our_prob > market_price


def test_match_kalshi_rain_skips_when_no_market_price():
    """Markets with blank yes_ask (listed but untraded) must be skipped."""
    from src.rain_matcher import match_kalshi_rain

    precip = {"New York": {"daily": [
        {"date": "2026-04-21", "forecast_prob_any_rain": 0.82, "forecast_amount_in": 0.4},
    ]}}
    markets = [{
        "ticker": "KXRAINNYC-26APR21-T0",
        "outcome": "Yes",
        "yes_sub_title": "Rain in NYC",
        "city": "New York",
        "market_date": "2026-04-21",
        "yes_ask": None,           # blank, as seen in discovery CSV
        "volume_24h": None,
        "close_time": "2026-04-21T23:59:00Z",
    }]
    assert match_kalshi_rain(precip, markets, min_edge=0.15) == []


def test_match_kalshi_rain_respects_min_edge():
    from src.rain_matcher import match_kalshi_rain

    precip = {"New York": {"daily": [
        {"date": "2026-04-21", "forecast_prob_any_rain": 0.60, "forecast_amount_in": 0.2},
    ]}}
    markets = [{
        "ticker": "KXRAINNYC-26APR21-T0",
        "outcome": "Yes", "yes_sub_title": "Rain in NYC",
        "city": "New York", "market_date": "2026-04-21",
        "yes_ask": 0.55, "close_time": "2026-04-21T23:59:00Z",
    }]
    # edge = 0.60 - 0.55 = 0.05, below the 0.15 threshold
    assert match_kalshi_rain(precip, markets, min_edge=0.15) == []


def test_match_kalshi_rain_blends_hrrr_inside_horizon():
    """Within the HRRR blend horizon, the probability used for calibration
    must be a weighted average of raw Open-Meteo and HRRR."""
    from src.rain_matcher import match_kalshi_rain
    from datetime import datetime, timezone

    precip = {"New York": {"daily": [
        {"date": "2026-04-21", "forecast_prob_any_rain": 0.50, "forecast_amount_in": 0.2},
    ]}}
    hrrr = {"New York": {"2026-04-21": {"total_in": 0.15}}}  # HRRR says wet
    markets = [{
        "ticker": "KXRAINNYC-26APR21-T0", "outcome": "Yes", "yes_sub_title": "Rain in NYC",
        "city": "New York", "market_date": "2026-04-21",
        "yes_ask": 0.30, "close_time": "2026-04-21T23:59:00Z",
    }]
    # now = 6 hours before close, horizon = 12 → HRRR weight = min(0.7, 1 - 6/12) = 0.5
    now = datetime(2026, 4, 21, 17, 59, tzinfo=timezone.utc)

    opps = match_kalshi_rain(
        precip, markets, hrrr_data=hrrr, hrrr_blend_horizon_hours=12.0,
        now_utc=now, min_edge=0.10,
    )
    assert len(opps) == 1
    opp = opps[0]
    assert opp["forecast_blend_source"] == "open-meteo+hrrr"
    # our_prob = 0.5 * 0.50 + 0.5 * 1.0 = 0.75 (HRRR wet = 1.0 because 0.15 >= 0.01)
    assert opp["our_probability"] == pytest.approx(0.75, abs=0.01)


def test_match_kalshi_rain_skips_hrrr_outside_horizon():
    """Beyond the HRRR blend horizon, no blend should be applied."""
    from src.rain_matcher import match_kalshi_rain
    from datetime import datetime, timezone

    precip = {"New York": {"daily": [
        {"date": "2026-04-22", "forecast_prob_any_rain": 0.50, "forecast_amount_in": 0.2},
    ]}}
    hrrr = {"New York": {"2026-04-22": {"total_in": 0.15}}}
    markets = [{
        "ticker": "KXRAINNYC-26APR22-T0", "outcome": "Yes", "yes_sub_title": "Rain in NYC",
        "city": "New York", "market_date": "2026-04-22",
        "yes_ask": 0.30, "close_time": "2026-04-22T23:59:00Z",
    }]
    # now = 48 hours before close, horizon = 12 → outside
    now = datetime(2026, 4, 20, 23, 59, tzinfo=timezone.utc)

    opps = match_kalshi_rain(
        precip, markets, hrrr_data=hrrr, hrrr_blend_horizon_hours=12.0,
        now_utc=now, min_edge=0.10,
    )
    assert len(opps) == 1
    assert opps[0]["forecast_blend_source"] == "open-meteo"
    # our_prob == raw == 0.50 (no blend)
    assert opps[0]["our_probability"] == pytest.approx(0.50)


def test_match_kalshi_rain_uses_calibration_manager_when_present():
    """If a calibration manager is provided and returns a dict, the matcher
    must use the calibrated probability and tag the sources correctly."""
    from src.rain_matcher import match_kalshi_rain
    from datetime import datetime, timezone

    class _FakeMgr:
        def calibrate_rain_probability(self, city, raw_prob):
            return {
                "calibrated_prob": 0.85,
                "forecast_calibration_source": "logistic",
                "probability_calibration_source": "isotonic",
            }

    precip = {"New York": {"daily": [
        {"date": "2026-04-21", "forecast_prob_any_rain": 0.55, "forecast_amount_in": 0.2},
    ]}}
    markets = [{
        "ticker": "KXRAINNYC-26APR21-T0", "outcome": "Yes", "yes_sub_title": "Rain in NYC",
        "city": "New York", "market_date": "2026-04-21",
        "yes_ask": 0.50, "close_time": "2026-04-21T23:59:00Z",
    }]
    now = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)

    opps = match_kalshi_rain(
        precip, markets, calibration_manager=_FakeMgr(),
        now_utc=now, min_edge=0.10,
    )
    assert len(opps) == 1
    assert opps[0]["our_probability"] == pytest.approx(0.85)
    assert opps[0]["forecast_calibration_source"] == "logistic"
    assert opps[0]["probability_calibration_source"] == "isotonic"


def test_match_kalshi_rain_filters_negative_edge_for_buy_only():
    """P1 is BUY-only: opportunities where edge < 0 (market > our_prob) must
    not be emitted, even when abs_edge exceeds min_edge."""
    from src.rain_matcher import match_kalshi_rain
    from datetime import datetime, timezone

    precip = {"New York": {"daily": [
        {"date": "2026-04-21", "forecast_prob_any_rain": 0.30, "forecast_amount_in": 0.05},
    ]}}
    markets = [{
        "ticker": "KXRAINNYC-26APR21-T0", "outcome": "Yes", "yes_sub_title": "Rain in NYC",
        "city": "New York", "market_date": "2026-04-21",
        "yes_ask": 0.80, "close_time": "2026-04-21T23:59:00Z",
    }]
    now = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)

    # abs_edge = 0.5, min_edge = 0.15, but edge = -0.5 (our_prob < market_price)
    # P1 BUY-only must drop this.
    assert match_kalshi_rain(precip, markets, min_edge=0.15, now_utc=now) == []
