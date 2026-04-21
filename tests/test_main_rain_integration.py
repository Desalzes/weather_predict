from unittest.mock import patch


def test_run_scan_includes_rain_opportunities_when_enabled(tmp_path):
    """With enable_rain_vertical=true, run_scan returns rain opportunities
    alongside any temperature opportunities."""
    from main import run_scan

    config = {
        "locations": [{"name": "New York", "lat": 40.7, "lon": -74.0}],
        "forecast_hours": 48,
        "min_edge_threshold": 0.1,
        "uncertainty_std_f": 2.0,
        "enable_ensemble": False,
        "enable_calibration": False,
        "enable_hrrr": False,
        "enable_kalshi": True,
        "enable_polymarket": False,
        "enable_rain_vertical": True,
        "rain_watchlist": ["New York"],
        "rain_min_edge_threshold": 0.1,
        "rain_hrrr_blend_horizon_hours": 12,
        "rain_strategy_policy_path": "strategy/rain_policy_v1.json",
        "strategy_policy_path": "strategy/strategy_policy.json",
        "enable_deepseek_worker": False,
        "enable_paper_trading": False,
        "opportunity_archive_enabled": False,
    }

    precip = {"New York": {"daily": [
        {"date": "2026-04-21", "forecast_prob_any_rain": 0.82, "forecast_amount_in": 0.4}
    ]}}
    rain_market = {
        "ticker": "KXRAINNYC-26APR21-T0",
        "outcome": "Yes",
        "yes_sub_title": "Rain in NYC",
        "city": "New York",
        "market_date": "2026-04-21",
        "yes_ask": 0.55,
        "volume_24h": 1500,
        "close_time": "2026-04-21T23:59:00+00:00",
    }

    # Temperature fetchers return a minimal payload; no temp markets returned.
    with patch("src.fetch_forecasts.fetch_multi_location", return_value={
                 "New York": {"daily": {"time": ["2026-04-21"], "temperature_2m_max": [70], "temperature_2m_min": [50]}}
             }), \
         patch("src.fetch_kalshi.fetch_weather_markets", return_value=[rain_market]), \
         patch("src.fetch_precipitation.fetch_precipitation_multi", return_value=precip), \
         patch("src.fetch_precipitation.fetch_precipitation_ensemble_multi", return_value={}), \
         patch("src.station_truth.archive_forecast_snapshot"):
        result = run_scan(config=config)

    rain_opps = [o for o in result["opportunities"] if o.get("market_category") == "rain"]
    assert len(rain_opps) == 1, f"expected 1 rain opp, got {len(rain_opps)}: {rain_opps}"
    assert rain_opps[0]["ticker"] == "KXRAINNYC-26APR21-T0"
    assert rain_opps[0]["direction"] == "BUY"
    assert rain_opps[0]["our_probability"] > rain_opps[0]["market_price"]
