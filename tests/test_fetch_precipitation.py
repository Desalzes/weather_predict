import pytest
from unittest.mock import patch


def test_fetch_precipitation_returns_daily_probs_and_amounts():
    from src.fetch_precipitation import fetch_precipitation_multi

    fake_payload = {
        "daily": {
            "time": ["2026-04-21", "2026-04-22"],
            "precipitation_sum": [0.0, 5.1],
            "precipitation_probability_max": [5, 82],
        }
    }

    with patch("src.fetch_precipitation.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = fake_payload

        result = fetch_precipitation_multi(
            [{"name": "New York", "lat": 40.7, "lon": -74.0}],
            forecast_hours=72,
        )

    ny = result["New York"]
    assert ny["daily"][0]["date"] == "2026-04-21"
    assert ny["daily"][0]["forecast_prob_any_rain"] == pytest.approx(0.05)
    assert ny["daily"][0]["forecast_amount_in"] == pytest.approx(0.0, abs=1e-6)
    assert ny["daily"][1]["forecast_prob_any_rain"] == pytest.approx(0.82)
    assert ny["daily"][1]["forecast_amount_in"] == pytest.approx(5.1 / 25.4, rel=1e-3)


def test_ensemble_precip_summarizer_computes_wet_fraction():
    from src.fetch_precipitation import _summarize_ensemble_precip

    payload = {
        "hourly": {
            "time": ["2026-04-21T00:00", "2026-04-21T12:00", "2026-04-22T00:00"],
            "precipitation_member01": [0.5, 0.0, 0.0],
            "precipitation_member02": [0.0, 0.0, 2.0],
            "precipitation_member03": [0.0, 0.0, 0.0],
            "precipitation_member04": [1.0, 0.0, 1.5],
        }
    }

    result = _summarize_ensemble_precip(payload)
    apr21 = next(d for d in result["daily"] if d["date"] == "2026-04-21")
    apr22 = next(d for d in result["daily"] if d["date"] == "2026-04-22")

    assert apr21["ensemble_wet_fraction"] == pytest.approx(0.5)   # members 01 + 04 wet
    assert apr22["ensemble_wet_fraction"] == pytest.approx(0.5)   # members 02 + 04 wet
    assert apr21["member_count"] == 4
