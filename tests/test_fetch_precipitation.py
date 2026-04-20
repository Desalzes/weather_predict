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


def test_ensemble_precip_summarizer_excludes_bare_precipitation_series():
    """Open-Meteo sometimes emits a bare "precipitation" key (the control run)
    alongside "precipitation_memberNN" keys. The summarizer must exclude it —
    counting it as a member would inflate member_count and skew wet_fraction.
    """
    from src.fetch_precipitation import _summarize_ensemble_precip

    payload = {
        "hourly": {
            "time": ["2026-04-21T00:00", "2026-04-21T12:00"],
            "precipitation": [5.0, 5.0],           # control/deterministic — must be ignored
            "precipitation_member01": [0.0, 0.0],
            "precipitation_member02": [0.0, 0.0],
        }
    }

    result = _summarize_ensemble_precip(payload)
    apr21 = next(d for d in result["daily"] if d["date"] == "2026-04-21")

    assert apr21["member_count"] == 2  # not 3; the bare "precipitation" is excluded
    assert apr21["ensemble_wet_fraction"] == 0.0  # both true members stayed dry


def test_hrrr_precip_returns_daily_accumulation(monkeypatch):
    """HRRR APCP extraction returns per-date cumulative inches per city."""
    from src import fetch_precipitation

    # Simulate extracted member timeseries
    fake_series = [
        {"valid_time": "2026-04-21T00:00:00", "apcp_kg_m2": 0.0},
        {"valid_time": "2026-04-21T06:00:00", "apcp_kg_m2": 2.54},   # 0.1 in
        {"valid_time": "2026-04-21T12:00:00", "apcp_kg_m2": 2.54},   # unchanged (no new rain)
        {"valid_time": "2026-04-21T18:00:00", "apcp_kg_m2": 7.62},   # +0.2 in
    ]

    def fake_extract(loc, fxx, **kwargs):
        return fake_series

    monkeypatch.setattr(fetch_precipitation, "_extract_hrrr_apcp_series", fake_extract)

    result = fetch_precipitation.fetch_hrrr_precip_multi(
        [{"name": "New York", "lat": 40.7, "lon": -74.0}],
        fxx=18,
    )

    ny = result["New York"]
    assert ny["2026-04-21"]["total_in"] == pytest.approx(0.3, rel=1e-3)
