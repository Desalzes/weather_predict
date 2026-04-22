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
        # forecast_days = max(1, int(forecast_hours / 24)); 72h must map to 3d.
        assert mock_get.call_args.kwargs["params"]["forecast_days"] == 3

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


def test_hrrr_precip_returns_none_when_all_hours_fail(monkeypatch, caplog):
    """When every hour's extraction fails, the city's entry must be None
    (not an empty dict, which would look like 'no rain')."""
    import logging
    from src import fetch_precipitation

    def failing_extract(location, fxx, **kwargs):
        return []  # simulate 0/19 successful fetches

    monkeypatch.setattr(fetch_precipitation, "_extract_hrrr_apcp_series", failing_extract)

    with caplog.at_level(logging.WARNING, logger="weather.fetch_precipitation"):
        result = fetch_precipitation.fetch_hrrr_precip_multi(
            [{"name": "New York", "lat": 40.7, "lon": -74.0}],
            fxx=18,
        )

    assert result["New York"] is None
    assert any("0/19" in rec.message and "New York" in rec.message for rec in caplog.records)


def test_hrrr_apcp_extractor_pins_run_time_across_lead_hours(monkeypatch):
    """I1: every Herbie(...) invocation in a single extract call must use
    the same run_time_utc so APCP doesn't cross a run-reset boundary."""
    pytest.importorskip("herbie")

    from datetime import datetime, timezone
    from src import fetch_precipitation

    class _FakeDataset:
        def __init__(self, valid_time, value):
            self._value = value

            class _V:
                def __init__(self, v):
                    self.values = v

            self.valid_time = _V(valid_time)

        def sel(self, **kwargs):
            class _S:
                def __init__(self, v):
                    self.values = v

            return _S(self._value)

    run_times_seen = []

    class _FakeHerbie:
        def __init__(self, *, model, product, fxx, **kwargs):
            run_times_seen.append(kwargs.get("run_time") or kwargs.get("date"))
            self._fxx = fxx

        def xarray(self, *args, **kwargs):
            return _FakeDataset(f"2026-04-21T{self._fxx:02d}:00:00", float(self._fxx))

    # Replace the Herbie import target so _extract_hrrr_apcp_series picks it up
    import herbie  # noqa: F401
    monkeypatch.setattr("herbie.Herbie", _FakeHerbie)

    pinned_run = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
    result = fetch_precipitation._extract_hrrr_apcp_series(
        {"lat": 40.7, "lon": -74.0},
        fxx=5,
        run_time_utc=pinned_run,
    )

    assert len(run_times_seen) >= 1
    # All calls used the pinned run, not whatever Herbie's default would pick
    assert all(rt == pinned_run for rt in run_times_seen)
    # And the extract should have produced 6 rows (f00..f05)
    assert len(result) == 6
