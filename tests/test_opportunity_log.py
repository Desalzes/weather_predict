"""Tests for opportunity archive logging."""

import pandas as pd
import pytest
from pathlib import Path

from src.opportunity_log import log_opportunities, OPPORTUNITY_ARCHIVE_COLUMNS


def test_log_opportunities_creates_file_with_expected_columns(tmp_path):
    scan_id = "2026-04-17T12:00:00+00:00"
    opps = [
        {
            "source": "kalshi",
            "ticker": "KXHIGHTAUS-26APR17-T75",
            "city": "Austin",
            "market_type": "high",
            "market_date": "2026-04-17",
            "outcome": ">75°F",
            "our_probability": 0.62,
            "raw_probability": 0.58,
            "market_price": 0.45,
            "edge": 0.17,
            "abs_edge": 0.17,
            "forecast_value_f": 77.5,
            "raw_forecast_value_f": 77.0,
            "uncertainty_std_f": 2.3,
            "forecast_blend_source": "open-meteo",
            "forecast_calibration_source": "ngr",
            "probability_calibration_source": "raw",
            "hours_to_settlement": 22.0,
            "volume24hr": 4200,
        }
    ]
    path = log_opportunities(scan_id, opps, archive_dir=tmp_path)
    assert path.exists()
    assert path.name == "2026-04-17.csv"

    frame = pd.read_csv(path)
    assert set(frame.columns) == set(OPPORTUNITY_ARCHIVE_COLUMNS)
    assert frame.iloc[0]["ticker"] == "KXHIGHTAUS-26APR17-T75"
    assert frame.iloc[0]["our_probability"] == 0.62


def test_log_opportunities_appends_on_second_scan_same_day(tmp_path):
    scan_id_a = "2026-04-17T12:00:00+00:00"
    scan_id_b = "2026-04-17T12:30:00+00:00"

    opp = {
        "source": "kalshi", "ticker": "X", "city": "A", "market_type": "high",
        "market_date": "2026-04-17", "outcome": ">75°F",
        "our_probability": 0.5, "raw_probability": 0.5, "market_price": 0.3,
        "edge": 0.2, "abs_edge": 0.2, "forecast_value_f": 76.0,
        "raw_forecast_value_f": 76.0, "uncertainty_std_f": 2.0,
        "forecast_blend_source": "open-meteo", "forecast_calibration_source": "ngr",
        "probability_calibration_source": "raw", "hours_to_settlement": 23.0,
        "volume24hr": 1000,
    }
    log_opportunities(scan_id_a, [opp], archive_dir=tmp_path)
    log_opportunities(scan_id_b, [opp], archive_dir=tmp_path)

    frame = pd.read_csv(tmp_path / "2026-04-17.csv")
    assert len(frame) == 2
    assert frame.iloc[0]["scan_id"] == scan_id_a
    assert frame.iloc[1]["scan_id"] == scan_id_b


def test_log_opportunities_empty_list_is_noop(tmp_path):
    path = log_opportunities("2026-04-17T12:00:00+00:00", [], archive_dir=tmp_path)
    assert path is None


def test_settle_opportunity_archive_fills_outcome(tmp_path):
    from src.opportunity_log import settle_opportunity_archive

    archive_file = tmp_path / "2026-04-17.csv"
    df = pd.DataFrame([
        {
            "scan_id": "s1", "recorded_at_utc": "s1", "source": "kalshi",
            "ticker": "T1", "city": "Austin", "market_type": "high",
            "market_date": "2026-04-17", "outcome": ">75°F",
            "our_probability": 0.6, "raw_probability": 0.55, "market_price": 0.4,
            "edge": 0.2, "abs_edge": 0.2, "forecast_value_f": 77.0,
            "raw_forecast_value_f": 77.0, "uncertainty_std_f": 2.0,
            "forecast_blend_source": "open-meteo", "forecast_calibration_source": "ngr",
            "probability_calibration_source": "raw", "hours_to_settlement": 20.0,
            "volume24hr": 5000, "yes_outcome": None, "actual_value_f": None,
            "settled_at_utc": None,
        },
        {
            "scan_id": "s2", "recorded_at_utc": "s2", "source": "kalshi",
            "ticker": "T2", "city": "Austin", "market_type": "high",
            "market_date": "2026-04-17", "outcome": ">80°F",
            "our_probability": 0.3, "raw_probability": 0.25, "market_price": 0.5,
            "edge": -0.2, "abs_edge": 0.2, "forecast_value_f": 77.0,
            "raw_forecast_value_f": 77.0, "uncertainty_std_f": 2.0,
            "forecast_blend_source": "open-meteo", "forecast_calibration_source": "ngr",
            "probability_calibration_source": "raw", "hours_to_settlement": 20.0,
            "volume24hr": 5000, "yes_outcome": None, "actual_value_f": None,
            "settled_at_utc": None,
        },
    ])
    df.to_csv(archive_file, index=False)

    # Austin hit 78F: first opp (>75) YES, second (>80) NO
    station_actuals = {
        "Austin": pd.DataFrame([
            {"date": "2026-04-17", "tmax_f": 78.0, "tmin_f": 60.0},
        ]),
    }

    settled = settle_opportunity_archive(archive_dir=tmp_path, station_actuals=station_actuals)
    assert settled == 2

    after = pd.read_csv(archive_file)
    row_gt75 = after.loc[after["ticker"] == "T1"].iloc[0]
    row_gt80 = after.loc[after["ticker"] == "T2"].iloc[0]
    assert row_gt75["yes_outcome"] == 1
    assert row_gt80["yes_outcome"] == 0
    assert row_gt75["actual_value_f"] == 78.0
