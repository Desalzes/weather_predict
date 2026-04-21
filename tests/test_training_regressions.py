import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from evaluate_calibration import (
    _report_level_limitations,
    evaluate_market_type_holdout,
    split_chronological_holdout,
)
from src.calibration import (
    SELECTIVE_RAW_FALLBACK_SOURCE,
    SELECTIVE_RAW_FALLBACK_TARGETS,
    CalibrationManager,
    EMOSCalibrator,
    IsotonicCalibrator,
    _bucket_bounds_from_bucket_start,
)
from src.matcher import (
    _kalshi_bucket_bounds_from_threshold,
    _should_apply_probability_calibration,
    match_kalshi_markets,
)
from src.station_truth import (
    _prepare_forecast_archive_frame,
    build_training_set,
    forecast_archive_path,
    station_actuals_path,
)


class TrainingRegressionTests(unittest.TestCase):
    @staticmethod
    def _build_hourly_forecast(target_date: str, temps_f: list[float]) -> dict:
        return {
            "hourly": {
                "time": [f"{target_date}T{hour:02d}:00" for hour in range(len(temps_f))],
                "temperature_2m": [((temp_f - 32.0) * 5.0 / 9.0) for temp_f in temps_f],
            },
            "timezone": "America/New_York",
        }

    @staticmethod
    def _build_kalshi_bucket_market(
        city: str,
        market_type: str,
        target_date: str,
        threshold: float,
        close_time: str,
    ) -> dict:
        date_code = datetime.fromisoformat(target_date).strftime("%y%b%d").upper()
        city_code = "".join(part[:1] for part in city.split()).upper() or "X"
        return {
            "city": city,
            "type": market_type,
            "threshold": threshold,
            "ticker": f"KX{market_type.upper()}{city_code}-{date_code}-B{threshold}",
            "title": f"{city} {market_type} bucket",
            "close_time": close_time,
            "last_price": 0.40,
            "yes_bid": 0.39,
            "yes_ask": 0.41,
            "volume_24h": 1000,
        }

    def _match_single_bucket_market(
        self,
        *,
        city: str,
        market_type: str,
        temps_f: list[float],
        calibration_manager: CalibrationManager,
        target_date: str = "2026-04-05",
        threshold: float = 36.5,
        close_time: str = "2026-04-05T23:00:00+00:00",
        now_utc: datetime | None = None,
    ) -> dict:
        forecasts = {city: self._build_hourly_forecast(target_date, temps_f)}
        markets = [
            self._build_kalshi_bucket_market(
                city,
                market_type,
                target_date,
                threshold,
                close_time,
            )
        ]

        opportunities = match_kalshi_markets(
            forecasts,
            markets,
            min_edge=0.0,
            uncertainty_std_f=2.0,
            calibration_manager=calibration_manager,
            hrrr_blend_horizon_hours=18.0,
            now_utc=now_utc,
        )

        self.assertEqual(len(opportunities), 1)
        return opportunities[0]

    @staticmethod
    def _build_synthetic_training_rows(
        *,
        total_rows: int = 20,
        start_date: datetime | None = None,
        lead_day_fn=None,
    ) -> list[dict]:
        base_date = (start_date or datetime(2026, 1, 1, tzinfo=timezone.utc)).date()
        rows = []
        for offset in range(total_rows):
            forecast_high = 60.0 + offset * 0.4
            forecast_low = 38.0 + offset * 0.2
            row = {
                "date": (base_date + timedelta(days=offset)).isoformat(),
                "forecast_high_f": forecast_high,
                "actual_high_f": forecast_high + (1.5 if offset % 2 == 0 else -0.5),
                "forecast_low_f": forecast_low,
                "actual_low_f": forecast_low + (-1.0 if offset % 3 == 0 else 0.75),
                "ensemble_high_std_f": 2.0,
                "ensemble_low_std_f": 1.5,
            }
            if lead_day_fn is not None:
                row["forecast_lead_days"] = lead_day_fn(offset)
            rows.append(row)
        return rows

    def test_prepare_forecast_archive_frame_parses_iso8601_microseconds(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-04-01"],
                "as_of_utc": ["2026-04-01T15:41:31.071823+00:00"],
            }
        )

        prepared = _prepare_forecast_archive_frame(frame)

        self.assertFalse(pd.isna(prepared.loc[0, "as_of_utc"]))
        self.assertEqual(prepared.loc[0, "forecast_source"], "live_scan")
        self.assertEqual(int(prepared.loc[0, "training_priority"]), 0)

    def test_build_training_set_prefers_previous_run_forecast_rows(self):
        city = "Test City"
        target_date = datetime.now(timezone.utc).date() - timedelta(days=2)
        prior_date = target_date - timedelta(days=1)
        run_dir = Path("data") / "test_runs" / f"training_{uuid4().hex}"
        actuals_dir = run_dir / "actuals"
        forecast_dir = run_dir / "archive"
        actuals_dir.mkdir(parents=True, exist_ok=True)
        forecast_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {
                    "date": target_date.isoformat(),
                    "tmax_f": 72.0,
                    "tmin_f": 51.0,
                }
            ]
        ).to_csv(station_actuals_path(city, base_dir=actuals_dir), index=False)

        pd.DataFrame(
            [
                {
                    "as_of_utc": f"{prior_date.isoformat()}T12:00:00+00:00",
                    "date": target_date.isoformat(),
                    "forecast_high_f": 70.0,
                    "forecast_low_f": 50.0,
                    "ensemble_high_std_f": 1.2,
                    "ensemble_low_std_f": 0.8,
                    "forecast_source": "open_meteo_previous_runs",
                    "forecast_lead_days": 1,
                },
                {
                    "as_of_utc": f"{target_date.isoformat()}T15:41:31.071823+00:00",
                    "date": target_date.isoformat(),
                    "forecast_high_f": 88.0,
                    "forecast_low_f": 65.0,
                    "ensemble_high_std_f": 1.0,
                    "ensemble_low_std_f": 1.0,
                    "forecast_source": "live_scan",
                    "forecast_lead_days": "",
                },
            ]
        ).to_csv(forecast_archive_path(city, base_dir=forecast_dir), index=False)

        training = build_training_set(
            city,
            days=30,
            station_actuals_dir=actuals_dir,
            forecast_archive_dir=forecast_dir,
        )

        self.assertEqual(len(training), 1)
        row = training.iloc[0]
        self.assertEqual(row["date"], target_date.isoformat())
        self.assertAlmostEqual(float(row["forecast_high_f"]), 70.0)
        self.assertAlmostEqual(float(row["forecast_low_f"]), 50.0)
        self.assertEqual(int(row["forecast_lead_days"]), 1)
        self.assertIn(prior_date.isoformat(), str(row["as_of_utc"]))

    def test_bucket_training_bounds_match_live_market_bucket(self):
        self.assertEqual(
            _bucket_bounds_from_bucket_start(66),
            _kalshi_bucket_bounds_from_threshold(66.5),
        )

    def test_probability_calibration_is_disabled_for_intraday_regime(self):
        self.assertFalse(
            _should_apply_probability_calibration("open-meteo+hrrr", 12.0, 18.0)
        )
        self.assertFalse(
            _should_apply_probability_calibration("open-meteo", 12.0, 18.0)
        )
        self.assertTrue(
            _should_apply_probability_calibration("open-meteo", 24.0, 18.0)
        )

    def test_split_chronological_holdout_reserves_newest_dates(self):
        frame = pd.DataFrame(
            {
                "date": [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                ],
                "forecast_high_f": [60, 61, 62, 63, 64],
                "actual_high_f": [59, 60, 63, 62, 65],
                "forecast_low_f": [40, 41, 42, 43, 44],
                "actual_low_f": [39, 42, 41, 44, 43],
            }
        )

        train_df, holdout_df = split_chronological_holdout(frame, holdout_days=2)

        self.assertEqual(train_df["date"].tolist(), ["2026-01-01", "2026-01-02", "2026-01-03"])
        self.assertEqual(holdout_df["date"].tolist(), ["2026-01-04", "2026-01-05"])

    def test_selective_fallback_target_list_stays_on_benchmarked_five_pairs(self):
        self.assertEqual(
            set(SELECTIVE_RAW_FALLBACK_TARGETS),
            {
                ("Boston", "low"),
                ("Minneapolis", "low"),
                ("Philadelphia", "low"),
                ("New Orleans", "high"),
                ("San Francisco", "low"),
            },
        )
        self.assertEqual(len(SELECTIVE_RAW_FALLBACK_TARGETS), 5)

    def test_evaluate_market_type_holdout_reports_metrics_and_counts(self):
        rows = self._build_synthetic_training_rows(
            lead_day_fn=lambda offset: 1 if offset % 2 == 0 else 2
        )

        result = evaluate_market_type_holdout(
            "Synthetic City",
            pd.DataFrame(rows),
            "high",
            holdout_days=4,
            min_train_rows=10,
            min_holdout_rows=4,
        )

        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["train_rows"], 16)
        self.assertEqual(result["holdout_rows"], 4)
        self.assertEqual(result["train_end_date"], "2026-01-16")
        self.assertEqual(result["holdout_start_date"], "2026-01-17")
        broad = result["policies"]["broad_emos_isotonic"]
        selective = result["policies"]["selective_raw_fallback"]
        self.assertIsNotNone(broad["raw_mae_f"])
        self.assertIsNotNone(broad["policy_mae_f"])
        self.assertIsNotNone(broad["raw_brier"])
        self.assertIsNotNone(broad["policy_brier"])
        self.assertGreater(broad["holdout_probability_examples"], 0)
        self.assertIn("temperature_diagnostics", broad)
        self.assertIn("probability_diagnostics", broad)
        self.assertAlmostEqual(broad["temperature_diagnostics"]["raw_bias_f"], -0.5)
        self.assertIsNotNone(broad["temperature_diagnostics"]["policy_bias_f"])
        self.assertEqual(
            broad["temperature_diagnostics"]["raw_bias_direction"],
            "cold_or_low",
        )
        self.assertEqual(
            broad["temperature_diagnostics"]["regime_diagnostics"]["forecast_regime"]["status"],
            "evaluated",
        )
        self.assertEqual(
            broad["temperature_diagnostics"]["regime_diagnostics"]["holdout_half"]["status"],
            "evaluated",
        )
        self.assertEqual(
            broad["temperature_diagnostics"]["regime_diagnostics"]["lead_time_regime"]["status"],
            "evaluated",
        )
        self.assertEqual(
            broad["temperature_diagnostics"]["regime_diagnostics"]["lead_time_regime"]["bucket_definition"],
            {
                "day_ahead": "forecast_lead_days == 1",
                "multi_day": "forecast_lead_days >= 2",
            },
        )
        self.assertEqual(
            [segment["segment"] for segment in broad["temperature_diagnostics"]["regime_diagnostics"]["lead_time_regime"]["segments"]],
            ["day_ahead", "multi_day"],
        )
        self.assertEqual(
            broad["temperature_diagnostics"]["regime_diagnostics"]["spread_regime"]["status"],
            "unsupported",
        )
        self.assertTrue(
            any(item["dimension"] == "spread_regime" for item in result["diagnostic_limitations"])
        )
        self.assertFalse(
            any(item["dimension"] == "lead_time_regime" for item in result["diagnostic_limitations"])
        )
        self.assertEqual(broad["probability_diagnostics"]["status"], "evaluated")
        self.assertIsNotNone(broad["probability_diagnostics"]["raw_probability_bias"])
        self.assertEqual(selective["temperature_source"], "emos")
        self.assertAlmostEqual(selective["policy_mae_f"], broad["policy_mae_f"])

    def test_evaluate_market_type_holdout_reports_explicit_lead_time_insufficiency(self):
        rows = self._build_synthetic_training_rows(lead_day_fn=lambda _offset: 1)

        result = evaluate_market_type_holdout(
            "Lead Time Limited City",
            pd.DataFrame(rows),
            "high",
            holdout_days=4,
            min_train_rows=10,
            min_holdout_rows=4,
        )

        lead_time = result["policies"]["broad_emos_isotonic"]["temperature_diagnostics"]["regime_diagnostics"][
            "lead_time_regime"
        ]
        self.assertEqual(lead_time["status"], "unsupported")
        self.assertIn("usable rows only covered day_ahead", lead_time["reason"])
        self.assertEqual(lead_time["usable_holdout_rows"], 4)
        self.assertEqual(lead_time["ignored_holdout_rows"], 0)
        self.assertTrue(
            any(item["dimension"] == "lead_time_regime" for item in result["diagnostic_limitations"])
        )

    def test_selective_fallback_uses_raw_forecast_only_for_targeted_pairs(self):
        start_date = datetime(2026, 1, 1, tzinfo=timezone.utc).date()
        rows = []
        for offset in range(20):
            forecast_low = 30.0 + offset * 0.5
            rows.append(
                {
                    "date": (start_date + timedelta(days=offset)).isoformat(),
                    "forecast_high_f": 55.0 + offset * 0.3,
                    "actual_high_f": 55.5 + offset * 0.3,
                    "forecast_low_f": forecast_low,
                    "actual_low_f": forecast_low + 1.5,
                    "ensemble_high_std_f": 2.0,
                    "ensemble_low_std_f": 1.5,
                }
            )

        result = evaluate_market_type_holdout(
            "Boston",
            pd.DataFrame(rows),
            "low",
            holdout_days=4,
            min_train_rows=10,
            min_holdout_rows=4,
        )

        broad = result["policies"]["broad_emos_isotonic"]
        selective = result["policies"]["selective_raw_fallback"]
        self.assertTrue(result["is_targeted_fallback_pair"])
        self.assertEqual(broad["temperature_source"], "emos")
        self.assertEqual(selective["temperature_source"], SELECTIVE_RAW_FALLBACK_SOURCE)
        self.assertEqual(selective["probability_input_source"], "raw")
        self.assertAlmostEqual(selective["policy_mae_f"], selective["raw_mae_f"])
        self.assertAlmostEqual(selective["temperature_diagnostics"]["mean_adjustment_f"], 0.0)
        self.assertNotAlmostEqual(broad["policy_mae_f"], broad["raw_mae_f"])

    def test_runtime_targeted_pair_returns_raw_with_explicit_fallback_source(self):
        city = "Boston"
        market_type = "low"
        manager = CalibrationManager(model_dir=Path.cwd())
        manager._emos_cache[(city, market_type)] = EMOSCalibrator(
            city=city,
            market_type=market_type,
            a=5.0,
            b=1.0,
            c=0.0,
            training_rows=20,
            is_fitted=True,
        )

        opportunity = self._match_single_bucket_market(
            city=city,
            market_type=market_type,
            temps_f=[42.0, 39.0, 37.0, 35.0, 36.0, 38.0],
            calibration_manager=manager,
        )

        self.assertEqual(opportunity["forecast_calibration_source"], SELECTIVE_RAW_FALLBACK_SOURCE)
        self.assertEqual(opportunity["raw_forecast_value_f"], 35.0)
        self.assertEqual(opportunity["forecast_value_f"], 35.0)

    def test_runtime_non_targeted_pair_keeps_emos_forecast_routing(self):
        city = "Chicago"
        market_type = "low"
        manager = CalibrationManager(model_dir=Path.cwd())
        manager._emos_cache[(city, market_type)] = EMOSCalibrator(
            city=city,
            market_type=market_type,
            a=5.0,
            b=1.0,
            c=0.0,
            training_rows=20,
            is_fitted=True,
        )

        opportunity = self._match_single_bucket_market(
            city=city,
            market_type=market_type,
            temps_f=[42.0, 39.0, 37.0, 35.0, 36.0, 38.0],
            calibration_manager=manager,
        )

        self.assertEqual(opportunity["forecast_calibration_source"], "emos")
        self.assertEqual(opportunity["raw_forecast_value_f"], 35.0)
        self.assertEqual(opportunity["forecast_value_f"], 40.0)

    def test_runtime_targeted_pair_keeps_existing_isotonic_gate_when_eligible(self):
        city = "Boston"
        market_type = "low"
        manager = CalibrationManager(model_dir=Path.cwd())
        manager._emos_cache[(city, market_type)] = EMOSCalibrator(
            city=city,
            market_type=market_type,
            a=5.0,
            b=1.0,
            c=0.0,
            training_rows=20,
            is_fitted=True,
        )
        manager._isotonic_cache[(city, market_type)] = IsotonicCalibrator(
            city=city,
            market_type=market_type,
        ).fit(
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.75, 0.80, 0.85, 0.90, 0.95],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        )

        opportunity = self._match_single_bucket_market(
            city=city,
            market_type=market_type,
            temps_f=[42.0, 39.0, 37.0, 35.0, 36.0, 38.0],
            calibration_manager=manager,
            now_utc=datetime(2026, 4, 4, tzinfo=timezone.utc),
        )

        self.assertEqual(opportunity["forecast_calibration_source"], SELECTIVE_RAW_FALLBACK_SOURCE)
        self.assertEqual(opportunity["forecast_blend_source"], "open-meteo")
        self.assertEqual(opportunity["probability_calibration_source"], "isotonic")

    def test_evaluate_market_type_holdout_reports_explicit_skip_reason(self):
        frame = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "forecast_high_f": [60.0, 62.0, 64.0],
                "actual_high_f": [61.0, 63.0, 65.0],
                "forecast_low_f": [40.0, 42.0, 44.0],
                "actual_low_f": [39.0, 41.0, 43.0],
                "ensemble_high_std_f": [1.5, 1.5, 1.5],
                "ensemble_low_std_f": [1.5, 1.5, 1.5],
            }
        )

        result = evaluate_market_type_holdout(
            "Tiny City",
            frame,
            "low",
            holdout_days=2,
            min_train_rows=2,
            min_holdout_rows=2,
        )

        self.assertEqual(result["status"], "skipped")
        self.assertIn("need at least 2 train rows", result["reason"])
        self.assertEqual(result["train_rows"], 1)
        self.assertEqual(result["holdout_rows"], 2)
        for policy in result["policies"].values():
            self.assertEqual(policy["temperature_diagnostics"]["reason"], result["reason"])
            self.assertEqual(policy["probability_diagnostics"]["reason"], result["reason"])

    def test_report_level_limitations_summarize_partial_lead_time_support(self):
        supported_result = evaluate_market_type_holdout(
            "Supported City",
            pd.DataFrame(self._build_synthetic_training_rows(lead_day_fn=lambda offset: 1 if offset % 2 == 0 else 2)),
            "high",
            holdout_days=4,
            min_train_rows=10,
            min_holdout_rows=4,
        )
        unsupported_result = evaluate_market_type_holdout(
            "Unsupported City",
            pd.DataFrame(self._build_synthetic_training_rows(lead_day_fn=lambda _offset: 1)),
            "high",
            holdout_days=4,
            min_train_rows=10,
            min_holdout_rows=4,
        )

        limitations = _report_level_limitations([supported_result, unsupported_result])
        lead_time = next(item for item in limitations if item["dimension"] == "lead_time_regime")

        self.assertEqual(lead_time["status"], "partially_supported")
        self.assertEqual(lead_time["evaluated_pairs"], 2)
        self.assertEqual(lead_time["supported_pairs"], 1)
        self.assertEqual(lead_time["unsupported_pairs"], 1)
        self.assertEqual(
            lead_time["bucket_definition"],
            {
                "day_ahead": "forecast_lead_days == 1",
                "multi_day": "forecast_lead_days >= 2",
            },
        )
        self.assertIn("evaluated for 1/2 city-market pairs", lead_time["reason"])

    def test_report_level_limitations_treat_all_missing_lead_time_splits_as_data_limited(self):
        unsupported_result = evaluate_market_type_holdout(
            "Unsupported City",
            pd.DataFrame(self._build_synthetic_training_rows(lead_day_fn=lambda _offset: 1)),
            "high",
            holdout_days=4,
            min_train_rows=10,
            min_holdout_rows=4,
        )

        limitations = _report_level_limitations([unsupported_result])
        lead_time = next(item for item in limitations if item["dimension"] == "lead_time_regime")

        self.assertEqual(lead_time["status"], "insufficient_data")
        self.assertEqual(lead_time["supported_pairs"], 0)
        self.assertEqual(lead_time["unsupported_pairs"], 1)
        self.assertIn("wired through the evaluator", lead_time["reason"])


def test_ncei_cdo_request_includes_prcp(monkeypatch):
    """NCEI CDO fetch must request PRCP alongside TMAX/TMIN."""
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params

        class _R:
            status_code = 200

            def json(self):
                return {"results": []}

            def raise_for_status(self):
                pass

        return _R()

    import requests
    from src import station_truth
    monkeypatch.setattr(requests, "get", fake_get)

    station_truth.fetch_historical_daily(
        ghcnd_id="USW00094728",
        start="2025-04-01",
        end="2025-04-05",
        token="fake-token",
    )

    datatypes = captured["params"].get("datatypeid")
    if isinstance(datatypes, list):
        assert "PRCP" in datatypes
    else:
        assert "PRCP" in str(datatypes)
    assert "TMAX" in str(datatypes) and "TMIN" in str(datatypes)


def test_ncei_cdo_parses_prcp_tenths_mm_to_inches(monkeypatch):
    """NCEI CDO parse path must convert PRCP tenths-of-mm to inches and leave missing days NaN."""
    # Three dates:
    #   2025-04-01 — PRCP=254 tenths-mm (1.0 in) with TMAX/TMIN
    #   2025-04-02 — PRCP=127 tenths-mm (0.5 in) with TMAX/TMIN
    #   2025-04-03 — PRCP=25  tenths-mm (~0.098 in) with TMAX/TMIN
    #   2025-04-04 — TMAX/TMIN only, NO PRCP row (precip_in must be NaN)
    results = [
        {"date": "2025-04-01T00:00:00", "datatype": "TMAX", "value": 200},
        {"date": "2025-04-01T00:00:00", "datatype": "TMIN", "value": 100},
        {"date": "2025-04-01T00:00:00", "datatype": "PRCP", "value": 254},
        {"date": "2025-04-02T00:00:00", "datatype": "TMAX", "value": 210},
        {"date": "2025-04-02T00:00:00", "datatype": "TMIN", "value": 110},
        {"date": "2025-04-02T00:00:00", "datatype": "PRCP", "value": 127},
        {"date": "2025-04-03T00:00:00", "datatype": "TMAX", "value": 220},
        {"date": "2025-04-03T00:00:00", "datatype": "TMIN", "value": 120},
        {"date": "2025-04-03T00:00:00", "datatype": "PRCP", "value": 25},
        {"date": "2025-04-04T00:00:00", "datatype": "TMAX", "value": 230},
        {"date": "2025-04-04T00:00:00", "datatype": "TMIN", "value": 130},
    ]

    payload = {
        "results": results,
        "metadata": {
            "resultset": {
                "count": len(results),
                "offset": 1,
                "limit": 1000,
            }
        },
    }

    def fake_get(url, params=None, headers=None, timeout=None):
        class _R:
            status_code = 200

            def json(self):
                return payload

            def raise_for_status(self):
                pass

        return _R()

    import requests
    from src import station_truth
    monkeypatch.setattr(requests, "get", fake_get)

    frame = station_truth.fetch_historical_daily(
        ghcnd_id="USW00094728",
        start="2025-04-01",
        end="2025-04-04",
        token="fake-token",
    )

    assert list(frame.columns) == ["date", "tmax_f", "tmin_f", "precip_in"]
    assert len(frame) == 4

    by_date = {row["date"]: row for _, row in frame.iterrows()}

    # 254 tenths-mm -> exactly 1.0 in
    assert by_date["2025-04-01"]["precip_in"] == 1.0
    # 127 tenths-mm -> 0.5 in
    assert by_date["2025-04-02"]["precip_in"] == 0.5
    # 25 tenths-mm -> ~0.098 in (25/254 = 0.09842...; rounded to 3 decimals = 0.098)
    assert by_date["2025-04-03"]["precip_in"] == 0.098
    # 2025-04-04 has TMAX/TMIN but NO PRCP row -> precip_in must be NaN (not 0.0)
    missing = by_date["2025-04-04"]["precip_in"]
    assert pd.isna(missing), f"expected NaN for missing PRCP, got {missing!r}"


def test_fetch_previous_run_forecast_returns_precipitation(monkeypatch):
    """Previous-run forecast must expose precipitation_sum and probability.

    The real function returns a multi-day hourly payload for a date range
    rather than a single-day snapshot. Adapt the plan's mock/call shape to
    match that, but keep the two new precipitation field names intact: they
    are consumed by `archive_previous_run_precipitation` downstream.
    """
    import requests
    from src import fetch_forecasts

    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params

        class _R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                # Hourly precip totals 2.54 mm over 24 hours (0.1 inches).
                # Open-Meteo Previous Runs API uses the *_previous_day{N}
                # suffix on hourly variables only; for daily variables the
                # plain name is used.
                hourly_precip = [0.0] * 24
                hourly_precip[12] = 2.54  # one wet hour in the middle
                return {
                    "hourly": {
                        "time": [f"2025-04-01T{hour:02d}:00" for hour in range(24)],
                        "temperature_2m_previous_day1": [15.0] * 24,
                        "precipitation_previous_day1": hourly_precip,
                    },
                    "daily": {
                        "time": ["2025-04-01"],
                        "precipitation_probability_max": [78],
                    },
                    "hourly_units": {},
                    "timezone": "America/New_York",
                }

        return _R()

    monkeypatch.setattr(requests, "get", fake_get)

    result = fetch_forecasts.fetch_previous_run_forecast(
        latitude=40.7,
        longitude=-74.0,
        start_date="2025-04-01",
        end_date="2025-04-01",
        lead_days=1,
    )

    assert result is not None
    # Hourly request variables include the per-lead precipitation variable
    hourly_param = captured["params"].get("hourly", "")
    assert "precipitation_previous_day1" in str(hourly_param)
    assert "temperature_2m_previous_day1" in str(hourly_param)
    # Daily request variable is the plain probability field (no _previous_day
    # suffix — Open-Meteo does not support that on daily variables)
    daily_param = captured["params"].get("daily", "")
    assert "precipitation_probability_max" in str(daily_param)

    daily = result.get("daily", {})
    assert daily.get("time") == ["2025-04-01"]
    # Hourly precipitation summed then converted: 2.54 mm / 25.4 = 0.1 in
    assert daily["precipitation_sum_in"][0] == pytest.approx(0.1, rel=1e-3)
    # Probability is kept as percent in the raw payload (0-100)
    assert daily["precipitation_probability_max"][0] == 78


def test_archive_previous_run_precipitation_writes_expected_row(tmp_path):
    """`archive_previous_run_precipitation` must persist a normalized row."""
    from src import station_truth

    precip_dir = tmp_path / "precip_archive"
    snapshot = {
        "date": "2025-04-01",
        "precipitation_sum_in": 0.12,
        "precipitation_probability_max": 78,
        "lead_days": 1,
        "as_of_utc": "2025-03-31T12:00:00+00:00",
        "forecast_source": "open_meteo_previous_runs",
    }

    station_truth.archive_previous_run_precipitation(
        city="New York",
        snapshot=snapshot,
        base_dir=precip_dir,
    )

    df = pd.read_csv(precip_dir / "new_york.csv")
    assert df.iloc[0]["date"] == "2025-04-01"
    assert df.iloc[0]["forecast_amount_in"] == pytest.approx(0.12, rel=1e-3)
    # Probability percent (0-100) is converted to a fraction (0-1) on write
    assert df.iloc[0]["forecast_prob_any_rain"] == pytest.approx(0.78, rel=1e-3)
    assert df.iloc[0]["forecast_lead_days"] == 1
    assert df.iloc[0]["forecast_source"] == "open_meteo_previous_runs"


if __name__ == "__main__":
    unittest.main()
