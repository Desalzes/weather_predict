"""Tests for src/fetch_hrrr.py — HRRR blending, cache, and graceful degradation."""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import numpy as np

from src.fetch_hrrr import (
    _temp_to_f,
    _to_utc,
    _resolve_init_time,
    _normalize_locations,
    _location_signature,
    _get_cached_multi_result,
    _store_cached_multi_result,
    _extract_dataset_values,
    _MULTI_CACHE,
    get_hrrr_high_low,
    fetch_hrrr_point_forecast,
    fetch_hrrr_multi,
)


class TempConversionTests(unittest.TestCase):
    """Test _temp_to_f unit conversions."""

    def test_kelvin_explicit(self):
        # 300 K = 26.85 C = 80.33 F
        result = _temp_to_f(300.0, "kelvin")
        self.assertAlmostEqual(result, 80.33, places=1)

    def test_kelvin_inferred_from_magnitude(self):
        # >200 is treated as Kelvin when units empty
        result = _temp_to_f(273.15, "")
        self.assertAlmostEqual(result, 32.0, places=1)

    def test_celsius_explicit(self):
        result = _temp_to_f(0.0, "celsius")
        self.assertAlmostEqual(result, 32.0, places=1)

    def test_fahrenheit_passthrough(self):
        result = _temp_to_f(72.0, "fahrenheit")
        self.assertAlmostEqual(result, 72.0, places=1)

    def test_degc_unit_string(self):
        result = _temp_to_f(100.0, "degC")
        self.assertAlmostEqual(result, 212.0, places=1)


class ToUtcTests(unittest.TestCase):
    def test_none_returns_now(self):
        result = _to_utc(None)
        self.assertIsNotNone(result.tzinfo)

    def test_naive_gets_utc(self):
        naive = datetime(2026, 4, 15, 12, 0, 0)
        result = _to_utc(naive)
        self.assertEqual(result.tzinfo, timezone.utc)
        self.assertEqual(result.hour, 12)

    def test_aware_converted(self):
        eastern = timezone(timedelta(hours=-5))
        aware = datetime(2026, 4, 15, 12, 0, 0, tzinfo=eastern)
        result = _to_utc(aware)
        self.assertEqual(result.hour, 17)


class ResolveInitTimeTests(unittest.TestCase):
    def test_default_lag(self):
        now = datetime(2026, 4, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = _resolve_init_time(now=now)
        # 2-hour default lag: 14:00 - 2 = 12:00
        self.assertEqual(result.hour, 12)
        self.assertEqual(result.minute, 0)

    def test_custom_lag(self):
        now = datetime(2026, 4, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = _resolve_init_time(now=now, availability_lag_hours=4)
        self.assertEqual(result.hour, 10)

    def test_zero_lag(self):
        now = datetime(2026, 4, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = _resolve_init_time(now=now, availability_lag_hours=0)
        self.assertEqual(result.hour, 14)


class NormalizeLocationsTests(unittest.TestCase):
    def test_basic_normalization(self):
        locs = [{"name": "NYC", "lat": "40.71", "lon": "-74.01"}]
        result = _normalize_locations(locs)
        self.assertEqual(result[0]["name"], "NYC")
        self.assertAlmostEqual(result[0]["lat"], 40.71)
        self.assertAlmostEqual(result[0]["lon"], -74.01)

    def test_missing_name_uses_coords(self):
        locs = [{"lat": 40.71, "lon": -74.01}]
        result = _normalize_locations(locs)
        self.assertIn("40.71", result[0]["name"])

    def test_location_signature_rounding(self):
        sig = _location_signature({"name": "Test", "lat": 40.71234567, "lon": -74.01234567})
        self.assertEqual(sig, ("Test", 40.7123, -74.0123))


class CacheTests(unittest.TestCase):
    def setUp(self):
        # Reset cache between tests
        _MULTI_CACHE["run_time"] = None
        _MULTI_CACHE["fxx"] = -1
        _MULTI_CACHE["locations"] = {}
        _MULTI_CACHE["results"] = {}

    def test_cache_miss_on_empty(self):
        run_time = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        locs = _normalize_locations([{"name": "NYC", "lat": 40.71, "lon": -74.01}])
        result = _get_cached_multi_result(run_time, locs, 18)
        self.assertIsNone(result)

    def test_cache_hit_after_store(self):
        run_time = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        locs = _normalize_locations([{"name": "NYC", "lat": 40.71, "lon": -74.01}])
        stored = {"NYC": {"temperature_2m_f": [72.0], "times": ["2026-04-15T12:00:00+00:00"]}}
        _store_cached_multi_result(run_time, locs, 18, stored)

        result = _get_cached_multi_result(run_time, locs, 18)
        self.assertIsNotNone(result)
        self.assertEqual(result["NYC"]["temperature_2m_f"], [72.0])

    def test_cache_miss_on_different_run_time(self):
        run_time1 = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        run_time2 = datetime(2026, 4, 15, 13, 0, 0, tzinfo=timezone.utc)
        locs = _normalize_locations([{"name": "NYC", "lat": 40.71, "lon": -74.01}])
        _store_cached_multi_result(run_time1, locs, 18, {"NYC": None})

        result = _get_cached_multi_result(run_time2, locs, 18)
        self.assertIsNone(result)

    def test_cache_miss_on_higher_fxx(self):
        run_time = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        locs = _normalize_locations([{"name": "NYC", "lat": 40.71, "lon": -74.01}])
        _store_cached_multi_result(run_time, locs, 12, {"NYC": None})

        # Requesting fxx=18 but cache only has fxx=12
        result = _get_cached_multi_result(run_time, locs, 18)
        self.assertIsNone(result)

    def test_cache_hit_on_lower_fxx(self):
        run_time = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)
        locs = _normalize_locations([{"name": "NYC", "lat": 40.71, "lon": -74.01}])
        _store_cached_multi_result(run_time, locs, 18, {"NYC": None})

        # Requesting fxx=12 and cache has fxx=18 — should hit
        result = _get_cached_multi_result(run_time, locs, 12)
        self.assertIsNotNone(result)


class ExtractDatasetValuesTests(unittest.TestCase):
    def test_xarray_like_dataset(self):
        mock_array = MagicMock()
        mock_array.values = np.array([300.0, 301.0])
        mock_array.attrs = {"units": "K"}

        mock_ds = MagicMock()
        mock_ds.data_vars = {"t2m": mock_array}

        values, units = _extract_dataset_values(mock_ds)
        np.testing.assert_array_almost_equal(values, [300.0, 301.0])
        self.assertEqual(units, "K")

    def test_empty_dataset_raises(self):
        mock_array = MagicMock()
        mock_array.values = np.array([])
        mock_array.attrs = {"units": "K"}

        mock_ds = MagicMock()
        mock_ds.data_vars = {"t2m": mock_array}

        with self.assertRaises(ValueError):
            _extract_dataset_values(mock_ds)

    def test_no_data_vars_raises(self):
        mock_ds = MagicMock()
        mock_ds.data_vars = {}

        with self.assertRaises(ValueError):
            _extract_dataset_values(mock_ds)


class GetHrrrHighLowTests(unittest.TestCase):
    def test_basic_high_low(self):
        hrrr_data = {
            "times": [
                "2026-04-15T06:00:00+00:00",
                "2026-04-15T12:00:00+00:00",
                "2026-04-15T18:00:00+00:00",
            ],
            "temperature_2m_f": [55.0, 72.0, 68.0],
        }
        high, low = get_hrrr_high_low(hrrr_data, "2026-04-15", "UTC")
        self.assertEqual(high, 72.0)
        self.assertEqual(low, 55.0)

    def test_empty_data_returns_none(self):
        high, low = get_hrrr_high_low({}, "2026-04-15")
        self.assertIsNone(high)
        self.assertIsNone(low)

    def test_none_data_returns_none(self):
        high, low = get_hrrr_high_low(None, "2026-04-15")
        self.assertIsNone(high)
        self.assertIsNone(low)

    def test_no_matching_date(self):
        hrrr_data = {
            "times": ["2026-04-14T12:00:00+00:00"],
            "temperature_2m_f": [72.0],
        }
        high, low = get_hrrr_high_low(hrrr_data, "2026-04-15", "UTC")
        self.assertIsNone(high)
        self.assertIsNone(low)


class FetchHrrrMultiGracefulDegradationTests(unittest.TestCase):
    """Test that Herbie import failures degrade gracefully."""

    def test_missing_herbie_raises_runtime_error(self):
        """When herbie is not installed, _get_herbie_class raises RuntimeError
        which propagates through fetch_hrrr_multi."""
        with patch("src.fetch_hrrr._get_herbie_class", side_effect=RuntimeError("herbie not installed")):
            with self.assertRaises(RuntimeError):
                fetch_hrrr_multi(
                    [{"name": "NYC", "lat": 40.71, "lon": -74.01}],
                    fxx=6,
                    now=datetime(2026, 4, 15, 14, 0, 0, tzinfo=timezone.utc),
                )

    def test_empty_locations_returns_empty(self):
        result = fetch_hrrr_multi([], fxx=6)
        self.assertEqual(result, {})

    def test_single_point_delegates_to_multi(self):
        """fetch_hrrr_point_forecast is a thin wrapper around fetch_hrrr_multi."""
        mock_result = {
            "__point__": {
                "times": ["2026-04-15T12:00:00+00:00"],
                "temperature_2m_f": [72.0],
            }
        }
        with patch("src.fetch_hrrr.fetch_hrrr_multi", return_value=mock_result) as mock_multi:
            result = fetch_hrrr_point_forecast(40.71, -74.01, fxx=6)
            mock_multi.assert_called_once()
            self.assertEqual(result["temperature_2m_f"], [72.0])


if __name__ == "__main__":
    unittest.main()
