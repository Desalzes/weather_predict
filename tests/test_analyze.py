"""Tests for src/analyze.py — ensemble spread, event detection, confidence scoring."""

import math
import unittest

import numpy as np

from src.analyze import (
    DEFAULT_THRESHOLDS,
    compute_ensemble_spread,
    detect_weather_events,
    score_prediction_confidence,
)


class ComputeEnsembleSpreadTests(unittest.TestCase):
    """Test ensemble spread statistics computation."""

    def test_multi_member_ensemble(self):
        """2D array (members x timesteps) should produce real stats."""
        ensemble_data = {
            "hourly": {
                "time": ["2026-04-15T00:00", "2026-04-15T01:00", "2026-04-15T02:00"],
                "temperature_2m": [
                    [20.0, 22.0, 24.0],  # member 1
                    [21.0, 23.0, 25.0],  # member 2
                    [19.0, 21.0, 23.0],  # member 3
                ],
            }
        }
        spread = compute_ensemble_spread(ensemble_data)
        self.assertIn("temperature_2m", spread)
        stats = spread["temperature_2m"]

        # Mean should be [20, 22, 24]
        np.testing.assert_array_almost_equal(stats["mean"], [20.0, 22.0, 24.0])
        # Min should be [19, 21, 23]
        np.testing.assert_array_almost_equal(stats["min"], [19.0, 21.0, 23.0])
        # Max should be [21, 23, 25]
        np.testing.assert_array_almost_equal(stats["max"], [21.0, 23.0, 25.0])
        # Std should be 1.0 for all timesteps
        self.assertAlmostEqual(stats["std"][0], 1.0, places=3)
        # IQR should be > 0
        self.assertGreater(stats["iqr"][0], 0.0)

    def test_single_value_series(self):
        """1D array (no ensemble members) should have zero spread."""
        ensemble_data = {
            "hourly": {
                "time": ["2026-04-15T00:00", "2026-04-15T01:00"],
                "temperature_2m": [20.0, 22.0],
            }
        }
        spread = compute_ensemble_spread(ensemble_data)
        stats = spread["temperature_2m"]

        np.testing.assert_array_almost_equal(stats["mean"], [20.0, 22.0])
        np.testing.assert_array_almost_equal(stats["std"], [0.0, 0.0])

    def test_empty_hourly(self):
        spread = compute_ensemble_spread({"hourly": {}})
        self.assertEqual(spread, {})

    def test_time_key_excluded(self):
        """'time' key should not be in spread output."""
        ensemble_data = {
            "hourly": {
                "time": ["t1", "t2"],
                "temp": [10.0, 20.0],
            }
        }
        spread = compute_ensemble_spread(ensemble_data)
        self.assertNotIn("time", spread)
        self.assertIn("temp", spread)

    def test_normalized_spread_computation(self):
        """Normalized spread = std / (|mean| + epsilon)."""
        ensemble_data = {
            "hourly": {
                "time": ["t1"],
                "temp": [
                    [10.0],
                    [20.0],
                ],
            }
        }
        spread = compute_ensemble_spread(ensemble_data)
        mean = spread["temp"]["mean"][0]  # 15.0
        std = spread["temp"]["std"][0]
        ns = spread["temp"]["normalized_spread"][0]
        expected_ns = std / (abs(mean) + 1e-6)
        self.assertAlmostEqual(ns, expected_ns, places=4)


class DetectWeatherEventsTests(unittest.TestCase):
    """Test weather event detection from hourly forecasts."""

    def test_heavy_rain_detected(self):
        hourly = {
            "hourly": {
                "time": [f"t{i}" for i in range(24)],
                "precipitation": [0.0] * 10 + [15.0, 20.0, 12.0] + [0.0] * 11,
                "precipitation_probability": [0.0] * 24,
                "temperature_2m": [20.0] * 24,
                "wind_speed_10m": [10.0] * 24,
                "snowfall": [0.0] * 24,
            }
        }
        events = detect_weather_events(hourly)
        rain_events = [e for e in events if e["event_type"] == "heavy_rain"]
        self.assertGreater(len(rain_events), 0)
        self.assertEqual(rain_events[0]["start_hour"], 10)
        self.assertAlmostEqual(rain_events[0]["peak_value"], 20.0)

    def test_high_wind_detected(self):
        hourly = {
            "hourly": {
                "time": [f"t{i}" for i in range(24)],
                "precipitation": [0.0] * 24,
                "precipitation_probability": [0.0] * 24,
                "temperature_2m": [20.0] * 24,
                "wind_speed_10m": [10.0] * 5 + [60.0, 70.0, 55.0] + [10.0] * 16,
                "snowfall": [0.0] * 24,
            }
        }
        events = detect_weather_events(hourly)
        wind_events = [e for e in events if e["event_type"] == "high_wind"]
        self.assertGreater(len(wind_events), 0)
        self.assertAlmostEqual(wind_events[0]["peak_value"], 70.0)

    def test_temperature_swing_detected(self):
        # Create a sharp swing: 10C to 25C within 7 hours
        temps = [10.0] * 3 + [25.0] * 3 + [10.0] * 18
        hourly = {
            "hourly": {
                "time": [f"t{i}" for i in range(24)],
                "precipitation": [0.0] * 24,
                "precipitation_probability": [0.0] * 24,
                "temperature_2m": temps,
                "wind_speed_10m": [10.0] * 24,
                "snowfall": [0.0] * 24,
            }
        }
        events = detect_weather_events(hourly)
        swing_events = [e for e in events if e["event_type"] == "temperature_swing"]
        self.assertGreater(len(swing_events), 0)
        # Delta should be 15
        self.assertGreaterEqual(swing_events[0]["peak_value"], 15.0)

    def test_no_events_in_calm_weather(self):
        hourly = {
            "hourly": {
                "time": [f"t{i}" for i in range(24)],
                "precipitation": [0.0] * 24,
                "precipitation_probability": [0.0] * 24,
                "temperature_2m": [20.0] * 24,
                "wind_speed_10m": [10.0] * 24,
                "snowfall": [0.0] * 24,
            }
        }
        events = detect_weather_events(hourly)
        self.assertEqual(len(events), 0)

    def test_custom_thresholds(self):
        """Lower rain threshold should trigger more events."""
        hourly = {
            "hourly": {
                "time": [f"t{i}" for i in range(24)],
                "precipitation": [0.0] * 10 + [3.0] + [0.0] * 13,
                "precipitation_probability": [0.0] * 24,
                "temperature_2m": [20.0] * 24,
                "wind_speed_10m": [10.0] * 24,
                "snowfall": [0.0] * 24,
            }
        }
        # Default threshold (10mm) should NOT detect 3mm rain
        events_default = detect_weather_events(hourly)
        rain_default = [e for e in events_default if e["event_type"] == "heavy_rain"]
        self.assertEqual(len(rain_default), 0)

        # Custom threshold (2mm) should detect it
        events_custom = detect_weather_events(hourly, thresholds={"heavy_rain_mm": 2.0})
        rain_custom = [e for e in events_custom if e["event_type"] == "heavy_rain"]
        self.assertGreater(len(rain_custom), 0)

    def test_missing_fields_handled(self):
        """Missing wind/snowfall fields should not crash."""
        hourly = {
            "hourly": {
                "time": [f"t{i}" for i in range(24)],
                "precipitation": [0.0] * 24,
                "precipitation_probability": [0.0] * 24,
                "temperature_2m": [20.0] * 24,
                # wind_speed_10m and snowfall missing
            }
        }
        events = detect_weather_events(hourly)
        # Should not raise — missing fields padded with zeros
        self.assertIsInstance(events, list)

    def test_high_precip_probability_triggers_rain(self):
        """Rain event should also trigger on high probability, not just amount."""
        hourly = {
            "hourly": {
                "time": [f"t{i}" for i in range(24)],
                "precipitation": [0.0] * 24,  # no actual precip
                "precipitation_probability": [0.0] * 10 + [95.0] + [0.0] * 13,
                "temperature_2m": [20.0] * 24,
                "wind_speed_10m": [10.0] * 24,
                "snowfall": [0.0] * 24,
            }
        }
        events = detect_weather_events(hourly)
        rain_events = [e for e in events if e["event_type"] == "heavy_rain"]
        self.assertGreater(len(rain_events), 0)


class ScorePredictionConfidenceTests(unittest.TestCase):
    """Test per-hour confidence scoring."""

    def test_basic_scoring(self):
        hourly = {
            "hourly": {
                "time": [f"t{i}" for i in range(24)],
                "precipitation_probability": [5.0] * 24,
            }
        }
        scores = score_prediction_confidence(hourly)
        self.assertEqual(len(scores), 24)
        # All scores should be in [0, 1]
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_empty_returns_empty(self):
        scores = score_prediction_confidence({"hourly": {}})
        self.assertEqual(scores, [])

    def test_horizon_penalty_increases(self):
        """Later hours should have lower confidence due to horizon penalty."""
        hourly = {
            "hourly": {
                "time": [f"t{i}" for i in range(72)],
                "precipitation_probability": [5.0] * 72,
            }
        }
        scores = score_prediction_confidence(hourly)
        # First hour should be higher confidence than last
        self.assertGreater(scores[0], scores[-1])

    def test_ensemble_spread_penalty(self):
        """High ensemble spread should reduce confidence."""
        hourly = {
            "hourly": {
                "time": ["t0", "t1"],
                "precipitation_probability": [5.0, 5.0],
            }
        }
        # Without ensemble
        scores_base = score_prediction_confidence(hourly)

        # With high-spread ensemble
        ensemble = {
            "hourly": {
                "time": ["t0", "t1"],
                "temperature_2m": [
                    [10.0, 10.0],
                    [30.0, 30.0],
                ],
            }
        }
        scores_with_ensemble = score_prediction_confidence(hourly, ensemble_data=ensemble)

        # Ensemble should reduce scores
        self.assertLessEqual(scores_with_ensemble[0], scores_base[0])

    def test_low_precip_probability_gives_high_confidence(self):
        """Low precipitation probability (<5%) should yield high base confidence (0.95)."""
        hourly = {
            "hourly": {
                "time": ["t0"],
                "precipitation_probability": [2.0],
            }
        }
        scores = score_prediction_confidence(hourly)
        self.assertGreaterEqual(scores[0], 0.9)


class DefaultThresholdsTests(unittest.TestCase):
    """Verify DEFAULT_THRESHOLDS has expected keys and reasonable values."""

    def test_expected_keys(self):
        expected = {
            "heavy_rain_mm", "heavy_rain_prob", "temp_swing_delta",
            "high_wind_kmh", "snowfall_threshold", "heat_extreme_c",
            "cold_extreme_c",
        }
        self.assertEqual(set(DEFAULT_THRESHOLDS.keys()), expected)

    def test_values_are_numeric(self):
        for key, val in DEFAULT_THRESHOLDS.items():
            self.assertIsInstance(val, (int, float), f"{key} should be numeric")

    def test_rain_threshold_positive(self):
        self.assertGreater(DEFAULT_THRESHOLDS["heavy_rain_mm"], 0)

    def test_wind_threshold_positive(self):
        self.assertGreater(DEFAULT_THRESHOLDS["high_wind_kmh"], 0)


if __name__ == "__main__":
    unittest.main()
