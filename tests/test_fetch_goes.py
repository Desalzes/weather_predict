"""Tests for src/fetch_goes.py -- cloud adjustment math and scaffold."""

import unittest

from src.fetch_goes import (
    compute_cloud_adjustment_f,
    get_goes_forecast_adjustment,
    fetch_cloud_fraction,
    fetch_cloud_fraction_multi,
)


class ComputeCloudAdjustmentTests(unittest.TestCase):
    """Test the cloud-fraction-to-temperature adjustment formula."""

    def test_clear_sky_no_adjustment(self):
        result = compute_cloud_adjustment_f(0.0, 12)
        self.assertEqual(result, 0.0)

    def test_full_cloud_at_peak(self):
        # 100% cloud at noon. Peak mid=12.5, half_width=2.5
        # hour_weight at 12 = 1 - |12-12.5|/2.5 = 0.8
        # adjustment = -5.0 * 1.0 * 0.8 = -4.0
        result = compute_cloud_adjustment_f(1.0, 12)
        self.assertLess(result, 0.0)
        self.assertAlmostEqual(result, -4.0, delta=0.1)

    def test_off_peak_hours_no_adjustment(self):
        # 8am is before peak heating (10-15)
        result = compute_cloud_adjustment_f(1.0, 8)
        self.assertEqual(result, 0.0)

        # 6pm is after peak heating
        result = compute_cloud_adjustment_f(1.0, 18)
        self.assertEqual(result, 0.0)

    def test_partial_cloud_scales_linearly(self):
        full = compute_cloud_adjustment_f(1.0, 12)
        half = compute_cloud_adjustment_f(0.5, 12)
        # Half cloud should give roughly half adjustment
        self.assertAlmostEqual(half, full * 0.5, delta=0.1)

    def test_edge_of_peak_reduced(self):
        # Hour 10 (start of peak) should have less adjustment than hour 12 (mid)
        edge = compute_cloud_adjustment_f(1.0, 10)
        mid = compute_cloud_adjustment_f(1.0, 12)
        self.assertGreater(abs(mid), abs(edge))

    def test_custom_max_adjustment(self):
        # At hour 12, weight=0.8, so -3.0 * 1.0 * 0.8 = -2.4
        result = compute_cloud_adjustment_f(1.0, 12, max_adjustment_f=-3.0)
        self.assertAlmostEqual(result, -2.4, delta=0.1)

    def test_cloud_fraction_clamped_at_1(self):
        # Cloud fraction > 1.0 should be treated as 1.0
        result = compute_cloud_adjustment_f(1.5, 12)
        normal = compute_cloud_adjustment_f(1.0, 12)
        self.assertAlmostEqual(result, normal, places=2)


class GetGoesForecastAdjustmentTests(unittest.TestCase):
    def test_none_data_returns_zero(self):
        adj, source = get_goes_forecast_adjustment(None, 12)
        self.assertEqual(adj, 0.0)
        self.assertEqual(source, "no_goes_data")

    def test_clear_sky_returns_zero(self):
        data = {"cloud_fraction": 0.0}
        adj, source = get_goes_forecast_adjustment(data, 12)
        self.assertEqual(adj, 0.0)
        self.assertEqual(source, "goes_clear_or_off_peak")

    def test_cloudy_at_peak_returns_adjustment(self):
        data = {"cloud_fraction": 0.8}
        adj, source = get_goes_forecast_adjustment(data, 12)
        self.assertLess(adj, 0.0)
        self.assertEqual(source, "goes_cloud_adjustment")

    def test_cloudy_off_peak_returns_zero(self):
        data = {"cloud_fraction": 0.8}
        adj, source = get_goes_forecast_adjustment(data, 20)
        self.assertEqual(adj, 0.0)
        self.assertEqual(source, "goes_clear_or_off_peak")


class FetchCloudFractionScaffoldTests(unittest.TestCase):
    """Verify scaffold returns None gracefully (not yet implemented)."""

    def test_returns_none(self):
        result = fetch_cloud_fraction(40.71, -74.01)
        self.assertIsNone(result)

    def test_multi_returns_none_values(self):
        locs = [
            {"name": "NYC", "lat": 40.71, "lon": -74.01},
            {"name": "LA", "lat": 34.05, "lon": -118.24},
        ]
        results = fetch_cloud_fraction_multi(locs)
        self.assertEqual(len(results), 2)
        self.assertIsNone(results["NYC"])
        self.assertIsNone(results["LA"])


if __name__ == "__main__":
    unittest.main()
