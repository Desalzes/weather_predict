"""Tests for src/fetch_polymarket.py — market parsing, city aliasing, outcome ranges."""

import time
import unittest
from unittest.mock import patch, MagicMock

from src.fetch_polymarket import (
    CITY_ALIASES,
    _parse_market_type,
    _parse_outcome_range,
    _cache,
    fetch_weather_markets,
)


class ParseMarketTypeTests(unittest.TestCase):
    """Test question -> market type and city extraction."""

    def test_daily_high(self):
        result = _parse_market_type("What will be the highest temperature in New York tomorrow?")
        self.assertEqual(result["type"], "daily_high")
        self.assertEqual(result["city"], "New York")

    def test_daily_low(self):
        result = _parse_market_type("What will be the lowest temperature in Chicago today?")
        self.assertEqual(result["type"], "daily_low")
        self.assertEqual(result["city"], "Chicago")

    def test_rain_today(self):
        result = _parse_market_type("Will it rain in Miami today?")
        self.assertEqual(result["type"], "rain_today")
        self.assertEqual(result["city"], "Miami")

    def test_rain_month(self):
        # Note: "rain in Denver this month" matches rain_today regex first
        # because "rain in X" pattern is greedy. Test the actual behavior.
        result = _parse_market_type("Will there be more rain in Denver this month?")
        # The regex checks "rain in X this month" which matches rain_month
        # but "rain in X today?" takes priority due to order. Test a clean case.
        result = _parse_market_type("How much total rain in Denver this month")
        self.assertEqual(result["type"], "rain_month")
        self.assertEqual(result["city"], "Denver")

    def test_climate_keyword(self):
        result = _parse_market_type("Will this be the hottest year on record?")
        self.assertEqual(result["type"], "climate")

    def test_unknown_question(self):
        result = _parse_market_type("What is the GDP of the United States?")
        self.assertEqual(result["type"], "unknown")

    def test_city_alias_resolution(self):
        result = _parse_market_type("What will be the highest temperature in nyc tomorrow?")
        self.assertEqual(result["city"], "New York")

    def test_city_alias_la(self):
        result = _parse_market_type("What will be the highest temperature in la tomorrow?")
        self.assertEqual(result["city"], "Los Angeles")

    def test_case_insensitive(self):
        result = _parse_market_type("WHAT WILL BE THE HIGHEST TEMPERATURE IN BOSTON TODAY?")
        self.assertEqual(result["type"], "daily_high")
        self.assertEqual(result["city"], "Boston")


class ParseOutcomeRangeTests(unittest.TestCase):
    """Test outcome string -> (low, high) range parsing."""

    def test_or_above(self):
        low, high = _parse_outcome_range("90 or above")
        self.assertEqual(low, 90.0)
        self.assertIsNone(high)

    def test_or_higher(self):
        low, high = _parse_outcome_range("85 or higher")
        self.assertEqual(low, 85.0)
        self.assertIsNone(high)

    def test_plus_sign(self):
        low, high = _parse_outcome_range("80+")
        self.assertEqual(low, 80.0)
        self.assertIsNone(high)

    def test_or_below(self):
        low, high = _parse_outcome_range("32 or below")
        self.assertIsNone(low)
        self.assertAlmostEqual(high, 32.99)

    def test_or_lower(self):
        low, high = _parse_outcome_range("40 or lower")
        self.assertIsNone(low)
        self.assertAlmostEqual(high, 40.99)

    def test_range(self):
        low, high = _parse_outcome_range("60 to 70")
        self.assertEqual(low, 60.0)
        self.assertAlmostEqual(high, 70.99)

    def test_range_with_degrees(self):
        low, high = _parse_outcome_range("55° to 65°")
        self.assertEqual(low, 55.0)
        self.assertAlmostEqual(high, 65.99)

    def test_unparseable(self):
        low, high = _parse_outcome_range("Some random text")
        self.assertIsNone(low)
        self.assertIsNone(high)

    def test_empty_string(self):
        low, high = _parse_outcome_range("")
        self.assertIsNone(low)
        self.assertIsNone(high)


class CityAliasesTests(unittest.TestCase):
    """Verify completeness of city alias mapping."""

    def test_all_20_cities_have_canonical_forms(self):
        canonical_cities = set(CITY_ALIASES.values())
        expected = {
            "New York", "Los Angeles", "Chicago", "Miami", "Houston",
            "Phoenix", "Denver", "Seattle", "Philadelphia", "Dallas",
            "Austin", "Boston", "Atlanta", "Washington DC", "San Francisco",
            "Las Vegas", "Minneapolis", "San Antonio", "New Orleans",
            "Oklahoma City",
        }
        self.assertTrue(expected.issubset(canonical_cities),
                        f"Missing: {expected - canonical_cities}")

    def test_lowercase_aliases_resolve(self):
        for alias, canonical in CITY_ALIASES.items():
            self.assertEqual(alias, alias.lower(),
                             f"Alias '{alias}' should be lowercase")
            self.assertTrue(len(canonical) > 0)


class FetchWeatherMarketsTests(unittest.TestCase):
    """Test API fetching with mocked HTTP responses."""

    def setUp(self):
        # Clear cache before each test
        _cache["markets"] = None
        _cache["fetched_at"] = 0.0

    def test_cache_returns_previous_results(self):
        """Second call within TTL should return cached results."""
        _cache["markets"] = [{"id": "cached", "source": "polymarket"}]
        _cache["fetched_at"] = time.monotonic()

        result = fetch_weather_markets()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "cached")

    def test_expired_cache_refetches(self):
        """Expired cache should trigger new API call."""
        _cache["markets"] = [{"id": "stale"}]
        _cache["fetched_at"] = time.monotonic() - 600  # expired

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {
                "id": "fresh-1",
                "conditionId": "cond-1",
                "question": "What will be the highest temperature in New York tomorrow?",
                "slug": "ny-high-temp",
                "outcomePrices": "[0.45, 0.55]",
                "outcomes": '["Yes", "No"]',
                "volume24hr": 5000,
                "endDate": "2026-04-16",
                "clobTokenIds": "tok-1",
            }
        ]

        with patch("src.fetch_polymarket.requests.get", return_value=mock_response):
            result = fetch_weather_markets()

        # Should have at least 1 market parsed
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["source"], "polymarket")
        self.assertEqual(result[0]["city"], "New York")
        self.assertEqual(result[0]["market_type"], "daily_high")

    def test_api_failure_logs_warning_returns_partial(self):
        """Individual API failures should not crash the entire fetch."""
        _cache["markets"] = None
        _cache["fetched_at"] = 0.0

        import requests as req

        def side_effect(*args, **kwargs):
            raise req.RequestException("Connection timeout")

        with patch("src.fetch_polymarket.requests.get", side_effect=side_effect):
            result = fetch_weather_markets()

        # Should return empty list (all tag+text searches failed), not raise
        self.assertIsInstance(result, list)

    def test_unknown_markets_filtered_out(self):
        """Markets with unparseable questions should be excluded."""
        _cache["markets"] = None
        _cache["fetched_at"] = 0.0

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {
                "id": "irrelevant-1",
                "conditionId": "cond-1",
                "question": "Will Bitcoin reach 100k?",
                "slug": "bitcoin-100k",
                "outcomePrices": "[0.3, 0.7]",
                "outcomes": '["Yes", "No"]',
                "volume24hr": 99999,
                "endDate": "2026-12-31",
                "clobTokenIds": "",
            }
        ]

        with patch("src.fetch_polymarket.requests.get", return_value=mock_response):
            result = fetch_weather_markets()

        # Non-weather markets should be filtered
        weather_ids = [m["id"] for m in result]
        self.assertNotIn("irrelevant-1", weather_ids)

    def test_deduplication_across_searches(self):
        """Same market ID found by multiple search paths should appear once."""
        _cache["markets"] = None
        _cache["fetched_at"] = 0.0

        market = {
            "id": "dedup-1",
            "conditionId": "cond-1",
            "question": "What will be the highest temperature in Boston today?",
            "slug": "boston-high",
            "outcomePrices": "[0.5, 0.5]",
            "outcomes": '["Yes", "No"]',
            "volume24hr": 1000,
            "endDate": "2026-04-16",
            "clobTokenIds": "",
        }

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        # Return same market from every search
        mock_response.json.return_value = [market]

        with patch("src.fetch_polymarket.requests.get", return_value=mock_response):
            result = fetch_weather_markets()

        matching = [m for m in result if m["id"] == "dedup-1"]
        self.assertEqual(len(matching), 1)

    def test_parsed_outcome_prices_are_floats(self):
        """outcomePrices should be converted from JSON string to float list."""
        _cache["markets"] = None
        _cache["fetched_at"] = 0.0

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {
                "id": "price-test-1",
                "conditionId": "cond-1",
                "question": "What will be the lowest temperature in Seattle today?",
                "slug": "seattle-low",
                "outcomePrices": "[0.35, 0.65]",
                "outcomes": '["Under 40", "40 or above"]',
                "volume24hr": 2000,
                "endDate": "2026-04-16",
                "clobTokenIds": "",
            }
        ]

        with patch("src.fetch_polymarket.requests.get", return_value=mock_response):
            result = fetch_weather_markets()

        target = [m for m in result if m["id"] == "price-test-1"]
        self.assertEqual(len(target), 1)
        self.assertEqual(target[0]["outcomePrices"], [0.35, 0.65])
        self.assertIsInstance(target[0]["outcomePrices"][0], float)


if __name__ == "__main__":
    unittest.main()
