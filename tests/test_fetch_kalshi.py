"""Tests for src/fetch_kalshi.py -- ticker parsing, city codes, market discovery, grouping."""

import time
import unittest
from unittest.mock import patch, MagicMock

from src.fetch_kalshi import (
    KALSHI_CITY_CODES,
    _WEATHER_PREFIXES,
    _parse_temp_ticker,
    _cache,
    _fetch_via_requests,
    fetch_kalshi_rain_markets,
    fetch_weather_markets,
    group_markets_by_city,
)


class ParseTempTickerTests(unittest.TestCase):
    """Test ticker string parsing into components."""

    def test_high_temp_threshold(self):
        result = _parse_temp_ticker("KXHIGHTATL-26APR16-T84")
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "high")
        self.assertEqual(result["city_code"], "ATL")
        self.assertEqual(result["city"], "Atlanta")
        self.assertEqual(result["threshold"], 84.0)

    def test_high_temp_bucket(self):
        result = _parse_temp_ticker("KXHIGHTBOS-26APR16-B63.5")
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "high")
        self.assertEqual(result["city_code"], "BOS")
        self.assertEqual(result["city"], "Boston")
        self.assertEqual(result["threshold"], 63.5)

    def test_low_temp_threshold(self):
        result = _parse_temp_ticker("KXLOWTDAL-26APR16-T55")
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "low")
        self.assertEqual(result["city_code"], "DAL")
        self.assertEqual(result["city"], "Dallas")
        self.assertEqual(result["threshold"], 55.0)

    def test_low_temp_bucket(self):
        result = _parse_temp_ticker("KXLOWTSFO-26APR16-B49.5")
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "low")
        self.assertEqual(result["city_code"], "SFO")
        self.assertEqual(result["city"], "San Francisco")

    def test_unknown_prefix_returns_none(self):
        result = _parse_temp_ticker("SOMETHING-ELSE-T50")
        self.assertIsNone(result)

    def test_rain_ticker_returns_none(self):
        result = _parse_temp_ticker("KXRAINNYC-26APR16")
        self.assertIsNone(result)

    def test_case_insensitive_city_code(self):
        # Ticker parsing uppercases first
        result = _parse_temp_ticker("kxhightatl-26apr16-t84")
        self.assertIsNotNone(result)
        self.assertEqual(result["city"], "Atlanta")

    def test_four_char_city_code(self):
        result = _parse_temp_ticker("KXHIGHTSATX-26APR16-T82")
        self.assertIsNotNone(result)
        self.assertEqual(result["city_code"], "SATX")
        self.assertEqual(result["city"], "San Antonio")

    def test_nola_city_code(self):
        result = _parse_temp_ticker("KXHIGHTNOLA-26APR16-T85")
        self.assertIsNotNone(result)
        self.assertEqual(result["city"], "New Orleans")

    def test_unknown_city_code_returns_raw(self):
        result = _parse_temp_ticker("KXHIGHTXYZ-26APR16-T70")
        self.assertIsNotNone(result)
        self.assertEqual(result["city_code"], "XYZ")
        self.assertEqual(result["city"], "XYZ")  # No mapping, returns raw


class CityCodesTests(unittest.TestCase):
    """Verify completeness of city code mapping."""

    def test_all_20_cities_covered(self):
        cities = set(KALSHI_CITY_CODES.values())
        expected = {
            "New York", "Los Angeles", "Chicago", "Miami", "Houston",
            "Phoenix", "Denver", "Seattle", "Philadelphia", "Dallas",
            "Austin", "Boston", "Atlanta", "Washington DC", "San Francisco",
            "Las Vegas", "Minneapolis", "San Antonio", "New Orleans",
            "Oklahoma City",
        }
        self.assertTrue(expected.issubset(cities), f"Missing: {expected - cities}")

    def test_codes_are_uppercase(self):
        for code in KALSHI_CITY_CODES:
            self.assertEqual(code, code.upper(), f"Code '{code}' should be uppercase")

    def test_alternate_codes_map_correctly(self):
        # Some cities have multiple codes
        self.assertEqual(KALSHI_CITY_CODES["DFW"], KALSHI_CITY_CODES["DAL"])  # Dallas
        self.assertEqual(KALSHI_CITY_CODES["MSP"], KALSHI_CITY_CODES["MIN"])  # Minneapolis
        self.assertEqual(KALSHI_CITY_CODES["SAT"], KALSHI_CITY_CODES["SATX"])  # San Antonio
        self.assertEqual(KALSHI_CITY_CODES["MSY"], KALSHI_CITY_CODES["NOLA"])  # New Orleans
        self.assertEqual(KALSHI_CITY_CODES["DCA"], KALSHI_CITY_CODES["DC"])  # Washington DC


class WeatherPrefixTests(unittest.TestCase):
    def test_expected_prefixes(self):
        self.assertIn("KXHIGHT", _WEATHER_PREFIXES)
        self.assertIn("KXLOWT", _WEATHER_PREFIXES)
        self.assertIn("KXRAIN", _WEATHER_PREFIXES)
        self.assertIn("KXSNOW", _WEATHER_PREFIXES)


class FetchViaRequestsTests(unittest.TestCase):
    """Test unauthenticated trade fetching."""

    def test_extracts_weather_tickers_from_trades(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "trades": [
                {"ticker": "KXHIGHTATL-26APR16-T84"},
                {"ticker": "KXLOWTBOS-26APR16-B49.5"},
                {"ticker": "INXD-26APR16-T5000"},  # non-weather
            ],
            "cursor": None,
        }

        with patch("src.fetch_kalshi.requests.get", return_value=mock_response):
            tickers = _fetch_via_requests(pages=1, per_page=100)

        self.assertEqual(len(tickers), 2)
        self.assertIn("KXHIGHTATL-26APR16-T84", tickers)
        self.assertIn("KXLOWTBOS-26APR16-B49.5", tickers)

    def test_api_failure_returns_partial(self):
        import requests as req

        with patch("src.fetch_kalshi.requests.get", side_effect=req.RequestException("timeout")):
            tickers = _fetch_via_requests(pages=1, per_page=100)

        self.assertEqual(len(tickers), 0)

    def test_pagination_with_cursor(self):
        resp1 = MagicMock()
        resp1.raise_for_status.return_value = None
        resp1.json.return_value = {
            "trades": [{"ticker": "KXHIGHTATL-26APR16-T84"}],
            "cursor": "next_page",
        }
        resp2 = MagicMock()
        resp2.raise_for_status.return_value = None
        resp2.json.return_value = {
            "trades": [{"ticker": "KXLOWTBOS-26APR16-B49.5"}],
            "cursor": None,
        }

        with patch("src.fetch_kalshi.requests.get", side_effect=[resp1, resp2]), \
             patch("src.fetch_kalshi.time.sleep"):
            tickers = _fetch_via_requests(pages=2, per_page=100)

        self.assertEqual(len(tickers), 2)


class FetchWeatherMarketsTests(unittest.TestCase):
    def setUp(self):
        _cache["markets"] = None
        _cache["ts"] = 0.0

    def test_cache_hit(self):
        _cache["markets"] = [{"ticker": "cached", "source": "kalshi"}]
        _cache["ts"] = time.monotonic()

        result = fetch_weather_markets()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "cached")

    def test_filters_out_closed_markets(self):
        _cache["markets"] = None
        _cache["ts"] = 0.0

        # Mock _get_client to return None (unauthenticated)
        # Mock requests to return one weather ticker
        trade_resp = MagicMock()
        trade_resp.raise_for_status.return_value = None
        trade_resp.json.return_value = {
            "trades": [{"ticker": "KXHIGHTATL-26APR16-T84"}],
            "cursor": None,
        }

        market_resp = MagicMock()
        market_resp.raise_for_status.return_value = None
        market_resp.json.return_value = {
            "market": {
                "ticker": "KXHIGHTATL-26APR16-T84",
                "title": "Atlanta High Temp",
                "status": "closed",  # should be filtered
                "last_price": 0.5,
                "yes_ask": 0.55,
                "yes_bid": 0.45,
                "volume_24h": 1000,
                "close_time": "2026-04-16T23:00:00Z",
            }
        }

        with patch("src.fetch_kalshi._get_client", return_value=None), \
             patch("src.fetch_kalshi.requests.get", side_effect=[trade_resp, market_resp]), \
             patch("src.fetch_kalshi.time.sleep"):
            result = fetch_weather_markets(pages=1)

        # Closed markets should be filtered out
        self.assertEqual(len(result), 0)


class GroupMarketsByCityTests(unittest.TestCase):
    def test_groups_correctly(self):
        markets = [
            {"city": "Atlanta", "type": "high", "threshold": 84, "ticker": "A"},
            {"city": "Atlanta", "type": "high", "threshold": 86, "ticker": "B"},
            {"city": "Atlanta", "type": "low", "threshold": 55, "ticker": "C"},
            {"city": "Boston", "type": "high", "threshold": 60, "ticker": "D"},
        ]
        grouped = group_markets_by_city(markets)

        self.assertIn("Atlanta", grouped)
        self.assertIn("Boston", grouped)
        self.assertEqual(len(grouped["Atlanta"]["high"]), 2)
        self.assertEqual(len(grouped["Atlanta"]["low"]), 1)
        self.assertEqual(len(grouped["Boston"]["high"]), 1)

    def test_sorts_by_threshold(self):
        markets = [
            {"city": "Phoenix", "type": "high", "threshold": 90, "ticker": "A"},
            {"city": "Phoenix", "type": "high", "threshold": 86, "ticker": "B"},
            {"city": "Phoenix", "type": "high", "threshold": 88, "ticker": "C"},
        ]
        grouped = group_markets_by_city(markets)
        thresholds = [m["threshold"] for m in grouped["Phoenix"]["high"]]
        self.assertEqual(thresholds, [86, 88, 90])

    def test_skips_missing_city_or_type(self):
        markets = [
            {"city": None, "type": "high", "ticker": "X"},
            {"city": "Boston", "type": None, "ticker": "Y"},
            {"city": "Boston", "type": "high", "threshold": 60, "ticker": "Z"},
        ]
        grouped = group_markets_by_city(markets)
        self.assertEqual(len(grouped), 1)
        self.assertIn("Boston", grouped)


class FetchKalshiRainMarketsTests(unittest.TestCase):
    """Tests for the per-series KXRAIN discovery fetcher.

    The existing `fetch_weather_markets` call uses `/markets/trades`, which
    only returns recently-traded markets. KXRAIN listings rarely trade so
    they're invisible to that pathway. `fetch_kalshi_rain_markets` must hit
    `/markets?series_ticker=KXRAIN{CODE}&status=open` per city to pick them
    up.
    """

    def _mock_markets_response(self, markets):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"markets": markets, "cursor": None}
        return resp

    def test_fetch_kalshi_rain_markets_hits_per_series_endpoint(self):
        """One API call per city, each with the correct series_ticker param."""
        resp = self._mock_markets_response([])

        with patch("src.fetch_kalshi._get_client", return_value=None), \
             patch("src.fetch_kalshi.requests.get", return_value=resp) as mock_get, \
             patch("src.fetch_kalshi.time.sleep"):
            result = fetch_kalshi_rain_markets(cities=["New York", "Chicago"], limit=25)

        self.assertEqual(result, [])
        self.assertEqual(mock_get.call_count, 2)

        # Verify each call hit /markets with the correct series_ticker
        called_series = set()
        for call in mock_get.call_args_list:
            args, kwargs = call
            url = args[0] if args else kwargs.get("url")
            self.assertIn("/markets", url)
            self.assertNotIn("/markets/trades", url)
            params = kwargs.get("params", {})
            self.assertEqual(params.get("status"), "open")
            self.assertEqual(params.get("limit"), 25)
            called_series.add(params.get("series_ticker"))

        self.assertIn("KXRAINNYC", called_series)
        self.assertIn("KXRAINCHI", called_series)

    def test_fetch_kalshi_rain_markets_parses_ticker_date(self):
        """Ticker KXRAINNYC-26APR22-T0 must resolve to city=New York, market_date=2026-04-22."""
        sample_market = {
            "ticker": "KXRAINNYC-26APR22-T0",
            "title": "Will it **rain** in New York City on Tuesday?",
            "yes_sub_title": "Rain in NYC",
            "no_sub_title": "No Rain in NYC",
            "close_time": "2026-04-23T03:59:00Z",
            "yes_ask_dollars": "0.4500",
            "yes_bid_dollars": "0.4200",
            "volume_24h_fp": "1500",
            "status": "open",
        }
        resp = self._mock_markets_response([sample_market])

        with patch("src.fetch_kalshi._get_client", return_value=None), \
             patch("src.fetch_kalshi.requests.get", return_value=resp), \
             patch("src.fetch_kalshi.time.sleep"):
            result = fetch_kalshi_rain_markets(cities=["New York"])

        self.assertEqual(len(result), 1)
        m = result[0]
        self.assertEqual(m["ticker"], "KXRAINNYC-26APR22-T0")
        self.assertEqual(m["city"], "New York")
        self.assertEqual(m["market_date"], "2026-04-22")
        self.assertEqual(m["close_time"], "2026-04-23T03:59:00Z")
        self.assertEqual(m["source"], "kalshi")
        self.assertAlmostEqual(m["yes_ask"], 0.45)

    def test_fetch_kalshi_rain_markets_handles_empty_city(self):
        """When the API returns markets=[], return [] for that city without raising."""
        resp = self._mock_markets_response([])

        with patch("src.fetch_kalshi._get_client", return_value=None), \
             patch("src.fetch_kalshi.requests.get", return_value=resp), \
             patch("src.fetch_kalshi.time.sleep"):
            result = fetch_kalshi_rain_markets(cities=["Phoenix"])

        self.assertEqual(result, [])

    def test_fetch_kalshi_rain_markets_handles_http_error(self):
        """requests.RequestException during fetch must not raise -- skip the city."""
        import requests as req

        with patch("src.fetch_kalshi._get_client", return_value=None), \
             patch("src.fetch_kalshi.requests.get", side_effect=req.RequestException("boom")), \
             patch("src.fetch_kalshi.time.sleep"):
            result = fetch_kalshi_rain_markets(cities=["New York"])

        self.assertEqual(result, [])

    def test_fetch_kalshi_rain_markets_blank_yes_ask_preserved(self):
        """If yes_ask is null/blank in the API payload, the returned dict must keep
        yes_ask as None (not coerced to 0.0). The rain matcher skips these."""
        sample_market = {
            "ticker": "KXRAINNYC-26APR20-T0",
            "title": "Will it **rain** in New York City on Monday?",
            "yes_sub_title": "Rain in NYC",
            "close_time": "2026-04-21T03:59:00Z",
            "yes_ask_dollars": None,
            "yes_bid_dollars": None,
            "volume_24h_fp": "0",
            "status": "open",
        }
        resp = self._mock_markets_response([sample_market])

        with patch("src.fetch_kalshi._get_client", return_value=None), \
             patch("src.fetch_kalshi.requests.get", return_value=resp), \
             patch("src.fetch_kalshi.time.sleep"):
            result = fetch_kalshi_rain_markets(cities=["New York"])

        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["yes_ask"])

    def test_fetch_kalshi_rain_markets_default_cities_covers_map(self):
        """When cities=None the function iterates over a de-duplicated view of
        KALSHI_CITY_CODES (one call per unique city name)."""
        resp = self._mock_markets_response([])

        with patch("src.fetch_kalshi._get_client", return_value=None), \
             patch("src.fetch_kalshi.requests.get", return_value=resp) as mock_get, \
             patch("src.fetch_kalshi.time.sleep"):
            fetch_kalshi_rain_markets()

        unique_cities = set(KALSHI_CITY_CODES.values())
        self.assertEqual(mock_get.call_count, len(unique_cities))

    def test_fetch_kalshi_rain_markets_skips_unknown_city(self):
        """Cities not in the code map are skipped gracefully with no API call."""
        resp = self._mock_markets_response([])

        with patch("src.fetch_kalshi._get_client", return_value=None), \
             patch("src.fetch_kalshi.requests.get", return_value=resp) as mock_get, \
             patch("src.fetch_kalshi.time.sleep"):
            result = fetch_kalshi_rain_markets(cities=["Atlantis"])

        self.assertEqual(result, [])
        self.assertEqual(mock_get.call_count, 0)

    def test_fetch_kalshi_rain_markets_skips_malformed_ticker(self):
        """Markets whose ticker date portion fails to parse are dropped
        (debug log), but other markets in the same response survive."""
        bad_market = {
            "ticker": "KXRAINNYC-99ZZZ00-T0",  # month 'ZZZ' is invalid
            "close_time": "2026-04-23T03:59:00Z",
            "yes_ask_dollars": "0.3000",
            "status": "open",
        }
        good_market = {
            "ticker": "KXRAINNYC-26APR22-T0",
            "close_time": "2026-04-23T03:59:00Z",
            "yes_ask_dollars": "0.5000",
            "status": "open",
        }
        resp = self._mock_markets_response([bad_market, good_market])

        with patch("src.fetch_kalshi._get_client", return_value=None), \
             patch("src.fetch_kalshi.requests.get", return_value=resp), \
             patch("src.fetch_kalshi.time.sleep"):
            result = fetch_kalshi_rain_markets(cities=["New York"])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "KXRAINNYC-26APR22-T0")


if __name__ == "__main__":
    unittest.main()
