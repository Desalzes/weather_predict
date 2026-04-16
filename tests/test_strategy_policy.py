import unittest

from src.strategy_policy import filter_opportunities_for_policy


class StrategyPolicyTests(unittest.TestCase):
    def test_filter_opportunities_for_policy_applies_thresholds_and_limits(self):
        policy = {
            "selection": {
                "sources": ["kalshi"],
                "min_abs_edge": 0.08,
                "min_volume24hr": 1000,
                "max_candidates_per_scan": 2,
                "max_hours_to_settlement": 12,
                "allowed_market_types": ["high"],
                "allowed_cities": ["new york", "boston"],
                "blocked_cities": ["boston"],
            }
        }
        opportunities = [
            {
                "source": "kalshi",
                "city": "New York",
                "market_type": "high",
                "abs_edge": 0.18,
                "volume24hr": 1200,
                "hours_to_settlement": 6.0,
                "ticker": "A",
            },
            {
                "source": "kalshi",
                "city": "Boston",
                "market_type": "high",
                "abs_edge": 0.3,
                "volume24hr": 2000,
                "hours_to_settlement": 4.0,
                "ticker": "B",
            },
            {
                "source": "kalshi",
                "city": "New York",
                "market_type": "low",
                "abs_edge": 0.2,
                "volume24hr": 1500,
                "hours_to_settlement": 4.0,
                "ticker": "C",
            },
            {
                "source": "kalshi",
                "city": "New York",
                "market_type": "high",
                "abs_edge": 0.09,
                "volume24hr": 1200,
                "hours_to_settlement": 20.0,
                "ticker": "D",
            },
        ]

        filtered = filter_opportunities_for_policy(opportunities, policy)

        self.assertEqual([item["ticker"] for item in filtered], ["A"])

    def test_allowed_settlement_rules_filters_bucket_markets(self):
        """allowed_settlement_rules blocks between_inclusive (bucket) trades."""
        policy = {
            "selection": {
                "sources": ["kalshi"],
                "min_abs_edge": 0.05,
                "min_volume24hr": 0,
                "max_candidates_per_scan": 10,
                "allowed_settlement_rules": ["lte", "gt"],
            }
        }
        opportunities = [
            {
                "source": "kalshi",
                "city": "Phoenix",
                "market_type": "high",
                "abs_edge": 0.20,
                "volume24hr": 5000,
                "settlement_rule": "lte",
                "ticker": "THRESHOLD",
            },
            {
                "source": "kalshi",
                "city": "Phoenix",
                "market_type": "high",
                "abs_edge": 0.30,
                "volume24hr": 5000,
                "settlement_rule": "between_inclusive",
                "ticker": "BUCKET",
            },
            {
                "source": "kalshi",
                "city": "Phoenix",
                "market_type": "high",
                "abs_edge": 0.25,
                "volume24hr": 5000,
                "settlement_rule": "gt",
                "ticker": "GT_THRESHOLD",
            },
        ]

        filtered = filter_opportunities_for_policy(opportunities, policy)

        tickers = [item["ticker"] for item in filtered]
        self.assertIn("THRESHOLD", tickers)
        self.assertIn("GT_THRESHOLD", tickers)
        self.assertNotIn("BUCKET", tickers)

    def test_empty_settlement_rules_allows_all(self):
        """When allowed_settlement_rules is empty/absent, all rules pass."""
        policy = {
            "selection": {
                "sources": ["kalshi"],
                "min_abs_edge": 0.05,
                "min_volume24hr": 0,
                "max_candidates_per_scan": 10,
            }
        }
        opportunities = [
            {
                "source": "kalshi",
                "city": "Phoenix",
                "market_type": "high",
                "abs_edge": 0.20,
                "volume24hr": 5000,
                "settlement_rule": "between_inclusive",
                "ticker": "BUCKET",
            },
        ]

        filtered = filter_opportunities_for_policy(opportunities, policy)
        self.assertEqual(len(filtered), 1)


if __name__ == "__main__":
    unittest.main()
