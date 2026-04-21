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

    def test_allowed_position_sides_buy_only(self):
        """allowed_position_sides: [yes] blocks SELL (no) trades."""
        policy = {
            "selection": {
                "sources": ["kalshi"],
                "min_abs_edge": 0.05,
                "min_volume24hr": 0,
                "max_candidates_per_scan": 10,
                "allowed_position_sides": ["yes"],
            }
        }
        opportunities = [
            {
                "source": "kalshi",
                "city": "Phoenix",
                "market_type": "high",
                "abs_edge": 0.20,
                "volume24hr": 5000,
                "position_side": "yes",
                "ticker": "BUY_YES",
            },
            {
                "source": "kalshi",
                "city": "Phoenix",
                "market_type": "high",
                "abs_edge": 0.30,
                "volume24hr": 5000,
                "position_side": "no",
                "ticker": "SELL_NO",
            },
        ]

        filtered = filter_opportunities_for_policy(opportunities, policy)

        tickers = [item["ticker"] for item in filtered]
        self.assertIn("BUY_YES", tickers)
        self.assertNotIn("SELL_NO", tickers)

    def test_empty_position_sides_allows_all(self):
        """When allowed_position_sides is empty/absent, both yes and no pass."""
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
                "position_side": "no",
                "ticker": "SELL",
            },
        ]

        filtered = filter_opportunities_for_policy(opportunities, policy)
        self.assertEqual(len(filtered), 1)


def test_filter_respects_market_category_scope():
    """A policy that declares market_category must only see opportunities
    with the matching market_category field. Legacy opportunities without
    market_category default to 'temperature' to preserve existing behavior."""
    from src.strategy_policy import filter_opportunities_for_policy

    rain_policy = {
        "market_category": "rain",
        "selection": {
            "sources": ["kalshi"],
            "min_abs_edge": 0.10, "min_volume24hr": 0,
            "max_candidates_per_scan": 5,
            "max_hours_to_settlement": 48,
            "allowed_market_types": ["rain_binary"],
            "allowed_position_sides": ["yes"],
            "allowed_cities": ["New York"],
            "blocked_cities": [],
        },
    }
    opps = [
        {"source": "kalshi", "ticker": "KXRAINNYC-26APR21-T0", "market_category": "rain",
         "city": "New York", "market_type": "rain_binary", "position_side": "yes",
         "abs_edge": 0.2, "volume24hr": 1000, "hours_to_settlement": 12,
         "direction": "BUY"},
        {"source": "kalshi", "ticker": "KXHIGHTNYC-26APR21-T75", "market_category": "temperature",
         "city": "New York", "market_type": "high", "position_side": "yes",
         "abs_edge": 0.2, "volume24hr": 1000, "hours_to_settlement": 12,
         "direction": "BUY"},
    ]
    result = filter_opportunities_for_policy(opps, rain_policy)
    assert len(result) == 1
    assert result[0]["market_category"] == "rain"


def test_temperature_policy_without_category_still_accepts_legacy_opps():
    """Policies without market_category should behave exactly as before —
    legacy opportunities (no market_category key) must still be accepted."""
    from src.strategy_policy import filter_opportunities_for_policy

    policy = {
        "selection": {
            "sources": ["kalshi"], "min_abs_edge": 0.0, "min_volume24hr": 0,
            "max_candidates_per_scan": 10, "max_hours_to_settlement": 48,
            "allowed_market_types": ["high", "low"],
            "allowed_position_sides": ["yes", "no"],
            "allowed_cities": [], "blocked_cities": [],
        },
    }
    opps = [
        {"source": "kalshi", "ticker": "KXHIGHT", "city": "New York",
         "market_type": "high", "position_side": "yes",
         "abs_edge": 0.1, "volume24hr": 100, "hours_to_settlement": 12,
         "direction": "BUY"},
    ]
    assert len(filter_opportunities_for_policy(opps, policy)) == 1


def test_temperature_policy_with_category_rejects_rain_opps():
    """When a policy explicitly declares market_category='temperature',
    rain opportunities must be filtered out."""
    from src.strategy_policy import filter_opportunities_for_policy

    temp_policy = {
        "market_category": "temperature",
        "selection": {
            "sources": ["kalshi"], "min_abs_edge": 0.0, "min_volume24hr": 0,
            "max_candidates_per_scan": 10, "max_hours_to_settlement": 48,
            "allowed_market_types": ["high", "low", "rain_binary"],
            "allowed_position_sides": ["yes"],
            "allowed_cities": [], "blocked_cities": [],
        },
    }
    rain_opp = {
        "source": "kalshi", "ticker": "KXRAINNYC-26APR21-T0",
        "market_category": "rain", "city": "New York",
        "market_type": "rain_binary", "position_side": "yes",
        "abs_edge": 0.3, "volume24hr": 0, "hours_to_settlement": 12,
        "direction": "BUY",
    }
    assert filter_opportunities_for_policy([rain_opp], temp_policy) == []


if __name__ == "__main__":
    unittest.main()
