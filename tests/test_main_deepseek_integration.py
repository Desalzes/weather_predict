import sys
import types
import unittest
from unittest.mock import Mock, patch

import main


def _config() -> dict:
    return {
        "locations": [{"name": "New York", "lat": 40.71, "lon": -74.01}],
        "forecast_hours": 72,
        "min_edge_threshold": 0.05,
        "uncertainty_std_f": 2.0,
        "scan_interval_minutes": 30,
        "enable_kalshi": True,
        "enable_polymarket": False,
        "enable_ensemble": False,
        "enable_calibration": False,
        "enable_hrrr": False,
        "enable_paper_trading": True,
        "paper_trade_auto_settle": False,
        "paper_trade_contracts": 1.0,
        "paper_trade_ledger_path": "data/paper_trades/ledger.csv",
        "paper_trade_summary_path": "data/paper_trades/summary.json",
        "enable_deepseek_worker": True,
        "deepseek_trade_mode": "paper_gate",
        "strategy_policy_path": "strategy/strategy_policy.json",
    }


class MainDeepSeekIntegrationTests(unittest.TestCase):
    def test_run_scan_gates_paper_trades_to_deepseek_approved_subset(self):
        approved = {
            "source": "kalshi",
            "ticker": "KX1",
            "market_question": "Question 1",
            "city": "New York",
            "market_type": "high",
            "market_date": "2026-04-04",
            "outcome": "<65F",
            "direction": "BUY",
            "edge": 0.18,
            "abs_edge": 0.18,
            "our_probability": 0.62,
            "market_price": 0.41,
            "forecast_value_f": 64.8,
            "volume24hr": 1200,
            "hours_to_settlement": 6.0,
        }
        rejected = {
            "source": "kalshi",
            "ticker": "KX2",
            "market_question": "Question 2",
            "city": "New York",
            "market_type": "high",
            "market_date": "2026-04-04",
            "outcome": "<66F",
            "direction": "BUY",
            "edge": 0.1,
            "abs_edge": 0.1,
            "our_probability": 0.58,
            "market_price": 0.48,
            "forecast_value_f": 65.2,
            "volume24hr": 1200,
            "hours_to_settlement": 6.0,
        }

        log_paper_trades = Mock(
            return_value={
                "new_trades": 1,
                "skipped_existing_open_positions": 0,
            }
        )
        fake_modules = {
            "src.fetch_forecasts": types.SimpleNamespace(
                fetch_multi_location=lambda locations, forecast_hours: {"New York": {"hourly": {"time": [], "temperature_2m": []}}},
                fetch_ensemble_multi_location=lambda locations, forecast_hours: {},
            ),
            "src.station_truth": types.SimpleNamespace(
                archive_forecast_snapshot=lambda city, forecast, ensemble: None
            ),
            "src.fetch_kalshi": types.SimpleNamespace(
                fetch_weather_markets=lambda pages=3: [{"ticker": "KX1"}, {"ticker": "KX2"}]
            ),
            "src.matcher": types.SimpleNamespace(
                match_kalshi_markets=lambda *args, **kwargs: [approved, rejected],
                format_report=lambda opportunities: "report",
            ),
            "src.deepseek_worker": types.SimpleNamespace(
                maybe_run_deepseek_review=lambda opportunities, config, scan_timestamp, strategy_policy=None: {
                    "status": "completed",
                    "approved_count": 1,
                    "decisions": [
                        {
                            "opportunity_id": "kalshi|KX1|2026-04-04|<65F|BUY",
                            "decision": "approve",
                        },
                        {
                            "opportunity_id": "kalshi|KX2|2026-04-04|<66F|BUY",
                            "decision": "reject",
                        },
                    ],
                },
                approved_opportunities_from_review=lambda opportunities, review: [approved],
                format_review_summary=lambda review: "deepseek summary",
                deepseek_trade_mode=lambda config: "paper_gate",
            ),
            "src.strategy_policy": types.SimpleNamespace(
                load_strategy_policy=lambda path_value=None: (
                    {
                        "selection": {
                            "sources": ["kalshi"],
                            "min_abs_edge": 0.08,
                            "min_volume24hr": 500,
                            "max_candidates_per_scan": 5,
                            "max_hours_to_settlement": 24,
                            "allowed_market_types": ["high"],
                            "allowed_cities": [],
                            "blocked_cities": [],
                        }
                    },
                    "strategy/strategy_policy.json",
                ),
                filter_opportunities_for_policy=lambda opportunities, policy: opportunities,
            ),
            "src.paper_trading": types.SimpleNamespace(
                format_paper_trade_summary=lambda summary: "paper summary",
                log_paper_trades=log_paper_trades,
                refresh_station_actuals_for_open_trades=Mock(),
                settle_paper_trades=Mock(),
            ),
        }

        with patch.dict(sys.modules, fake_modules):
            result = main.run_scan(config=_config())

        log_paper_trades.assert_called_once_with(
            [approved],
            scan_timestamp=result["timestamp"],
            ledger_path="data/paper_trades/ledger.csv",
            contracts=1.0,
        )
        self.assertEqual(result["opportunities"], [approved, rejected])
        self.assertEqual(result["trade_opportunities"], [approved])
        self.assertEqual(result["deepseek_review"]["status"], "completed")

    def test_run_scan_falls_back_to_policy_candidates_between_deepseek_reviews(self):
        candidate = {
            "source": "kalshi",
            "ticker": "KX3",
            "market_question": "Question 3",
            "city": "New York",
            "market_type": "high",
            "market_date": "2026-04-04",
            "outcome": "<67F",
            "direction": "BUY",
            "edge": 0.12,
            "abs_edge": 0.12,
            "our_probability": 0.6,
            "market_price": 0.48,
            "forecast_value_f": 64.9,
            "volume24hr": 1600,
            "hours_to_settlement": 6.0,
        }

        log_paper_trades = Mock(
            return_value={
                "new_trades": 1,
                "skipped_existing_open_positions": 0,
            }
        )
        fake_modules = {
            "src.fetch_forecasts": types.SimpleNamespace(
                fetch_multi_location=lambda locations, forecast_hours: {"New York": {"hourly": {"time": [], "temperature_2m": []}}},
                fetch_ensemble_multi_location=lambda locations, forecast_hours: {},
            ),
            "src.station_truth": types.SimpleNamespace(
                archive_forecast_snapshot=lambda city, forecast, ensemble: None
            ),
            "src.fetch_kalshi": types.SimpleNamespace(
                fetch_weather_markets=lambda pages=3: [{"ticker": "KX3"}]
            ),
            "src.matcher": types.SimpleNamespace(
                match_kalshi_markets=lambda *args, **kwargs: [candidate],
                format_report=lambda opportunities: "report",
            ),
            "src.deepseek_worker": types.SimpleNamespace(
                maybe_run_deepseek_review=lambda opportunities, config, scan_timestamp, strategy_policy=None: {
                    "status": "skipped",
                    "reason": "interval_not_reached",
                    "selected_opportunity_count": 1,
                },
                approved_opportunities_from_review=lambda opportunities, review: [],
                format_review_summary=lambda review: "deepseek summary",
                deepseek_trade_mode=lambda config: "paper_gate",
            ),
            "src.strategy_policy": types.SimpleNamespace(
                load_strategy_policy=lambda path_value=None: (
                    {"selection": {"sources": ["kalshi"]}},
                    "strategy/strategy_policy.json",
                ),
                filter_opportunities_for_policy=lambda opportunities, policy: opportunities,
            ),
            "src.paper_trading": types.SimpleNamespace(
                format_paper_trade_summary=lambda summary: "paper summary",
                log_paper_trades=log_paper_trades,
                refresh_station_actuals_for_open_trades=Mock(),
                settle_paper_trades=Mock(),
            ),
        }

        with patch.dict(sys.modules, fake_modules):
            result = main.run_scan(config=_config())

        log_paper_trades.assert_called_once_with(
            [candidate],
            scan_timestamp=result["timestamp"],
            ledger_path="data/paper_trades/ledger.csv",
            contracts=1.0,
        )
        self.assertEqual(result["trade_opportunities"], [candidate])


if __name__ == "__main__":
    unittest.main()
