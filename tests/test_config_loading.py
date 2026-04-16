import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import main
import src.config as config_module


def _base_example_config() -> dict:
    return {
        "locations": [],
        "forecast_hours": 72,
        "min_edge_threshold": 0.05,
        "uncertainty_std_f": 2.0,
        "scan_interval_minutes": 30,
        "enable_kalshi": False,
        "enable_polymarket": False,
        "enable_ensemble": False,
        "enable_calibration": False,
        "enable_hrrr": False,
        "enable_paper_trading": True,
        "paper_trade_auto_settle": True,
        "paper_trade_contracts": 1.0,
        "paper_trade_ledger_path": "data/paper_trades/ledger.csv",
        "paper_trade_summary_path": "data/paper_trades/summary.json",
        "paper_trade_entry_fee_per_contract": 0.01,
        "paper_trade_settlement_fee_per_contract": 0.02,
        "enable_deepseek_worker": True,
        "deepseek_trade_mode": "paper_gate",
        "deepseek_review_interval_scans": 3,
        "deepseek_max_opportunities_per_review": 5,
        "deepseek_min_abs_edge": 0.08,
        "strategy_policy_path": "strategy/strategy_policy.json",
        "deepseek_state_path": "data/deepseek_worker/state.json",
        "deepseek_reviews_dir": "data/deepseek_worker/reviews",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class ConfigLoadingTests(unittest.TestCase):
    def test_run_scan_inherits_missing_paper_trade_defaults_from_example_config(self):
        run_dir = Path("data") / "test_runs" / f"config_loading_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        example_path = run_dir / "config.example.json"
        local_path = run_dir / "config.json"
        _write_json(example_path, _base_example_config())
        _write_json(
            local_path,
            {
                "locations": [],
                "enable_kalshi": False,
                "enable_polymarket": False,
            },
        )

        log_paper_trades = Mock(
            side_effect=lambda *args, **kwargs: call_order.append("log")
            or {
                "new_trades": 0,
                "skipped_existing_open_positions": 0,
            }
        )
        refresh_station_actuals_for_open_trades = Mock(
            side_effect=lambda *args, **kwargs: call_order.append("refresh")
            or {
                "attempted": False,
                "city_windows": [],
                "cutoff_date": "2026-04-02",
                "refresh_source": None,
            }
        )
        settle_paper_trades = Mock(
            side_effect=lambda *args, **kwargs: call_order.append("settle")
            or {
                "total_trades": 0,
                "open_trades": 0,
                "settled_trades": 0,
                "newly_settled_trades": 0,
                "total_fees": 0.0,
                "total_pnl": 0.0,
                "roi": None,
                "win_rate": None,
            }
        )
        call_order = []

        fake_modules = {
            "src.fetch_forecasts": types.SimpleNamespace(
                fetch_multi_location=lambda locations, forecast_hours: {},
                fetch_ensemble_multi_location=lambda locations, forecast_hours: {},
            ),
            "src.station_truth": types.SimpleNamespace(
                archive_forecast_snapshot=lambda city, forecast, ensemble: None
            ),
            "src.matcher": types.SimpleNamespace(
                format_report=lambda opportunities: "report"
            ),
            "src.deepseek_worker": types.SimpleNamespace(
                maybe_run_deepseek_review=lambda opportunities, config, scan_timestamp, strategy_policy=None: {
                    "status": "completed",
                    "approved_count": 0,
                    "decisions": [],
                },
                approved_opportunities_from_review=lambda opportunities, review: [],
                format_review_summary=lambda review: "deepseek summary",
                deepseek_trade_mode=lambda config: "paper_gate",
            ),
            "src.strategy_policy": types.SimpleNamespace(
                load_strategy_policy=lambda path_value=None: ({}, Path("strategy/strategy_policy.json")),
                filter_opportunities_for_policy=lambda opportunities, policy: opportunities,
            ),
            "src.paper_trading": types.SimpleNamespace(
                format_paper_trade_summary=lambda summary: "paper summary",
                log_paper_trades=log_paper_trades,
                refresh_station_actuals_for_open_trades=refresh_station_actuals_for_open_trades,
                settle_paper_trades=settle_paper_trades,
            ),
        }

        with patch.object(config_module, "EXAMPLE_CONFIG_PATH", example_path):
            config = main.load_config(str(local_path))

        self.assertTrue(config["enable_paper_trading"])
        self.assertTrue(config["paper_trade_auto_settle"])
        self.assertEqual(config["paper_trade_ledger_path"], "data/paper_trades/ledger.csv")
        self.assertEqual(config["paper_trade_summary_path"], "data/paper_trades/summary.json")
        self.assertEqual(config["paper_trade_entry_fee_per_contract"], 0.01)
        self.assertEqual(config["paper_trade_settlement_fee_per_contract"], 0.02)
        self.assertTrue(config["enable_deepseek_worker"])
        self.assertEqual(config["deepseek_trade_mode"], "paper_gate")
        self.assertEqual(config["deepseek_review_interval_scans"], 3)
        self.assertEqual(config["deepseek_max_opportunities_per_review"], 5)
        self.assertAlmostEqual(float(config["deepseek_min_abs_edge"]), 0.08)
        self.assertEqual(config["strategy_policy_path"], "strategy/strategy_policy.json")

        with patch.dict(sys.modules, fake_modules):
            result = main.run_scan(config=config)

        log_paper_trades.assert_called_once()
        args, kwargs = log_paper_trades.call_args
        self.assertEqual(args[0], [])
        self.assertEqual(kwargs["scan_timestamp"], result["timestamp"])
        self.assertEqual(kwargs["ledger_path"], "data/paper_trades/ledger.csv")
        self.assertEqual(kwargs["contracts"], 1.0)
        settle_paper_trades.assert_called_once_with(
            ledger_path="data/paper_trades/ledger.csv",
            summary_path="data/paper_trades/summary.json",
        )
        refresh_station_actuals_for_open_trades.assert_called_once_with(
            config=config,
            ledger_path="data/paper_trades/ledger.csv",
        )
        self.assertEqual(call_order, ["log", "refresh", "settle"])
        self.assertEqual(result["trade_opportunities"], [])
        self.assertEqual(result["deepseek_review"]["status"], "completed")

    def test_run_scan_preserves_explicit_local_disable_for_paper_trading(self):
        run_dir = Path("data") / "test_runs" / f"config_loading_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        example_path = run_dir / "config.example.json"
        local_path = run_dir / "config.json"
        _write_json(example_path, _base_example_config())
        _write_json(
            local_path,
            {
                "locations": [],
                "enable_kalshi": False,
                "enable_polymarket": False,
                "enable_paper_trading": False,
            },
        )

        log_paper_trades = Mock()
        refresh_station_actuals_for_open_trades = Mock()
        settle_paper_trades = Mock()
        fake_modules = {
            "src.fetch_forecasts": types.SimpleNamespace(
                fetch_multi_location=lambda locations, forecast_hours: {},
                fetch_ensemble_multi_location=lambda locations, forecast_hours: {},
            ),
            "src.station_truth": types.SimpleNamespace(
                archive_forecast_snapshot=lambda city, forecast, ensemble: None
            ),
            "src.matcher": types.SimpleNamespace(
                format_report=lambda opportunities: "report"
            ),
            "src.deepseek_worker": types.SimpleNamespace(
                maybe_run_deepseek_review=lambda opportunities, config, scan_timestamp, strategy_policy=None: {
                    "status": "completed",
                    "approved_count": 0,
                    "decisions": [],
                },
                approved_opportunities_from_review=lambda opportunities, review: [],
                format_review_summary=lambda review: "deepseek summary",
                deepseek_trade_mode=lambda config: "paper_gate",
            ),
            "src.strategy_policy": types.SimpleNamespace(
                load_strategy_policy=lambda path_value=None: ({}, Path("strategy/strategy_policy.json")),
                filter_opportunities_for_policy=lambda opportunities, policy: opportunities,
            ),
            "src.paper_trading": types.SimpleNamespace(
                format_paper_trade_summary=lambda summary: "paper summary",
                log_paper_trades=log_paper_trades,
                refresh_station_actuals_for_open_trades=refresh_station_actuals_for_open_trades,
                settle_paper_trades=settle_paper_trades,
            ),
        }

        with patch.object(config_module, "EXAMPLE_CONFIG_PATH", example_path):
            config = main.load_config(str(local_path))

        self.assertFalse(config["enable_paper_trading"])
        self.assertEqual(config["paper_trade_ledger_path"], "data/paper_trades/ledger.csv")
        self.assertEqual(config["paper_trade_entry_fee_per_contract"], 0.01)
        self.assertEqual(config["paper_trade_settlement_fee_per_contract"], 0.02)
        self.assertTrue(config["enable_deepseek_worker"])

        with patch.dict(sys.modules, fake_modules):
            result = main.run_scan(config=config)

        self.assertIsNone(result["paper_trade_log"])
        self.assertIsNone(result["paper_trade_summary"])
        log_paper_trades.assert_not_called()
        refresh_station_actuals_for_open_trades.assert_not_called()
        settle_paper_trades.assert_not_called()

    def test_settlement_command_uses_inherited_default_paths(self):
        run_dir = Path("data") / "test_runs" / f"config_loading_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        example_path = run_dir / "config.example.json"
        local_path = run_dir / "config.json"
        _write_json(example_path, _base_example_config())
        _write_json(local_path, {"locations": []})

        call_order = []
        refresh_station_actuals_for_open_trades = Mock(
            side_effect=lambda *args, **kwargs: call_order.append("refresh")
            or {
                "attempted": False,
                "city_windows": [],
                "cutoff_date": "2026-04-02",
                "refresh_source": None,
            }
        )
        settle_paper_trades = Mock(
            side_effect=lambda *args, **kwargs: call_order.append("settle")
            or {
                "total_trades": 0,
                "open_trades": 0,
                "settled_trades": 0,
                "newly_settled_trades": 0,
                "total_fees": 0.0,
                "total_pnl": 0.0,
                "roi": None,
                "win_rate": None,
            }
        )
        fake_modules = {
            "src.paper_trading": types.SimpleNamespace(
                format_paper_trade_summary=lambda summary: "paper summary",
                refresh_station_actuals_for_open_trades=refresh_station_actuals_for_open_trades,
                settle_paper_trades=settle_paper_trades,
            )
        }

        with patch.object(config_module, "EXAMPLE_CONFIG_PATH", example_path):
            config = main.load_config(str(local_path))

        self.assertEqual(config["paper_trade_entry_fee_per_contract"], 0.01)
        self.assertEqual(config["paper_trade_settlement_fee_per_contract"], 0.02)

        with patch.dict(sys.modules, fake_modules):
            summary = main.run_paper_trade_settlement(
                config=config,
                as_of_date="2026-04-02",
            )

        settle_paper_trades.assert_called_once_with(
            ledger_path="data/paper_trades/ledger.csv",
            summary_path="data/paper_trades/summary.json",
            as_of_date="2026-04-02",
        )
        refresh_station_actuals_for_open_trades.assert_called_once_with(
            config=config,
            ledger_path="data/paper_trades/ledger.csv",
            as_of_date="2026-04-02",
        )
        self.assertEqual(call_order, ["refresh", "settle"])
        self.assertEqual(summary["total_trades"], 0)


if __name__ == "__main__":
    unittest.main()
