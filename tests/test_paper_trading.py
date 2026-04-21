import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.paper_trading import (
    log_paper_trades,
    paper_trade_record_from_opportunity,
    plan_station_actuals_refresh,
    refresh_station_actuals_for_open_trades,
    settle_paper_trades,
)
from src.station_truth import station_actuals_path
from unittest.mock import patch


def _base_opportunity(**overrides):
    opportunity = {
        "source": "kalshi",
        "ticker": "KXHIGHTNYC-26APR01-T65",
        "market_question": "Will the high in New York be below 65F?",
        "city": "New York",
        "market_type": "high",
        "market_date": "2026-04-01",
        "outcome": "<65F",
        "our_probability": 0.58,
        "market_price": 0.42,
        "edge": 0.16,
        "direction": "BUY",
        "forecast_value_f": 64.1,
        "open_meteo_forecast_value_f": 64.8,
        "raw_forecast_value_f": 64.8,
        "forecast_blend_source": "open-meteo",
        "hours_to_settlement": 20.0,
        "uncertainty_std_f": 2.1,
        "raw_probability": 0.61,
        "forecast_calibration_source": "emos",
        "probability_calibration_source": "isotonic",
        "actual_field": "tmax_f",
        "settlement_rule": "lte",
        "settlement_low_f": None,
        "settlement_high_f": 65.0,
        "yes_bid": 0.4,
        "yes_ask": 0.44,
    }
    opportunity.update(overrides)
    return opportunity


def _paper_fee_config(
    entry_fee_per_contract: float = 0.0,
    settlement_fee_per_contract: float = 0.0,
) -> dict:
    return {
        "paper_trade_entry_fee_per_contract": entry_fee_per_contract,
        "paper_trade_settlement_fee_per_contract": settlement_fee_per_contract,
    }


def _write_station_actual_rows(actuals_dir: Path, city: str, rows: list[dict]) -> Path:
    path = station_actuals_path(city, base_dir=actuals_dir)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class PaperTradingTests(unittest.TestCase):
    def test_log_paper_trades_persists_routing_metadata(self):
        run_dir = Path("data") / "test_runs" / f"paper_routing_metadata_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"

        log_paper_trades(
            [
                _base_opportunity(
                    city="Boston",
                    market_type="low",
                    ticker="KXLOWBOS-26APR01-B50.5",
                    market_question="Will the low in Boston land between 50F and 51F?",
                    outcome="50-51F",
                    forecast_value_f=50.2,
                    open_meteo_forecast_value_f=51.4,
                    raw_forecast_value_f=50.2,
                    forecast_blend_source="open-meteo+hrrr",
                    hours_to_settlement=4.5,
                    our_probability=0.63,
                    raw_probability=0.63,
                    forecast_calibration_source="raw_selective_fallback",
                    probability_calibration_source="raw",
                    actual_field="tmin_f",
                    settlement_rule="between_inclusive",
                    settlement_low_f=50.0,
                    settlement_high_f=51.0,
                )
            ],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        ledger = pd.read_csv(ledger_path)

        self.assertEqual(ledger.loc[0, "forecast_calibration_source"], "raw_selective_fallback")
        self.assertEqual(ledger.loc[0, "probability_calibration_source"], "raw")
        self.assertEqual(ledger.loc[0, "forecast_blend_source"], "open-meteo+hrrr")
        self.assertAlmostEqual(float(ledger.loc[0, "hours_to_settlement"]), 4.5, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "open_meteo_forecast_value_f"]), 51.4, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "raw_forecast_value_f"]), 50.2, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "forecast_value_f"]), 50.2, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "raw_probability"]), 0.63, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "our_probability"]), 0.63, places=4)

    def test_paper_trade_record_uses_no_side_fill_from_yes_bid(self):
        record = paper_trade_record_from_opportunity(
            _base_opportunity(
                outcome=">70F",
                edge=-0.11,
                direction="SELL",
                market_price=0.63,
                settlement_rule="gt",
                settlement_low_f=70.0,
                settlement_high_f=None,
                yes_bid=0.62,
                yes_ask=0.64,
            ),
            scan_timestamp="2026-04-01T12:00:00+00:00",
        )

        self.assertEqual(record["position_side"], "no")
        self.assertEqual(record["fee_model"], "flat_fee_per_contract_v1")
        self.assertAlmostEqual(float(record["entry_price"]), 0.38)
        self.assertEqual(record["entry_price_source"], "synthetic_no_ask_from_yes_bid")
        self.assertAlmostEqual(float(record["entry_fee"]), 0.0)
        self.assertAlmostEqual(float(record["settlement_fee"]), 0.0)
        self.assertAlmostEqual(float(record["total_fees"]), 0.0)
        self.assertAlmostEqual(float(record["entry_cost"]), 0.38)

    def test_log_paper_trades_dedupes_existing_open_position(self):
        run_dir = Path("data") / "test_runs" / f"paper_dedupe_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"

        first = log_paper_trades(
            [_base_opportunity()],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )
        second = log_paper_trades(
            [_base_opportunity()],
            scan_timestamp="2026-04-01T12:30:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        ledger = pd.read_csv(ledger_path)

        self.assertEqual(first["new_trades"], 1)
        self.assertEqual(second["new_trades"], 0)
        self.assertEqual(second["skipped_existing_open_positions"], 1)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger.loc[0, "status"], "open")
        self.assertEqual(ledger.loc[0, "position_side"], "yes")

    def test_settle_paper_trades_scores_against_station_truth(self):
        run_dir = Path("data") / "test_runs" / f"paper_settle_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"
        summary_path = run_dir / "paper_trades" / "summary.json"
        actuals_dir = run_dir / "station_actuals"

        _write_station_actual_rows(
            actuals_dir,
            "New York",
            [
                {
                    "date": "2026-04-01",
                    "tmax_f": 64.0,
                    "tmin_f": 49.0,
                }
            ],
        )

        log_paper_trades(
            [_base_opportunity()],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )
        summary = settle_paper_trades(
            ledger_path=ledger_path,
            station_actuals_dir=actuals_dir,
            summary_path=summary_path,
            as_of_date="2026-04-02",
            config=_paper_fee_config(),
        )
        ledger = pd.read_csv(ledger_path)

        self.assertEqual(summary["settled_trades"], 1)
        self.assertEqual(summary["settlement_cutoff_date"], "2026-04-02")
        self.assertEqual(summary["eligible_open_trade_count"], 0)
        self.assertFalse(summary["missing_required_truth_blocker"])
        self.assertEqual(summary["missing_required_truth_trade_count"], 0)
        self.assertEqual(summary["missing_required_truth_date_count"], 0)
        self.assertEqual(summary["missing_required_truth_dates"], [])
        self.assertEqual(summary["missing_required_truth_city_windows"], [])
        self.assertAlmostEqual(float(summary["total_fees"]), 0.0, places=4)
        self.assertAlmostEqual(float(summary["total_pnl"]), 0.56, places=4)
        self.assertEqual(ledger.loc[0, "status"], "settled")
        self.assertEqual(ledger.loc[0, "fee_model"], "flat_fee_per_contract_v1")
        self.assertAlmostEqual(float(ledger.loc[0, "entry_fee"]), 0.0, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "settlement_fee"]), 0.0, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "total_fees"]), 0.0, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "entry_cost"]), 0.44, places=4)
        self.assertEqual(int(ledger.loc[0, "yes_outcome"]), 1)
        self.assertAlmostEqual(float(ledger.loc[0, "actual_value_f"]), 64.0)
        self.assertAlmostEqual(float(ledger.loc[0, "payout"]), 1.0)
        self.assertAlmostEqual(float(ledger.loc[0, "pnl"]), 0.56, places=4)

    def test_settle_paper_trades_uses_recorded_fee_assumptions_for_net_results(self):
        run_dir = Path("data") / "test_runs" / f"paper_settle_fees_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"
        summary_path = run_dir / "paper_trades" / "summary.json"
        actuals_dir = run_dir / "station_actuals"
        fee_config = _paper_fee_config(entry_fee_per_contract=0.02, settlement_fee_per_contract=0.03)

        _write_station_actual_rows(
            actuals_dir,
            "New York",
            [
                {
                    "date": "2026-04-01",
                    "tmax_f": 64.0,
                    "tmin_f": 49.0,
                }
            ],
        )

        log_paper_trades(
            [_base_opportunity()],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=fee_config,
        )

        summary = settle_paper_trades(
            ledger_path=ledger_path,
            station_actuals_dir=actuals_dir,
            summary_path=summary_path,
            as_of_date="2026-04-02",
            config=_paper_fee_config(),
        )
        ledger = pd.read_csv(ledger_path)

        self.assertAlmostEqual(float(ledger.loc[0, "entry_fee_per_contract"]), 0.02, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "settlement_fee_per_contract"]), 0.03, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "entry_fee"]), 0.02, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "settlement_fee"]), 0.03, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "total_fees"]), 0.05, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "entry_cost"]), 0.46, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "pnl"]), 0.51, places=4)
        self.assertAlmostEqual(float(ledger.loc[0, "roi"]), 1.1087, places=4)
        self.assertAlmostEqual(float(summary["total_fees"]), 0.05, places=4)
        self.assertAlmostEqual(float(summary["total_pnl"]), 0.51, places=4)
        self.assertAlmostEqual(float(summary["roi"]), 1.1087, places=4)

    def test_settle_paper_trades_backfills_fee_columns_for_pre_fee_ledger(self):
        run_dir = Path("data") / "test_runs" / f"paper_settle_legacy_fees_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"
        summary_path = run_dir / "paper_trades" / "summary.json"
        actuals_dir = run_dir / "station_actuals"
        fee_config = _paper_fee_config(entry_fee_per_contract=0.02, settlement_fee_per_contract=0.03)

        _write_station_actual_rows(
            actuals_dir,
            "New York",
            [
                {
                    "date": "2026-04-01",
                    "tmax_f": 64.0,
                    "tmin_f": 49.0,
                }
            ],
        )

        log_paper_trades(
            [_base_opportunity()],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        legacy_ledger = pd.read_csv(ledger_path)
        legacy_ledger = legacy_ledger.drop(
            columns=[
                "fee_model",
                "entry_fee_per_contract",
                "entry_fee",
                "settlement_fee_per_contract",
                "settlement_fee",
                "total_fees",
            ]
        )
        legacy_ledger.to_csv(ledger_path, index=False)

        summary = settle_paper_trades(
            ledger_path=ledger_path,
            station_actuals_dir=actuals_dir,
            summary_path=summary_path,
            as_of_date="2026-04-02",
            config=fee_config,
        )
        migrated_ledger = pd.read_csv(ledger_path)

        self.assertIn("fee_model", migrated_ledger.columns)
        self.assertIn("entry_fee_per_contract", migrated_ledger.columns)
        self.assertIn("settlement_fee_per_contract", migrated_ledger.columns)
        self.assertIn("total_fees", migrated_ledger.columns)
        self.assertAlmostEqual(float(migrated_ledger.loc[0, "entry_fee"]), 0.02, places=4)
        self.assertAlmostEqual(float(migrated_ledger.loc[0, "settlement_fee"]), 0.03, places=4)
        self.assertAlmostEqual(float(migrated_ledger.loc[0, "total_fees"]), 0.05, places=4)
        self.assertAlmostEqual(float(migrated_ledger.loc[0, "entry_cost"]), 0.46, places=4)
        self.assertAlmostEqual(float(migrated_ledger.loc[0, "pnl"]), 0.51, places=4)
        self.assertAlmostEqual(float(summary["total_fees"]), 0.05, places=4)
        self.assertAlmostEqual(float(summary["total_pnl"]), 0.51, places=4)

    def test_legacy_ledger_without_routing_columns_still_logs_and_settles_with_unknown_backfill(self):
        run_dir = Path("data") / "test_runs" / f"paper_settle_legacy_routes_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"
        summary_path = run_dir / "paper_trades" / "summary.json"
        actuals_dir = run_dir / "station_actuals"

        _write_station_actual_rows(
            actuals_dir,
            "New York",
            [
                {
                    "date": "2026-04-01",
                    "tmax_f": 64.0,
                    "tmin_f": 49.0,
                }
            ],
        )

        log_paper_trades(
            [_base_opportunity()],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        legacy_ledger = pd.read_csv(ledger_path)
        legacy_ledger = legacy_ledger.drop(
            columns=[
                "raw_probability",
                "open_meteo_forecast_value_f",
                "raw_forecast_value_f",
                "forecast_blend_source",
                "hours_to_settlement",
                "forecast_calibration_source",
                "probability_calibration_source",
            ]
        )
        legacy_ledger.to_csv(ledger_path, index=False)

        log_paper_trades(
            [
                _base_opportunity(
                    ticker="KXHIGHTNYC-26APR01-T66",
                    market_question="Will the high in New York be below 66F?",
                    outcome="<66F",
                    settlement_high_f=66.0,
                    forecast_calibration_source="raw",
                    probability_calibration_source="raw",
                    raw_probability=0.59,
                    our_probability=0.59,
                    forecast_value_f=64.8,
                    raw_forecast_value_f=64.8,
                    open_meteo_forecast_value_f=64.8,
                )
            ],
            scan_timestamp="2026-04-01T13:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        summary = settle_paper_trades(
            ledger_path=ledger_path,
            station_actuals_dir=actuals_dir,
            summary_path=summary_path,
            as_of_date="2026-04-02",
            config=_paper_fee_config(),
        )
        migrated_ledger = pd.read_csv(ledger_path)

        self.assertEqual(summary["settled_trades"], 2)
        self.assertIn("forecast_calibration_source", migrated_ledger.columns)
        self.assertIn("probability_calibration_source", migrated_ledger.columns)
        self.assertIn("forecast_blend_source", migrated_ledger.columns)
        self.assertEqual(migrated_ledger.loc[0, "forecast_calibration_source"], "legacy_unknown")
        self.assertEqual(migrated_ledger.loc[0, "probability_calibration_source"], "legacy_unknown")
        self.assertEqual(migrated_ledger.loc[0, "forecast_blend_source"], "legacy_unknown")
        self.assertTrue(pd.isna(migrated_ledger.loc[0, "raw_probability"]))
        self.assertEqual(migrated_ledger.loc[1, "forecast_calibration_source"], "raw")
        self.assertEqual(migrated_ledger.loc[1, "probability_calibration_source"], "raw")
        self.assertEqual(migrated_ledger.loc[1, "forecast_blend_source"], "open-meteo")
        self.assertIn("legacy_unknown", summary["forecast_calibration_source_breakdown"])
        self.assertIn("raw", summary["forecast_calibration_source_breakdown"])

    def test_settle_paper_trades_summary_breaks_out_profitability_by_forecast_route(self):
        run_dir = Path("data") / "test_runs" / f"paper_route_summary_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"
        summary_path = run_dir / "paper_trades" / "summary.json"
        actuals_dir = run_dir / "station_actuals"

        _write_station_actual_rows(
            actuals_dir,
            "New York",
            [
                {
                    "date": "2026-04-01",
                    "tmax_f": 64.0,
                    "tmin_f": 49.0,
                }
            ],
        )

        opportunities = [
            _base_opportunity(
                ticker="KXHIGHTNYC-26APR01-T65A",
                forecast_calibration_source="emos",
                probability_calibration_source="isotonic",
                our_probability=0.58,
                raw_probability=0.61,
            ),
            _base_opportunity(
                ticker="KXHIGHTNYC-26APR01-T70",
                market_question="Will the high in New York be above 70F?",
                outcome=">70F",
                market_price=0.35,
                edge=0.21,
                direction="BUY",
                forecast_value_f=71.2,
                open_meteo_forecast_value_f=71.2,
                raw_forecast_value_f=71.2,
                our_probability=0.56,
                raw_probability=0.56,
                forecast_calibration_source="raw",
                probability_calibration_source="raw",
                settlement_rule="gt",
                settlement_low_f=70.0,
                settlement_high_f=None,
                yes_bid=0.34,
                yes_ask=0.4,
            ),
            _base_opportunity(
                ticker="KXHIGHTNYC-26APR01-T65B",
                forecast_value_f=64.0,
                open_meteo_forecast_value_f=64.0,
                raw_forecast_value_f=64.0,
                our_probability=0.62,
                raw_probability=0.62,
                forecast_calibration_source="raw_selective_fallback",
                probability_calibration_source="raw",
            ),
        ]

        log_paper_trades(
            opportunities,
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        summary = settle_paper_trades(
            ledger_path=ledger_path,
            station_actuals_dir=actuals_dir,
            summary_path=summary_path,
            as_of_date="2026-04-02",
            config=_paper_fee_config(),
        )

        breakdown = summary["forecast_calibration_source_breakdown"]

        self.assertEqual(set(summary["forecast_calibration_source_breakdown"].keys()), {"emos", "raw", "raw_selective_fallback"})
        self.assertEqual(breakdown["emos"]["total_trades"], 1)
        self.assertEqual(breakdown["emos"]["settled_trades"], 1)
        self.assertAlmostEqual(float(breakdown["emos"]["total_pnl"]), 0.56, places=4)
        self.assertEqual(breakdown["raw"]["total_trades"], 1)
        self.assertEqual(breakdown["raw"]["settled_trades"], 1)
        self.assertAlmostEqual(float(breakdown["raw"]["total_pnl"]), -0.4, places=4)
        self.assertEqual(breakdown["raw_selective_fallback"]["total_trades"], 1)
        self.assertEqual(breakdown["raw_selective_fallback"]["settled_trades"], 1)
        self.assertAlmostEqual(float(breakdown["raw_selective_fallback"]["total_pnl"]), 0.56, places=4)
        self.assertAlmostEqual(float(summary["total_pnl"]), 0.72, places=4)

    def test_refresh_and_settlement_surface_missing_required_truth_dates_after_fallback_failure(self):
        run_dir = Path("data") / "test_runs" / f"paper_refresh_missing_truth_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"
        summary_path = run_dir / "paper_trades" / "summary.json"
        actuals_dir = run_dir / "station_actuals"

        existing_path = _write_station_actual_rows(
            actuals_dir,
            "New York",
            [
                {
                    "date": "2026-03-31",
                    "tmax_f": 63.0,
                    "tmin_f": 48.0,
                }
            ],
        )

        log_paper_trades(
            [_base_opportunity(market_date="2026-04-01")],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        with patch("src.paper_trading.load_station_map", return_value={}), patch(
            "src.paper_trading.backfill_station_actuals",
            return_value=existing_path,
        ) as cdo_backfill_mock, patch(
            "src.paper_trading.backfill_station_actuals_from_cli_archive",
            return_value=existing_path,
        ) as cli_backfill_mock:
            refresh_result = refresh_station_actuals_for_open_trades(
                config={"ncei_api_token": "secret-token"},
                ledger_path=ledger_path,
                station_actuals_dir=actuals_dir,
                as_of_date="2026-04-02",
                today="2026-04-03",
            )

        self.assertTrue(refresh_result["attempted"])
        self.assertEqual(refresh_result["refresh_source"], "cdo")
        self.assertTrue(refresh_result["fallback_attempted"])
        self.assertEqual(refresh_result["fallback_refresh_source"], "cli_archive")
        self.assertEqual(
            refresh_result["fallback_city_windows"],
            [
                {
                    "city": "New York",
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-01",
                    "market_dates": ["2026-04-01"],
                    "open_trade_count": 1,
                }
            ],
        )
        self.assertEqual(
            [(call.args[0], call.args[1], call.args[2]) for call in cdo_backfill_mock.call_args_list],
            [("New York", "2026-04-01", "2026-04-02")],
        )
        self.assertEqual(
            [(call.args[0], call.kwargs["start"], call.kwargs["end"]) for call in cli_backfill_mock.call_args_list],
            [("New York", "2026-04-01", "2026-04-01")],
        )
        self.assertTrue(refresh_result["missing_required_truth_blocker"])
        self.assertEqual(refresh_result["missing_required_truth_trade_count"], 1)
        self.assertEqual(refresh_result["missing_required_truth_date_count"], 1)
        self.assertEqual(
            refresh_result["missing_required_truth_dates"],
            [
                {
                    "city": "New York",
                    "market_date": "2026-04-01",
                    "actual_field": "tmax_f",
                    "open_trade_count": 1,
                    "latest_local_date": "2026-03-31",
                }
            ],
        )
        self.assertEqual(
            refresh_result["missing_required_truth_city_windows"],
            [
                {
                    "city": "New York",
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-01",
                    "market_dates": ["2026-04-01"],
                    "open_trade_count": 1,
                    "latest_local_date": "2026-03-31",
                }
            ],
        )

        summary = settle_paper_trades(
            ledger_path=ledger_path,
            station_actuals_dir=actuals_dir,
            summary_path=summary_path,
            as_of_date="2026-04-02",
            config=_paper_fee_config(),
        )

        self.assertEqual(summary["newly_settled_trades"], 0)
        self.assertEqual(summary["open_trades"], 1)
        self.assertTrue(summary["missing_required_truth_blocker"])
        self.assertEqual(summary["missing_required_truth_trade_count"], 1)
        self.assertEqual(summary["missing_required_truth_date_count"], 1)
        self.assertEqual(
            summary["missing_required_truth_city_windows"],
            [
                {
                    "city": "New York",
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-01",
                    "market_dates": ["2026-04-01"],
                    "open_trade_count": 1,
                    "latest_local_date": "2026-03-31",
                }
            ],
        )

    def test_plan_station_actuals_refresh_limits_cities_and_cutoff(self):
        run_dir = Path("data") / "test_runs" / f"paper_refresh_plan_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"

        log_paper_trades(
            [
                _base_opportunity(city="New York", ticker="K1", market_date="2026-04-01"),
                _base_opportunity(city="New York", ticker="K2", market_date="2026-04-03"),
                _base_opportunity(city="Chicago", ticker="K3", market_date="2026-04-04"),
                _base_opportunity(city="Boston", ticker="K4", market_date="2026-04-05"),
            ],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )
        ledger = pd.read_csv(ledger_path)
        ledger.loc[ledger["ticker"] == "K3", "status"] = "settled"
        ledger.to_csv(ledger_path, index=False)

        plan = plan_station_actuals_refresh(
            ledger_path=ledger_path,
            as_of_date="2026-04-06",
            today="2026-04-05",
        )

        self.assertEqual(plan["cutoff_date"], "2026-04-04")
        self.assertEqual(plan["open_trade_count"], 3)
        self.assertEqual(plan["eligible_open_trade_count"], 2)
        self.assertEqual(
            plan["city_windows"],
            [
                {
                    "city": "New York",
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-04",
                    "market_dates": ["2026-04-01", "2026-04-03"],
                    "open_trade_count": 2,
                }
            ],
        )

    def test_refresh_station_actuals_for_open_trades_skips_without_eligible_open_trades(self):
        run_dir = Path("data") / "test_runs" / f"paper_refresh_skip_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"

        log_paper_trades(
            [_base_opportunity(city="Boston", ticker="K1", market_date="2026-04-05")],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        with patch("src.paper_trading.load_station_map") as load_station_map_mock, patch(
            "src.paper_trading.backfill_station_actuals_from_cli_archive"
        ) as cli_backfill_mock:
            result = refresh_station_actuals_for_open_trades(
                config={},
                ledger_path=ledger_path,
                as_of_date="2026-04-04",
                today="2026-04-05",
            )

        self.assertFalse(result["attempted"])
        self.assertFalse(result["needs_refresh"])
        self.assertEqual(result["cutoff_date"], "2026-04-04")
        self.assertEqual(result["eligible_open_trade_count"], 0)
        load_station_map_mock.assert_not_called()
        cli_backfill_mock.assert_not_called()

    def test_refresh_station_actuals_for_open_trades_falls_back_to_cli_for_still_missing_dates(self):
        run_dir = Path("data") / "test_runs" / f"paper_refresh_fallback_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"
        actuals_dir = run_dir / "station_actuals"

        _write_station_actual_rows(
            actuals_dir,
            "New York",
            [
                {
                    "date": "2026-03-31",
                    "tmax_f": 63.0,
                    "tmin_f": 48.0,
                }
            ],
        )

        log_paper_trades(
            [_base_opportunity(city="New York", ticker="K1", market_date="2026-04-01")],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        def cli_side_effect(city, *, start, end, base_dir, **kwargs):
            self.assertEqual((city, start, end), ("New York", "2026-04-01", "2026-04-01"))
            return _write_station_actual_rows(
                Path(base_dir),
                city,
                [
                    {
                        "date": "2026-03-31",
                        "tmax_f": 63.0,
                        "tmin_f": 48.0,
                    },
                    {
                        "date": "2026-04-01",
                        "tmax_f": 64.0,
                        "tmin_f": 49.0,
                    },
                ],
            )

        with patch("src.paper_trading.load_station_map", return_value={}), patch(
            "src.paper_trading.backfill_station_actuals",
            return_value=station_actuals_path("New York", base_dir=actuals_dir),
        ) as cdo_backfill_mock, patch(
            "src.paper_trading.backfill_station_actuals_from_cli_archive",
            side_effect=cli_side_effect,
        ) as cli_backfill_mock:
            result = refresh_station_actuals_for_open_trades(
                config={"ncei_api_token": "secret-token"},
                ledger_path=ledger_path,
                station_actuals_dir=actuals_dir,
                as_of_date="2026-04-02",
                today="2026-04-03",
            )

        self.assertTrue(result["attempted"])
        self.assertEqual(result["refresh_source"], "cdo")
        self.assertTrue(result["fallback_attempted"])
        self.assertEqual(result["fallback_refresh_source"], "cli_archive")
        self.assertEqual(
            result["fallback_city_windows"],
            [
                {
                    "city": "New York",
                    "start_date": "2026-04-01",
                    "end_date": "2026-04-01",
                    "market_dates": ["2026-04-01"],
                    "open_trade_count": 1,
                }
            ],
        )
        self.assertEqual(
            [(call.args[0], call.args[1], call.args[2]) for call in cdo_backfill_mock.call_args_list],
            [("New York", "2026-04-01", "2026-04-02")],
        )
        self.assertEqual(
            [(call.args[0], call.kwargs["start"], call.kwargs["end"]) for call in cli_backfill_mock.call_args_list],
            [("New York", "2026-04-01", "2026-04-01")],
        )
        self.assertFalse(result["missing_required_truth_blocker"])
        self.assertEqual(result["missing_required_truth_trade_count"], 0)
        self.assertEqual(result["missing_required_truth_date_count"], 0)
        self.assertEqual(result["missing_required_truth_dates"], [])
        self.assertEqual(result["missing_required_truth_city_windows"], [])

    def test_refresh_station_actuals_for_open_trades_skips_fallback_when_primary_source_satisfies_truth(self):
        run_dir = Path("data") / "test_runs" / f"paper_refresh_cdo_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = run_dir / "paper_trades" / "ledger.csv"
        actuals_dir = run_dir / "station_actuals"

        log_paper_trades(
            [
                _base_opportunity(city="New York", ticker="K1", market_date="2026-04-01"),
                _base_opportunity(city="Boston", ticker="K2", market_date="2026-04-03"),
                _base_opportunity(city="Boston", ticker="K3", market_date="2026-04-05"),
            ],
            scan_timestamp="2026-04-01T12:00:00+00:00",
            ledger_path=ledger_path,
            config=_paper_fee_config(),
        )

        with patch("src.paper_trading.load_station_map", return_value={}) as load_station_map_mock, patch(
            "src.paper_trading.backfill_station_actuals"
        ) as cdo_backfill_mock, patch(
            "src.paper_trading.backfill_station_actuals_from_cli_archive"
        ) as cli_backfill_mock:
            def cdo_side_effect(city, start, end, **kwargs):
                return _write_station_actual_rows(
                    Path(kwargs["base_dir"]),
                    city,
                    [
                        {
                            "date": start,
                            "tmax_f": 64.0,
                            "tmin_f": 49.0,
                        }
                    ],
                )

            cdo_backfill_mock.side_effect = cdo_side_effect
            result = refresh_station_actuals_for_open_trades(
                config={"ncei_api_token": "secret-token"},
                ledger_path=ledger_path,
                station_actuals_dir=actuals_dir,
                as_of_date="2026-04-06",
                today="2026-04-05",
            )

        self.assertTrue(result["attempted"])
        self.assertEqual(result["refresh_source"], "cdo")
        self.assertFalse(result["fallback_attempted"])
        self.assertEqual(result["fallback_refresh_source"], "cli_archive")
        self.assertEqual(result["fallback_city_windows"], [])
        self.assertFalse(result["missing_required_truth_blocker"])
        self.assertEqual(
            [(call.args[0], call.args[1], call.args[2]) for call in cdo_backfill_mock.call_args_list],
            [
                ("Boston", "2026-04-03", "2026-04-04"),
                ("New York", "2026-04-01", "2026-04-04"),
            ],
        )
        load_station_map_mock.assert_called_once()
        cli_backfill_mock.assert_not_called()


def test_log_paper_trades_records_market_category(tmp_path):
    from src.paper_trading import log_paper_trades
    opps = [{
        "source": "kalshi", "ticker": "KXRAINNYC-26APR21-T0",
        "market_category": "rain", "city": "New York", "market_type": "rain_binary",
        "market_date": "2026-04-21", "outcome": "Yes", "position_side": "yes",
        "direction": "BUY",
        "our_probability": 0.8, "market_price": 0.55, "edge": 0.25,
        "abs_edge": 0.25, "forecast_blend_source": "open-meteo",
        "forecast_calibration_source": "logistic",
        "probability_calibration_source": "isotonic",
        "hours_to_settlement": 10, "raw_probability": 0.7,
        "volume24hr": 1500, "yes_outcome": True,
    }]
    ledger = tmp_path / "ledger.csv"
    log_paper_trades(opps, scan_timestamp="2026-04-20T12:00:00+00:00",
                     ledger_path=ledger, contracts=1)
    import pandas as pd
    df = pd.read_csv(ledger)
    assert "market_category" in df.columns
    assert df.iloc[0]["market_category"] == "rain"


def test_legacy_ledger_migrates_market_category_to_temperature(tmp_path):
    import pandas as pd
    from src.paper_trading import _ensure_ledger_schema

    legacy = tmp_path / "legacy.csv"
    pd.DataFrame({
        "trade_id": ["t1"], "scan_id": ["s1"], "source": ["kalshi"],
        "ticker": ["KXHIGHT"], "city": ["Boston"], "market_type": ["high"],
        "market_date": ["2026-04-20"], "status": ["open"],
    }).to_csv(legacy, index=False)

    df = _ensure_ledger_schema(pd.read_csv(legacy))
    assert "market_category" in df.columns
    assert df.iloc[0]["market_category"] == "temperature"


def test_settle_paper_trades_reports_category_breakdown(tmp_path):
    """Summary must report per-category PnL/ROI/trade counts and a
    daily-PnL Pearson correlation when both categories have settled rows."""
    import pandas as pd
    from src.paper_trading import settle_paper_trades

    ledger = tmp_path / "ledger.csv"
    rows = []
    # Temperature: 5 settled trades across 5 distinct days
    for i in range(5):
        rows.append({
            "trade_id": f"t{i}", "scan_id": "s", "status": "settled",
            "source": "kalshi", "ticker": f"KXHIGHT-{i}", "market_category": "temperature",
            "city": "New York", "market_type": "high",
            "market_date": f"2026-04-{10+i:02d}", "outcome": "Yes", "position_side": "yes",
            "direction": "BUY",
            "contracts": 1, "entry_price": 0.5, "entry_cost": 0.5,
            "payout": 1.0 if i % 2 == 0 else 0.0,
            "pnl": 0.5 if i % 2 == 0 else -0.5,
            "roi": 1.0 if i % 2 == 0 else -1.0,
            "total_fees": 0, "entry_fee": 0, "settlement_fee": 0,
            "yes_outcome": i % 2 == 0,
            "settled_at_utc": f"2026-04-{10+i:02d}T23:00:00+00:00",
        })
    # Rain: 5 settled trades on the same 5 days
    for i in range(5):
        rows.append({
            "trade_id": f"r{i}", "scan_id": "s", "status": "settled",
            "source": "kalshi", "ticker": f"KXRAIN-{i}", "market_category": "rain",
            "city": "New York", "market_type": "rain_binary",
            "market_date": f"2026-04-{10+i:02d}", "outcome": "Yes", "position_side": "yes",
            "direction": "BUY",
            "contracts": 1, "entry_price": 0.5, "entry_cost": 0.5,
            "payout": 1.0 if i == 0 else 0.0,
            "pnl": 0.5 if i == 0 else -0.5,
            "roi": 1.0 if i == 0 else -1.0,
            "total_fees": 0, "entry_fee": 0, "settlement_fee": 0,
            "yes_outcome": i == 0,
            "settled_at_utc": f"2026-04-{10+i:02d}T23:00:00+00:00",
        })
    pd.DataFrame(rows).to_csv(ledger, index=False)

    summary_path = tmp_path / "summary.json"
    summary = settle_paper_trades(ledger_path=ledger, summary_path=summary_path)

    cb = summary["category_breakdown"]
    assert cb["temperature"]["trade_count"] == 5
    assert cb["rain"]["trade_count"] == 5
    # Correlation is a float in [-1, 1] when both series have variance
    assert -1.0 <= cb["correlation_30d"] <= 1.0
    assert cb["correlation_sample_size"] == 5


if __name__ == "__main__":
    unittest.main()
