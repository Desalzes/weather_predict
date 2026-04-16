import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from src.deepseek_worker import (
    approved_opportunities_from_review,
    format_review_summary,
    maybe_run_deepseek_review,
)


def _base_config(run_dir: Path) -> dict:
    return {
        "enable_deepseek_worker": True,
        "deepseek_trade_mode": "paper_gate",
        "deepseek_api_base": "https://api.deepseek.com",
        "deepseek_model": "deepseek-chat",
        "deepseek_review_interval_scans": 3,
        "deepseek_max_opportunities_per_review": 4,
        "deepseek_min_abs_edge": 0.08,
        "deepseek_timeout_seconds": 10,
        "deepseek_max_response_tokens": 600,
        "deepseek_pull_kalshi_market_context": False,
        "deepseek_state_path": str(run_dir / "state.json"),
        "deepseek_reviews_dir": str(run_dir / "reviews"),
    }


def _opportunity(ticker: str, edge: float, source: str = "kalshi") -> dict:
    return {
        "source": source,
        "ticker": ticker,
        "market_question": f"Question for {ticker}",
        "city": "New York",
        "market_type": "high",
        "market_date": "2026-04-04",
        "outcome": "<65F",
        "direction": "BUY" if edge >= 0 else "SELL",
        "edge": edge,
        "abs_edge": abs(edge),
        "our_probability": 0.62,
        "market_price": 0.41,
        "forecast_value_f": 64.8,
        "open_meteo_forecast_value_f": 64.4,
        "raw_forecast_value_f": 64.4,
        "uncertainty_std_f": 2.1,
        "hours_to_settlement": 6.0,
        "forecast_calibration_source": "emos",
        "probability_calibration_source": "isotonic",
        "forecast_blend_source": "open-meteo+hrrr",
        "volume24hr": 1200,
        "yes_bid": 0.4,
        "yes_ask": 0.42,
        "settlement_rule": "lte",
        "settlement_low_f": None,
        "settlement_high_f": 65.0,
    }


class DeepSeekWorkerTests(unittest.TestCase):
    def test_first_eligible_scan_runs_review_and_persists_artifacts(self):
        run_dir = Path("data") / "test_runs" / f"deepseek_review_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)
        opportunities = [
            _opportunity("KX1", 0.18),
            _opportunity("KX2", 0.11),
            _opportunity("KX3", 0.04),
        ]

        response_payload = {
            "id": "ds-test-1",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "Approve the highest-quality setup only.",
                                "decisions": [
                                    {
                                        "opportunity_id": "kalshi|KX1|2026-04-04|<65F|BUY",
                                        "decision": "approve",
                                        "confidence": 0.83,
                                        "max_contracts": 2,
                                        "reason": "Strong edge with acceptable liquidity.",
                                    },
                                    {
                                        "opportunity_id": "kalshi|KX2|2026-04-04|<65F|BUY",
                                        "decision": "watch",
                                        "confidence": 0.51,
                                        "max_contracts": 0,
                                        "reason": "Edge is real but weaker.",
                                    },
                                ],
                            }
                        )
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 80,
                "total_tokens": 180,
            },
        }
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = response_payload

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            return_value=mock_response,
        ) as post_mock:
            review = maybe_run_deepseek_review(
                opportunities,
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
                strategy_policy={
                    "policy_version": 1,
                    "objective": "Only take the best paper trades.",
                    "selection": {"min_abs_edge": 0.08},
                    "execution": {"max_contracts_per_trade": 1},
                    "deepseek": {"instruction": "Stay selective."},
                },
            )

        self.assertEqual(review["status"], "completed")
        self.assertEqual(review["approved_count"], 1)
        self.assertEqual(review["selected_opportunity_count"], 2)
        self.assertEqual(len(approved_opportunities_from_review(opportunities, review)), 1)
        self.assertIn("reviewed 2 opportunities, approved 1", format_review_summary(review))
        self.assertTrue(Path(review["review_path"]).exists())
        state = json.loads(Path(config["deepseek_state_path"]).read_text(encoding="utf-8"))
        self.assertEqual(state["scan_count"], 1)
        self.assertEqual(state["review_runs"], 1)
        self.assertEqual(state["last_review_status"], "completed")
        self.assertEqual(state["last_review_scan_count"], 1)
        self.assertEqual(state["last_review_scan_id"], "2026-04-04T12:00:00+00:00")
        self.assertEqual(state["last_review_path"], review["review_path"])
        request_payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(request_payload["model"], "deepseek-chat")
        self.assertEqual(request_payload["response_format"], {"type": "json_object"})
        self.assertEqual(request_payload["thinking"], {"type": "disabled"})
        self.assertIn("strategy_policy", json.loads(request_payload["messages"][1]["content"]))

    def test_review_interval_skips_until_due_after_completed_run(self):
        run_dir = Path("data") / "test_runs" / f"deepseek_interval_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)
        opportunities = [_opportunity("KX1", 0.18)]

        response_payload = {
            "id": "ds-test-interval",
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "Approve it.",
                                "decisions": [
                                    {
                                        "opportunity_id": "kalshi|KX1|2026-04-04|<65F|BUY",
                                        "decision": "approve",
                                        "confidence": 0.8,
                                        "max_contracts": 1,
                                        "reason": "Best available trade.",
                                    }
                                ],
                            }
                        )
                    }
                }
            ],
            "usage": {},
        }
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = response_payload

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            return_value=mock_response,
        ) as post_mock:
            first = maybe_run_deepseek_review(opportunities, config=config, scan_timestamp="2026-04-04T12:00:00+00:00")
            second = maybe_run_deepseek_review(opportunities, config=config, scan_timestamp="2026-04-04T12:30:00+00:00")
            third = maybe_run_deepseek_review(opportunities, config=config, scan_timestamp="2026-04-04T13:00:00+00:00")
            fourth = maybe_run_deepseek_review(opportunities, config=config, scan_timestamp="2026-04-04T13:30:00+00:00")

        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "skipped")
        self.assertEqual(second["reason"], "interval_not_reached")
        self.assertEqual(third["status"], "skipped")
        self.assertEqual(third["reason"], "interval_not_reached")
        self.assertEqual(fourth["status"], "completed")
        self.assertEqual(post_mock.call_count, 2)
        state = json.loads(Path(config["deepseek_state_path"]).read_text(encoding="utf-8"))
        self.assertEqual(state["scan_count"], 4)
        self.assertEqual(state["review_runs"], 2)
        self.assertEqual(state["last_review_scan_count"], 4)

    def test_missing_api_key_skips_without_request(self):
        run_dir = Path("data") / "test_runs" / f"deepseek_no_key_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        with patch.dict("os.environ", {}, clear=True), patch("src.deepseek_worker.requests.post") as post_mock:
            review = maybe_run_deepseek_review(
                [_opportunity("KX1", 0.18)],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        self.assertEqual(review["status"], "skipped")
        self.assertEqual(review["reason"], "missing_api_key")
        post_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
