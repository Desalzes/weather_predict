"""Tests for DeepSeek worker failure modes — timeout, malformed JSON, HTTP errors, rate limiting."""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import requests as req

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
        "deepseek_review_interval_scans": 1,
        "deepseek_max_opportunities_per_review": 4,
        "deepseek_min_abs_edge": 0.08,
        "deepseek_timeout_seconds": 5,
        "deepseek_max_response_tokens": 600,
        "deepseek_pull_kalshi_market_context": False,
        "deepseek_state_path": str(run_dir / "state.json"),
        "deepseek_reviews_dir": str(run_dir / "reviews"),
    }


def _opportunity(ticker: str = "KX1", edge: float = 0.18) -> dict:
    return {
        "source": "kalshi",
        "ticker": ticker,
        "market_question": f"Question for {ticker}",
        "city": "New York",
        "market_type": "high",
        "market_date": "2026-04-04",
        "outcome": "<65F",
        "direction": "BUY",
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


class TimeoutTests(unittest.TestCase):
    """DeepSeek API timeout should result in failed status, not crash."""

    def test_timeout_returns_failed_status(self):
        run_dir = Path("data") / "test_runs" / f"ds_timeout_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            side_effect=req.Timeout("Connection timed out after 5s"),
        ):
            review = maybe_run_deepseek_review(
                [_opportunity()],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        self.assertEqual(review["status"], "failed")
        self.assertEqual(review["reason"], "request_failed")
        self.assertIn("timed out", review["error"].lower())

    def test_timeout_persists_failed_state(self):
        run_dir = Path("data") / "test_runs" / f"ds_timeout_state_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            side_effect=req.Timeout("timeout"),
        ):
            maybe_run_deepseek_review(
                [_opportunity()],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        state = json.loads(Path(config["deepseek_state_path"]).read_text(encoding="utf-8"))
        self.assertEqual(state["last_review_status"], "failed")
        self.assertEqual(state["scan_count"], 1)
        # review_runs should NOT increment on failure
        self.assertEqual(state["review_runs"], 0)

    def test_approved_opportunities_empty_on_failed_review(self):
        failed_review = {"status": "failed", "reason": "request_failed"}
        result = approved_opportunities_from_review([_opportunity()], failed_review)
        self.assertEqual(result, [])


class MalformedResponseTests(unittest.TestCase):
    """DeepSeek returns non-JSON or malformed JSON in choices."""

    def test_invalid_json_content_returns_failed(self):
        run_dir = Path("data") / "test_runs" / f"ds_malformed_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        response_payload = {
            "id": "ds-malformed",
            "choices": [{"message": {"content": "This is not valid JSON at all {{"}}],
            "usage": {},
        }
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = response_payload

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            return_value=mock_response,
        ):
            review = maybe_run_deepseek_review(
                [_opportunity()],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        self.assertEqual(review["status"], "failed")
        self.assertEqual(review["reason"], "request_failed")

    def test_empty_choices_returns_failed(self):
        """Empty choices list causes IndexError on [0] access -> caught as failed."""
        run_dir = Path("data") / "test_runs" / f"ds_empty_choices_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        response_payload = {"id": "ds-empty", "choices": [], "usage": {}}
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = response_payload

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            return_value=mock_response,
        ):
            review = maybe_run_deepseek_review(
                [_opportunity()],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        # Empty choices -> IndexError on choices[0] -> caught by except -> "failed"
        self.assertEqual(review["status"], "failed")
        self.assertEqual(review["reason"], "request_failed")

    def test_missing_decisions_defaults_to_watch(self):
        """If DeepSeek returns JSON but no decisions array, all should default to watch."""
        run_dir = Path("data") / "test_runs" / f"ds_no_decisions_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        response_payload = {
            "id": "ds-no-decisions",
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"summary": "I have no decisions for you."})
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
        ):
            review = maybe_run_deepseek_review(
                [_opportunity()],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        self.assertEqual(review["status"], "completed")
        self.assertEqual(review["approved_count"], 0)
        self.assertEqual(review["watch_count"], 1)
        # Decision should be the default watch
        self.assertEqual(review["decisions"][0]["decision"], "watch")
        self.assertIn("No explicit decision", review["decisions"][0]["reason"])


class HttpErrorTests(unittest.TestCase):
    """HTTP 4xx/5xx error responses."""

    def test_429_rate_limit_returns_failed(self):
        run_dir = Path("data") / "test_runs" / f"ds_429_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = req.HTTPError(
            response=Mock(status_code=429, text="Rate limit exceeded")
        )

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            return_value=mock_response,
        ):
            review = maybe_run_deepseek_review(
                [_opportunity()],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        self.assertEqual(review["status"], "failed")
        self.assertEqual(review["reason"], "request_failed")

    def test_500_server_error_returns_failed(self):
        run_dir = Path("data") / "test_runs" / f"ds_500_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = req.HTTPError(
            response=Mock(status_code=500, text="Internal Server Error")
        )

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            return_value=mock_response,
        ):
            review = maybe_run_deepseek_review(
                [_opportunity()],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        self.assertEqual(review["status"], "failed")

    def test_connection_error_returns_failed(self):
        run_dir = Path("data") / "test_runs" / f"ds_conn_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            side_effect=req.ConnectionError("DNS resolution failed"),
        ):
            review = maybe_run_deepseek_review(
                [_opportunity()],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        self.assertEqual(review["status"], "failed")
        self.assertEqual(review["reason"], "request_failed")


class DisabledAndSkipTests(unittest.TestCase):
    """Edge cases: disabled worker, no eligible opportunities."""

    def test_disabled_worker_returns_disabled(self):
        run_dir = Path("data") / "test_runs" / f"ds_disabled_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)
        config["enable_deepseek_worker"] = False

        review = maybe_run_deepseek_review(
            [_opportunity()],
            config=config,
            scan_timestamp="2026-04-04T12:00:00+00:00",
        )
        self.assertEqual(review["status"], "disabled")
        self.assertEqual(review["reason"], "worker_disabled")

    def test_no_eligible_opportunities_skips(self):
        """All opportunities below min_abs_edge should result in skip."""
        run_dir = Path("data") / "test_runs" / f"ds_no_eligible_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)
        config["deepseek_min_abs_edge"] = 0.50  # very high bar

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            review = maybe_run_deepseek_review(
                [_opportunity(edge=0.10)],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        self.assertEqual(review["status"], "skipped")
        self.assertEqual(review["reason"], "no_eligible_opportunities")

    def test_empty_opportunities_list_skips(self):
        run_dir = Path("data") / "test_runs" / f"ds_empty_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            review = maybe_run_deepseek_review(
                [],
                config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )

        self.assertEqual(review["status"], "skipped")
        self.assertEqual(review["reason"], "no_eligible_opportunities")


class FormatReviewSummaryTests(unittest.TestCase):
    """Test format_review_summary edge cases."""

    def test_none_review(self):
        self.assertEqual(format_review_summary(None), "DeepSeek worker: not run.")

    def test_failed_review(self):
        review = {"status": "failed", "reason": "request_failed"}
        result = format_review_summary(review)
        self.assertIn("failed", result)
        self.assertIn("request_failed", result)

    def test_skipped_review(self):
        review = {"status": "skipped", "reason": "missing_api_key"}
        result = format_review_summary(review)
        self.assertIn("skipped", result)
        self.assertIn("missing_api_key", result)

    def test_completed_review(self):
        review = {
            "status": "completed",
            "selected_opportunity_count": 3,
            "approved_count": 1,
            "rejected_count": 1,
            "watch_count": 1,
        }
        result = format_review_summary(review)
        self.assertIn("reviewed 3 opportunities", result)
        self.assertIn("approved 1", result)


class RecoveryAfterFailureTests(unittest.TestCase):
    """Verify the worker can recover after a failure."""

    def test_successful_review_after_failure(self):
        run_dir = Path("data") / "test_runs" / f"ds_recovery_{uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _base_config(run_dir)

        # First call: fails
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            side_effect=req.ConnectionError("network down"),
        ):
            r1 = maybe_run_deepseek_review(
                [_opportunity()], config=config,
                scan_timestamp="2026-04-04T12:00:00+00:00",
            )
        self.assertEqual(r1["status"], "failed")

        # Second call: succeeds
        success_payload = {
            "id": "ds-recovery",
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "summary": "Recovered.",
                            "decisions": [
                                {
                                    "opportunity_id": "kalshi|KX1|2026-04-04|<65F|BUY",
                                    "decision": "approve",
                                    "confidence": 0.9,
                                    "max_contracts": 1,
                                    "reason": "Strong setup.",
                                }
                            ],
                        })
                    }
                }
            ],
            "usage": {},
        }
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = success_payload

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "src.deepseek_worker.requests.post",
            return_value=mock_response,
        ):
            r2 = maybe_run_deepseek_review(
                [_opportunity()], config=config,
                scan_timestamp="2026-04-04T12:30:00+00:00",
            )

        self.assertEqual(r2["status"], "completed")
        self.assertEqual(r2["approved_count"], 1)

        state = json.loads(Path(config["deepseek_state_path"]).read_text(encoding="utf-8"))
        self.assertEqual(state["scan_count"], 2)
        self.assertEqual(state["review_runs"], 1)
        self.assertEqual(state["last_review_status"], "completed")


if __name__ == "__main__":
    unittest.main()
