from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import PROJECT_ROOT
from src.strategy_policy import compact_policy_for_prompt

logger = logging.getLogger("weather.deepseek_worker")

DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_STATE_PATH = PROJECT_ROOT / "data" / "deepseek_worker" / "state.json"
DEFAULT_REVIEWS_DIR = PROJECT_ROOT / "data" / "deepseek_worker" / "reviews"


def _normalize_iso_timestamp(value: str | datetime | None = None) -> str:
    if isinstance(value, datetime):
        dt = value
    elif value:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    else:
        dt = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _resolve_path(path_value: str | os.PathLike | None, default_path: Path) -> Path:
    path = Path(path_value) if path_value is not None else default_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_dir(path_value: str | os.PathLike | None, default_path: Path) -> Path:
    path = Path(path_value) if path_value is not None else default_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def deepseek_enabled(config: dict | None = None) -> bool:
    return bool((config or {}).get("enable_deepseek_worker", False))


def deepseek_trade_mode(config: dict | None = None) -> str:
    mode = str((config or {}).get("deepseek_trade_mode", "review_only") or "review_only").strip().lower()
    return mode if mode in {"review_only", "paper_gate"} else "review_only"


def deepseek_api_key(config: dict | None = None) -> str:
    config = config or {}
    return os.getenv("DEEPSEEK_API_KEY", "").strip() or str(config.get("deepseek_api_key", "") or "").strip()


def _state_path(config: dict | None = None) -> Path:
    return _resolve_path((config or {}).get("deepseek_state_path"), DEFAULT_STATE_PATH)


def _reviews_dir(config: dict | None = None) -> Path:
    return _resolve_dir((config or {}).get("deepseek_reviews_dir"), DEFAULT_REVIEWS_DIR)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "scan_count": 0,
            "review_runs": 0,
            "last_review_scan_count": None,
            "last_review_scan_id": None,
            "last_review_status": None,
            "last_review_path": None,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("DeepSeek worker state was unreadable at %s; resetting.", path)
        return {
            "scan_count": 0,
            "review_runs": 0,
            "last_review_scan_count": None,
            "last_review_scan_id": None,
            "last_review_status": None,
            "last_review_path": None,
        }


def _write_state(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _opportunity_id(opportunity: dict) -> str:
    return "|".join(
        [
            str(opportunity.get("source", "")).strip(),
            str(opportunity.get("ticker", "")).strip(),
            str(opportunity.get("market_date", "")).strip(),
            str(opportunity.get("outcome", "")).strip(),
            str(opportunity.get("direction", "")).strip(),
        ]
    )


def _opportunity_payload(opportunity: dict) -> dict:
    return {
        "opportunity_id": _opportunity_id(opportunity),
        "source": opportunity.get("source"),
        "ticker": opportunity.get("ticker"),
        "market_question": opportunity.get("market_question"),
        "city": opportunity.get("city"),
        "market_type": opportunity.get("market_type"),
        "market_date": opportunity.get("market_date"),
        "outcome": opportunity.get("outcome"),
        "direction": opportunity.get("direction"),
        "edge": opportunity.get("edge"),
        "abs_edge": opportunity.get("abs_edge"),
        "our_probability": opportunity.get("our_probability"),
        "market_price": opportunity.get("market_price"),
        "forecast_value_f": opportunity.get("forecast_value_f"),
        "open_meteo_forecast_value_f": opportunity.get("open_meteo_forecast_value_f"),
        "raw_forecast_value_f": opportunity.get("raw_forecast_value_f"),
        "uncertainty_std_f": opportunity.get("uncertainty_std_f"),
        "hours_to_settlement": opportunity.get("hours_to_settlement"),
        "forecast_calibration_source": opportunity.get("forecast_calibration_source"),
        "probability_calibration_source": opportunity.get("probability_calibration_source"),
        "forecast_blend_source": opportunity.get("forecast_blend_source"),
        "volume24hr": opportunity.get("volume24hr"),
        "yes_bid": opportunity.get("yes_bid"),
        "yes_ask": opportunity.get("yes_ask"),
        "settlement_rule": opportunity.get("settlement_rule"),
        "settlement_low_f": opportunity.get("settlement_low_f"),
        "settlement_high_f": opportunity.get("settlement_high_f"),
    }


def _select_review_candidates(opportunities: list[dict], config: dict) -> list[dict]:
    max_items = max(int(config.get("deepseek_max_opportunities_per_review", 5) or 5), 1)
    min_abs_edge = float(config.get("deepseek_min_abs_edge", config.get("min_edge_threshold", 0.05)) or 0.05)
    eligible = [
        opportunity
        for opportunity in opportunities
        if float(opportunity.get("abs_edge", 0.0) or 0.0) >= min_abs_edge
    ]
    eligible.sort(key=lambda item: float(item.get("abs_edge", 0.0) or 0.0), reverse=True)
    return eligible[:max_items]


def _load_kalshi_market_context(opportunities: list[dict], config: dict) -> dict:
    if not config.get("deepseek_pull_kalshi_market_context", True):
        return {}

    kalshi_opportunities = [item for item in opportunities if str(item.get("source", "")).lower() == "kalshi"]
    if not kalshi_opportunities:
        return {}

    from src.kalshi_client import KalshiClient

    client = KalshiClient()
    context = {}
    for opportunity in kalshi_opportunities:
        ticker = str(opportunity.get("ticker", "")).strip()
        if not ticker:
            continue
        try:
            market = client.get_market(ticker)
        except Exception as exc:
            logger.warning("DeepSeek market-context pull failed for %s: %s", ticker, exc)
            continue

        context[_opportunity_id(opportunity)] = {
            "status": market.get("status"),
            "result": market.get("result"),
            "close_time": market.get("close_time"),
            "last_price": market.get("last_price"),
            "yes_bid": market.get("yes_bid"),
            "yes_ask": market.get("yes_ask"),
            "volume": market.get("volume"),
            "volume_24h": market.get("volume_24h"),
            "open_interest": market.get("open_interest"),
            "liquidity": market.get("liquidity"),
        }
    return context


def _build_messages(
    selected_opportunities: list[dict],
    market_context: dict,
    config: dict,
    strategy_policy: dict | None = None,
) -> list[dict]:
    trade_mode = deepseek_trade_mode(config)
    payload = {
        "objective": "Maximize risk-adjusted profit in weather prediction markets.",
        "execution_context": {
            "trade_mode": trade_mode,
            "paper_only": trade_mode == "paper_gate",
            "scan_style": "intermittent_review_worker",
        },
        "instructions": [
            "Return only valid JSON.",
            "Use only the data provided here.",
            "Approve only opportunities you would actually trade now.",
            "Reject opportunities with weak edge quality, poor liquidity, or fragile calibration.",
            "Use watch when the setup is interesting but not strong enough for immediate action.",
        ],
        "response_schema": {
            "summary": "string",
            "decisions": [
                {
                    "opportunity_id": "string",
                    "decision": "approve|reject|watch",
                    "confidence": "0_to_1_float",
                    "max_contracts": "positive_integer",
                    "reason": "short_string",
                }
            ],
        },
        "opportunities": [_opportunity_payload(item) for item in selected_opportunities],
        "kalshi_market_context": market_context,
    }
    if strategy_policy:
        payload["strategy_policy"] = compact_policy_for_prompt(strategy_policy)
    return [
        {
            "role": "system",
            "content": (
                "You are a weather-market trading reviewer. Produce JSON only. "
                "Be conservative with thin or noisy setups. Approve only the best opportunities."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, indent=2),
        },
    ]


def _artifact_name(scan_timestamp: str) -> str:
    safe = scan_timestamp.replace(":", "").replace("-", "").replace("+", "Z")
    return f"deepseek_review_{safe}.json"


def _normalize_decisions(parsed: dict, selected_opportunities: list[dict]) -> list[dict]:
    decision_by_id = {}
    for item in parsed.get("decisions", []):
        opportunity_id = str(item.get("opportunity_id", "")).strip()
        if not opportunity_id:
            continue
        decision = str(item.get("decision", "watch") or "watch").strip().lower()
        if decision not in {"approve", "reject", "watch"}:
            decision = "watch"
        confidence = item.get("confidence")
        try:
            confidence_value = max(0.0, min(float(confidence), 1.0))
        except (TypeError, ValueError):
            confidence_value = 0.0
        max_contracts = item.get("max_contracts", 1)
        try:
            max_contracts_value = max(int(max_contracts), 0)
        except (TypeError, ValueError):
            max_contracts_value = 0
        decision_by_id[opportunity_id] = {
            "opportunity_id": opportunity_id,
            "decision": decision,
            "confidence": round(confidence_value, 4),
            "max_contracts": max_contracts_value,
            "reason": str(item.get("reason", "") or "").strip(),
        }

    normalized = []
    for opportunity in selected_opportunities:
        opportunity_id = _opportunity_id(opportunity)
        normalized.append(
            decision_by_id.get(
                opportunity_id,
                {
                    "opportunity_id": opportunity_id,
                    "decision": "watch",
                    "confidence": 0.0,
                    "max_contracts": 0,
                    "reason": "No explicit decision returned by DeepSeek.",
                },
            )
        )
    return normalized


def _review_payload(
    *,
    scan_timestamp: str,
    config: dict,
    selected_opportunities: list[dict],
    market_context: dict,
    response_json: dict,
    parsed_response: dict,
    normalized_decisions: list[dict],
    state: dict,
    strategy_policy: dict | None = None,
) -> dict:
    approved = [item for item in normalized_decisions if item["decision"] == "approve"]
    rejected = [item for item in normalized_decisions if item["decision"] == "reject"]
    watched = [item for item in normalized_decisions if item["decision"] == "watch"]
    return {
        "status": "completed",
        "scan_timestamp": scan_timestamp,
        "reviewed_at_utc": _normalize_iso_timestamp(),
        "trade_mode": deepseek_trade_mode(config),
        "model": str(config.get("deepseek_model", DEFAULT_MODEL) or DEFAULT_MODEL),
        "api_base": str(config.get("deepseek_api_base", DEFAULT_API_BASE) or DEFAULT_API_BASE),
        "scan_count": state.get("scan_count"),
        "review_runs": state.get("review_runs"),
        "summary": str(parsed_response.get("summary", "") or "").strip(),
        "selected_opportunity_count": len(selected_opportunities),
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "watch_count": len(watched),
        "selected_opportunities": [_opportunity_payload(item) for item in selected_opportunities],
        "kalshi_market_context": market_context,
        "strategy_policy": compact_policy_for_prompt(strategy_policy) if strategy_policy else None,
        "decisions": normalized_decisions,
        "usage": response_json.get("usage", {}),
        "response_id": response_json.get("id"),
    }


def maybe_run_deepseek_review(
    opportunities: list[dict],
    *,
    config: dict | None = None,
    scan_timestamp: str | datetime | None = None,
    strategy_policy: dict | None = None,
) -> dict:
    config = config or {}
    normalized_scan_timestamp = _normalize_iso_timestamp(scan_timestamp)
    state_path = _state_path(config)
    reviews_dir = _reviews_dir(config)
    state = _load_state(state_path)
    state["scan_count"] = int(state.get("scan_count", 0) or 0) + 1

    if not deepseek_enabled(config):
        state["last_review_status"] = "disabled"
        _write_state(state_path, state)
        return {
            "status": "disabled",
            "reason": "worker_disabled",
            "scan_timestamp": normalized_scan_timestamp,
            "scan_count": state["scan_count"],
            "trade_mode": deepseek_trade_mode(config),
        }

    api_key = deepseek_api_key(config)
    if not api_key:
        state["last_review_status"] = "skipped"
        _write_state(state_path, state)
        return {
            "status": "skipped",
            "reason": "missing_api_key",
            "scan_timestamp": normalized_scan_timestamp,
            "scan_count": state["scan_count"],
            "trade_mode": deepseek_trade_mode(config),
        }

    selected_opportunities = _select_review_candidates(opportunities, config)
    if not selected_opportunities:
        state["last_review_status"] = "skipped"
        _write_state(state_path, state)
        return {
            "status": "skipped",
            "reason": "no_eligible_opportunities",
            "scan_timestamp": normalized_scan_timestamp,
            "scan_count": state["scan_count"],
            "trade_mode": deepseek_trade_mode(config),
        }

    review_interval = max(int(config.get("deepseek_review_interval_scans", 3) or 3), 1)
    last_review_scan_count = state.get("last_review_scan_count")
    has_completed_review = int(state.get("review_runs", 0) or 0) > 0
    scans_since_review = None
    if has_completed_review and last_review_scan_count is not None:
        scans_since_review = state["scan_count"] - int(last_review_scan_count)
    should_run_review = not has_completed_review or scans_since_review is None or scans_since_review >= review_interval

    if not should_run_review:
        state["last_review_status"] = "skipped"
        _write_state(state_path, state)
        return {
            "status": "skipped",
            "reason": "interval_not_reached",
            "scan_timestamp": normalized_scan_timestamp,
            "scan_count": state["scan_count"],
            "trade_mode": deepseek_trade_mode(config),
            "selected_opportunity_count": len(selected_opportunities),
            "scans_until_review": max(review_interval - int(scans_since_review or 0), 0),
        }

    market_context = _load_kalshi_market_context(selected_opportunities, config)
    request_payload = {
        "model": str(config.get("deepseek_model", DEFAULT_MODEL) or DEFAULT_MODEL),
        "messages": _build_messages(selected_opportunities, market_context, config, strategy_policy),
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0.2,
        "max_tokens": max(int(config.get("deepseek_max_response_tokens", 1200) or 1200), 256),
    }

    api_base = str(config.get("deepseek_api_base", DEFAULT_API_BASE) or DEFAULT_API_BASE).rstrip("/")
    endpoint = f"{api_base}/chat/completions"
    timeout_seconds = max(float(config.get("deepseek_timeout_seconds", 45) or 45), 5.0)

    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        response_json = response.json()
        content = (
            response_json.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        ) or "{}"
        parsed_response = json.loads(content)
        normalized_decisions = _normalize_decisions(parsed_response, selected_opportunities)
    except Exception as exc:
        state["last_review_status"] = "failed"
        _write_state(state_path, state)
        logger.warning("DeepSeek review failed: %s", exc)
        return {
            "status": "failed",
            "reason": "request_failed",
            "error": str(exc),
            "scan_timestamp": normalized_scan_timestamp,
            "scan_count": state["scan_count"],
            "trade_mode": deepseek_trade_mode(config),
            "selected_opportunity_count": len(selected_opportunities),
        }

    state["review_runs"] = int(state.get("review_runs", 0) or 0) + 1
    state["last_review_scan_count"] = state["scan_count"]
    state["last_review_scan_id"] = normalized_scan_timestamp
    state["last_review_status"] = "completed"

    review = _review_payload(
        scan_timestamp=normalized_scan_timestamp,
        config=config,
        selected_opportunities=selected_opportunities,
        market_context=market_context,
        response_json=response_json,
        parsed_response=parsed_response,
        normalized_decisions=normalized_decisions,
        state=state,
        strategy_policy=strategy_policy,
    )
    review_path = reviews_dir / _artifact_name(normalized_scan_timestamp)
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    state["last_review_path"] = str(review_path)
    _write_state(state_path, state)

    review["review_path"] = str(review_path)
    return review


def approved_opportunities_from_review(opportunities: list[dict], review: dict | None) -> list[dict]:
    if not review or str(review.get("status", "")) != "completed":
        return []

    approved_ids = {
        str(item.get("opportunity_id", "")).strip()
        for item in review.get("decisions", [])
        if str(item.get("decision", "")).strip().lower() == "approve"
    }
    return [
        opportunity
        for opportunity in opportunities
        if _opportunity_id(opportunity) in approved_ids
    ]


def format_review_summary(review: dict | None) -> str:
    if not review:
        return "DeepSeek worker: not run."

    status = str(review.get("status", "unknown"))
    if status == "completed":
        return (
            "DeepSeek worker: reviewed "
            f"{review.get('selected_opportunity_count', 0)} opportunities, "
            f"approved {review.get('approved_count', 0)}, "
            f"rejected {review.get('rejected_count', 0)}, "
            f"watch {review.get('watch_count', 0)}."
        )
    reason = str(review.get("reason", "unspecified"))
    return f"DeepSeek worker: {status} ({reason})."
