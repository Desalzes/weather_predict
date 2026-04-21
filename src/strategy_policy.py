from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from src.config import PROJECT_ROOT

DEFAULT_STRATEGY_POLICY_PATH = PROJECT_ROOT / "strategy" / "strategy_policy.json"

DEFAULT_STRATEGY_POLICY = {
    "policy_version": 1,
    "status": "bootstrap",
    "generated_at_utc": None,
    "generated_by": "bootstrap",
    "objective": "Take only the strongest paper trades under hard risk limits.",
    "selection": {
        "sources": ["kalshi"],
        "min_abs_edge": 0.08,
        "min_volume24hr": 500,
        "max_candidates_per_scan": 5,
        "max_hours_to_settlement": 24,
        "allowed_market_types": ["high", "low"],
        "allowed_cities": [],
        "blocked_cities": [],
    },
    "execution": {
        "max_contracts_per_trade": 1,
        "max_new_orders_per_day": 3,
        "max_order_cost_dollars": 10.0,
        "time_in_force": "fill_or_kill",
    },
    "deepseek": {
        "instruction": (
            "Approve only high-confidence trades that fit the policy, avoid thin/noisy setups, "
            "and keep token use minimal by relying on the supplied structured inputs."
        ),
        "temperature": 0.1,
        "max_tokens": 600,
    },
}


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_strategy_policy_path(path_value: str | None = None) -> Path:
    path = Path(path_value) if path_value else DEFAULT_STRATEGY_POLICY_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_strategy_policy(path_value: str | None = None) -> tuple[dict, Path]:
    path = resolve_strategy_policy_path(path_value)
    if not path.exists():
        return deepcopy(DEFAULT_STRATEGY_POLICY), path

    payload = json.loads(path.read_text(encoding="utf-8"))
    return _merge_dicts(DEFAULT_STRATEGY_POLICY, payload), path


def persist_strategy_policy(policy: dict, path_value: str | None = None) -> Path:
    path = resolve_strategy_policy_path(path_value)
    path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return path


def compact_policy_for_prompt(policy: dict) -> dict:
    selection = policy.get("selection", {})
    execution = policy.get("execution", {})
    deepseek = policy.get("deepseek", {})
    return {
        "policy_version": policy.get("policy_version"),
        "status": policy.get("status"),
        "objective": policy.get("objective"),
        "selection": {
            "sources": selection.get("sources", []),
            "min_abs_edge": selection.get("min_abs_edge"),
            "min_volume24hr": selection.get("min_volume24hr"),
            "max_candidates_per_scan": selection.get("max_candidates_per_scan"),
            "max_hours_to_settlement": selection.get("max_hours_to_settlement"),
            "allowed_market_types": selection.get("allowed_market_types", []),
            "allowed_settlement_rules": selection.get("allowed_settlement_rules", []),
            "allowed_position_sides": selection.get("allowed_position_sides", []),
            "allowed_cities": selection.get("allowed_cities", []),
            "blocked_cities": selection.get("blocked_cities", []),
        },
        "execution": {
            "max_contracts_per_trade": execution.get("max_contracts_per_trade"),
            "max_new_orders_per_day": execution.get("max_new_orders_per_day"),
            "max_order_cost_dollars": execution.get("max_order_cost_dollars"),
            "time_in_force": execution.get("time_in_force"),
        },
        "deepseek": {
            "instruction": deepseek.get("instruction"),
        },
    }


def filter_opportunities_for_policy(opportunities: list[dict], policy: dict) -> list[dict]:
    policy_category = (policy or {}).get("market_category")
    if policy_category:
        opportunities = [
            o for o in opportunities
            if (o.get("market_category") or "temperature") == policy_category
        ]
    selection = policy.get("selection", {})
    allowed_sources = {
        str(item).strip().lower()
        for item in selection.get("sources", [])
        if str(item).strip()
    }
    allowed_market_types = {
        str(item).strip().lower()
        for item in selection.get("allowed_market_types", [])
        if str(item).strip()
    }
    allowed_cities = {
        str(item).strip().lower()
        for item in selection.get("allowed_cities", [])
        if str(item).strip()
    }
    blocked_cities = {
        str(item).strip().lower()
        for item in selection.get("blocked_cities", [])
        if str(item).strip()
    }
    allowed_settlement_rules = {
        str(item).strip().lower()
        for item in selection.get("allowed_settlement_rules", [])
        if str(item).strip()
    }
    allowed_position_sides = {
        str(item).strip().lower()
        for item in selection.get("allowed_position_sides", [])
        if str(item).strip()
    }
    min_abs_edge = float(selection.get("min_abs_edge", 0.0) or 0.0)
    min_volume24hr = float(selection.get("min_volume24hr", 0.0) or 0.0)
    max_hours_to_settlement = selection.get("max_hours_to_settlement")
    max_candidates = max(int(selection.get("max_candidates_per_scan", 5) or 5), 1)

    filtered = []
    for opportunity in opportunities:
        source = str(opportunity.get("source", "")).strip().lower()
        city = str(opportunity.get("city", "")).strip().lower()
        market_type = str(opportunity.get("market_type", "")).strip().lower()
        abs_edge = float(opportunity.get("abs_edge", 0.0) or 0.0)
        volume24hr = float(opportunity.get("volume24hr", 0.0) or 0.0)
        hours_to_settlement = opportunity.get("hours_to_settlement")

        if allowed_sources and source not in allowed_sources:
            continue
        if allowed_market_types and market_type not in allowed_market_types:
            continue
        if allowed_cities and city not in allowed_cities:
            continue
        if blocked_cities and city in blocked_cities:
            continue
        if allowed_settlement_rules:
            settlement_rule = str(opportunity.get("settlement_rule", "")).strip().lower()
            if settlement_rule and settlement_rule not in allowed_settlement_rules:
                continue
        if allowed_position_sides:
            position_side = str(opportunity.get("position_side", "")).strip().lower()
            if position_side and position_side not in allowed_position_sides:
                continue
        if abs_edge < min_abs_edge:
            continue
        if volume24hr < min_volume24hr:
            continue
        if max_hours_to_settlement is not None and hours_to_settlement is not None:
            if float(hours_to_settlement) > float(max_hours_to_settlement):
                continue

        filtered.append(opportunity)

    filtered.sort(key=lambda item: float(item.get("abs_edge", 0.0) or 0.0), reverse=True)
    return filtered[:max_candidates]
