from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import load_app_config
from src.station_truth import (
    backfill_station_actuals,
    backfill_station_actuals_from_cli_archive,
    load_station_map,
    station_actuals_path,
)

logger = logging.getLogger("weather.paper_trading")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_TRADES_DIR = PROJECT_ROOT / "data" / "paper_trades"
DEFAULT_LEDGER_PATH = PAPER_TRADES_DIR / "ledger.csv"
DEFAULT_SUMMARY_PATH = PAPER_TRADES_DIR / "summary.json"
PAPER_TRADE_FEE_MODEL = "flat_fee_per_contract_v1"
LEGACY_UNKNOWN_ROUTE = "legacy_unknown"
ROUTING_SOURCE_COLUMNS = [
    "forecast_calibration_source",
    "probability_calibration_source",
    "forecast_blend_source",
]
OBJECT_COLUMNS = [
    "trade_id",
    "scan_id",
    "recorded_at_utc",
    "status",
    "source",
    "ticker",
    "market_question",
    "city",
    "market_type",
    "market_date",
    "outcome",
    "position_side",
    "signal_direction",
    "fee_model",
    "entry_price_source",
    "forecast_calibration_source",
    "probability_calibration_source",
    "forecast_blend_source",
    "actual_field",
    "settlement_rule",
    "resolution_source",
    "settled_at_utc",
]

LEDGER_COLUMNS = [
    "trade_id",
    "scan_id",
    "recorded_at_utc",
    "status",
    "source",
    "ticker",
    "market_question",
    "city",
    "market_type",
    "market_date",
    "outcome",
    "position_side",
    "signal_direction",
    "contracts",
    "entry_price",
    "entry_price_source",
    "fee_model",
    "entry_fee_per_contract",
    "entry_fee",
    "settlement_fee_per_contract",
    "settlement_fee",
    "total_fees",
    "entry_cost",
    "our_probability",
    "raw_probability",
    "market_price",
    "edge",
    "forecast_value_f",
    "open_meteo_forecast_value_f",
    "raw_forecast_value_f",
    "forecast_blend_source",
    "hours_to_settlement",
    "forecast_calibration_source",
    "probability_calibration_source",
    "uncertainty_std_f",
    "actual_field",
    "settlement_rule",
    "settlement_low_f",
    "settlement_high_f",
    "yes_outcome",
    "actual_value_f",
    "resolution_source",
    "settled_at_utc",
    "payout",
    "pnl",
    "roi",
]


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_text(value: object, default: str | None = None) -> str | None:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    return text if text else default


def _normalize_iso_timestamp(value: str | datetime | None = None) -> str:
    if isinstance(value, datetime):
        dt = value
    elif value:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        dt = None if pd.isna(parsed) else parsed.to_pydatetime()
    else:
        dt = datetime.now(timezone.utc)

    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _normalize_date(value: str | date | datetime | None) -> Optional[str]:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _resolve_path(path_value: str | Path | None, default_path: Path) -> Path:
    path = Path(path_value) if path_value is not None else default_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_paper_trade_paths(config: Optional[dict] = None) -> tuple[Path, Path]:
    config = config or {}
    ledger_path = _resolve_path(config.get("paper_trade_ledger_path"), DEFAULT_LEDGER_PATH)
    summary_path = _resolve_path(config.get("paper_trade_summary_path"), DEFAULT_SUMMARY_PATH)
    return ledger_path, summary_path


def _runtime_paper_trade_config() -> dict:
    try:
        return load_app_config()
    except FileNotFoundError:
        return {}


def _resolve_fee_settings(config: Optional[dict] = None, *, load_runtime_defaults: bool = False) -> dict:
    effective_config = config
    if effective_config is None and load_runtime_defaults:
        effective_config = _runtime_paper_trade_config()
    if effective_config is None:
        effective_config = {}

    entry_fee_per_contract = _coerce_float(effective_config.get("paper_trade_entry_fee_per_contract"))
    settlement_fee_per_contract = _coerce_float(
        effective_config.get("paper_trade_settlement_fee_per_contract")
    )
    return {
        "fee_model": PAPER_TRADE_FEE_MODEL,
        "entry_fee_per_contract": round(max(entry_fee_per_contract or 0.0, 0.0), 4),
        "settlement_fee_per_contract": round(max(settlement_fee_per_contract or 0.0, 0.0), 4),
    }


def _calculate_fee_totals(contracts: float, fee_settings: dict) -> dict:
    entry_fee = float(contracts) * float(fee_settings["entry_fee_per_contract"])
    settlement_fee = float(contracts) * float(fee_settings["settlement_fee_per_contract"])
    return {
        "entry_fee": round(entry_fee, 4),
        "settlement_fee": round(settlement_fee, 4),
        "total_fees": round(entry_fee + settlement_fee, 4),
    }


def _entry_cost(entry_price: float, contracts: float, entry_fee: float) -> float:
    return round((float(entry_price) * float(contracts)) + float(entry_fee), 4)


def _row_fee_settings(row: dict | pd.Series, default_fee_settings: dict) -> dict:
    entry_fee_per_contract = _coerce_float(row.get("entry_fee_per_contract"))
    settlement_fee_per_contract = _coerce_float(row.get("settlement_fee_per_contract"))
    fee_model = default_fee_settings["fee_model"]
    fee_model_value = row.get("fee_model")
    if fee_model_value is not None:
        try:
            fee_model_missing = pd.isna(fee_model_value)
        except TypeError:
            fee_model_missing = False
        if not fee_model_missing and fee_model_value != "":
            fee_model = str(fee_model_value)

    return {
        "fee_model": fee_model,
        "entry_fee_per_contract": round(
            max(
                default_fee_settings["entry_fee_per_contract"]
                if entry_fee_per_contract is None
                else entry_fee_per_contract,
                0.0,
            ),
            4,
        ),
        "settlement_fee_per_contract": round(
            max(
                default_fee_settings["settlement_fee_per_contract"]
                if settlement_fee_per_contract is None
                else settlement_fee_per_contract,
                0.0,
            ),
            4,
        ),
    }


def _apply_fee_model_to_frame(frame: pd.DataFrame, default_fee_settings: dict) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    normalized = frame.copy()
    for index, row in normalized.iterrows():
        contracts = _coerce_float(row.get("contracts")) or 0.0
        entry_price = _coerce_float(row.get("entry_price")) or 0.0
        fee_settings = _row_fee_settings(row, default_fee_settings)
        fee_totals = _calculate_fee_totals(contracts, fee_settings)
        entry_cost = _entry_cost(entry_price, contracts, fee_totals["entry_fee"])

        normalized.at[index, "fee_model"] = fee_settings["fee_model"]
        normalized.at[index, "entry_fee_per_contract"] = fee_settings["entry_fee_per_contract"]
        normalized.at[index, "entry_fee"] = fee_totals["entry_fee"]
        normalized.at[index, "settlement_fee_per_contract"] = fee_settings["settlement_fee_per_contract"]
        normalized.at[index, "settlement_fee"] = fee_totals["settlement_fee"]
        normalized.at[index, "total_fees"] = fee_totals["total_fees"]
        normalized.at[index, "entry_cost"] = entry_cost

        payout = _coerce_float(row.get("payout"))
        if str(row.get("status", "")) != "settled" or payout is None:
            continue

        pnl = payout - (entry_price * contracts) - fee_totals["total_fees"]
        roi = (pnl / entry_cost) if entry_cost > 0 else None
        normalized.at[index, "pnl"] = round(pnl, 4)
        normalized.at[index, "roi"] = pd.NA if roi is None else round(roi, 4)

    return normalized


def _normalize_routing_source(value: object) -> str:
    return _coerce_text(value, LEGACY_UNKNOWN_ROUTE) or LEGACY_UNKNOWN_ROUTE


def _apply_routing_metadata_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    normalized = frame.copy()
    for column in ROUTING_SOURCE_COLUMNS:
        normalized[column] = normalized[column].apply(_normalize_routing_source)

    for index, row in normalized.iterrows():
        forecast_value = _coerce_float(row.get("forecast_value_f"))
        our_probability = _coerce_float(row.get("our_probability"))
        raw_forecast_value = _coerce_float(row.get("raw_forecast_value_f"))
        open_meteo_forecast_value = _coerce_float(row.get("open_meteo_forecast_value_f"))
        hours_to_settlement = _coerce_float(row.get("hours_to_settlement"))
        raw_probability = _coerce_float(row.get("raw_probability"))

        forecast_calibration_source = _normalize_routing_source(row.get("forecast_calibration_source"))
        probability_calibration_source = _normalize_routing_source(row.get("probability_calibration_source"))
        forecast_blend_source = _normalize_routing_source(row.get("forecast_blend_source"))

        if raw_forecast_value is None and forecast_value is not None and forecast_calibration_source in {
            "raw",
            "raw_selective_fallback",
        }:
            raw_forecast_value = forecast_value

        if (
            open_meteo_forecast_value is None
            and forecast_blend_source == "open-meteo"
            and raw_forecast_value is not None
        ):
            open_meteo_forecast_value = raw_forecast_value

        if raw_probability is None and our_probability is not None and probability_calibration_source == "raw":
            raw_probability = our_probability

        normalized.at[index, "raw_forecast_value_f"] = (
            pd.NA if raw_forecast_value is None else raw_forecast_value
        )
        normalized.at[index, "open_meteo_forecast_value_f"] = (
            pd.NA if open_meteo_forecast_value is None else open_meteo_forecast_value
        )
        normalized.at[index, "hours_to_settlement"] = pd.NA if hours_to_settlement is None else hours_to_settlement
        normalized.at[index, "raw_probability"] = pd.NA if raw_probability is None else raw_probability

    return normalized


def _load_ledger_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)

    frame = pd.read_csv(path)
    for column in LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[LEDGER_COLUMNS].copy()
    for column in OBJECT_COLUMNS:
        frame[column] = frame[column].astype("object")
    return _apply_routing_metadata_defaults(frame)


def _write_ledger_frame(frame: pd.DataFrame, path: Path) -> None:
    ordered = _apply_routing_metadata_defaults(frame)
    for column in LEDGER_COLUMNS:
        if column not in ordered.columns:
            ordered[column] = pd.NA
    ordered = ordered[LEDGER_COLUMNS]
    ordered = ordered.sort_values(
        by=["status", "market_date", "recorded_at_utc", "source", "ticker", "position_side"],
        kind="stable",
        na_position="last",
    )
    ordered.to_csv(path, index=False)


def _actual_field_for_market_type(market_type: str) -> str:
    return "tmin_f" if "low" in str(market_type).lower() else "tmax_f"


def _position_side(opportunity: dict) -> str:
    edge = _coerce_float(opportunity.get("edge")) or 0.0
    return "yes" if edge >= 0 else "no"


def _entry_price_for_opportunity(opportunity: dict, side: str) -> tuple[float, str]:
    market_price = _coerce_float(opportunity.get("market_price")) or 0.0
    yes_bid = _coerce_float(opportunity.get("yes_bid"))
    yes_ask = _coerce_float(opportunity.get("yes_ask"))

    if side == "yes":
        if yes_ask is not None and yes_ask > 0:
            return min(max(yes_ask, 0.001), 0.999), "yes_ask"
        return min(max(market_price, 0.001), 0.999), "market_price"

    if yes_bid is not None and 0 <= yes_bid < 1:
        return min(max(1.0 - yes_bid, 0.001), 0.999), "synthetic_no_ask_from_yes_bid"
    return min(max(1.0 - market_price, 0.001), 0.999), "synthetic_no_price_from_market"


def _open_position_key(source: str, ticker: str, outcome: str, side: str) -> str:
    return "|".join([source or "", ticker or "", outcome or "", side or ""])


def paper_trade_record_from_opportunity(
    opportunity: dict,
    *,
    scan_timestamp: str | datetime | None = None,
    contracts: float = 1.0,
    config: Optional[dict] = None,
    fee_settings: Optional[dict] = None,
) -> dict:
    scan_id = _normalize_iso_timestamp(scan_timestamp)
    side = _position_side(opportunity)
    entry_price, entry_price_source = _entry_price_for_opportunity(opportunity, side)
    contracts = float(contracts)
    effective_fee_settings = fee_settings or _resolve_fee_settings(config)
    fee_totals = _calculate_fee_totals(contracts, effective_fee_settings)
    trade_key = "|".join(
        [
            scan_id,
            str(opportunity.get("source", "")),
            str(opportunity.get("ticker", "")),
            str(opportunity.get("outcome", "")),
            side,
        ]
    )
    trade_id = hashlib.sha1(trade_key.encode("utf-8")).hexdigest()[:16]

    settlement_low_f = _coerce_float(opportunity.get("settlement_low_f"))
    settlement_high_f = _coerce_float(opportunity.get("settlement_high_f"))

    return {
        "trade_id": trade_id,
        "scan_id": scan_id,
        "recorded_at_utc": scan_id,
        "status": "open",
        "source": opportunity.get("source"),
        "ticker": opportunity.get("ticker"),
        "market_question": opportunity.get("market_question"),
        "city": opportunity.get("city"),
        "market_type": opportunity.get("market_type"),
        "market_date": _normalize_date(opportunity.get("market_date")),
        "outcome": opportunity.get("outcome"),
        "position_side": side,
        "signal_direction": opportunity.get("direction"),
        "contracts": contracts,
        "entry_price": round(entry_price, 4),
        "entry_price_source": entry_price_source,
        "fee_model": effective_fee_settings["fee_model"],
        "entry_fee_per_contract": effective_fee_settings["entry_fee_per_contract"],
        "entry_fee": fee_totals["entry_fee"],
        "settlement_fee_per_contract": effective_fee_settings["settlement_fee_per_contract"],
        "settlement_fee": fee_totals["settlement_fee"],
        "total_fees": fee_totals["total_fees"],
        "entry_cost": _entry_cost(entry_price, contracts, fee_totals["entry_fee"]),
        "our_probability": _coerce_float(opportunity.get("our_probability")),
        "raw_probability": _coerce_float(opportunity.get("raw_probability")),
        "market_price": _coerce_float(opportunity.get("market_price")),
        "edge": _coerce_float(opportunity.get("edge")),
        "forecast_value_f": _coerce_float(opportunity.get("forecast_value_f")),
        "open_meteo_forecast_value_f": _coerce_float(opportunity.get("open_meteo_forecast_value_f")),
        "raw_forecast_value_f": _coerce_float(opportunity.get("raw_forecast_value_f")),
        "forecast_blend_source": _normalize_routing_source(opportunity.get("forecast_blend_source")),
        "hours_to_settlement": _coerce_float(opportunity.get("hours_to_settlement")),
        "forecast_calibration_source": _normalize_routing_source(
            opportunity.get("forecast_calibration_source")
        ),
        "probability_calibration_source": _normalize_routing_source(
            opportunity.get("probability_calibration_source")
        ),
        "uncertainty_std_f": _coerce_float(opportunity.get("uncertainty_std_f")),
        "actual_field": opportunity.get("actual_field")
        or _actual_field_for_market_type(str(opportunity.get("market_type", ""))),
        "settlement_rule": opportunity.get("settlement_rule"),
        "settlement_low_f": settlement_low_f,
        "settlement_high_f": settlement_high_f,
        "yes_outcome": pd.NA,
        "actual_value_f": pd.NA,
        "resolution_source": pd.NA,
        "settled_at_utc": pd.NA,
        "payout": pd.NA,
        "pnl": pd.NA,
        "roi": pd.NA,
    }


def log_paper_trades(
    opportunities: list[dict],
    *,
    scan_timestamp: str | datetime | None = None,
    ledger_path: str | Path | None = None,
    contracts: float = 1.0,
    config: Optional[dict] = None,
) -> dict:
    ledger = _resolve_path(ledger_path, DEFAULT_LEDGER_PATH)
    fee_settings = _resolve_fee_settings(config, load_runtime_defaults=config is None)
    frame = _apply_fee_model_to_frame(_load_ledger_frame(ledger), fee_settings)
    open_rows = frame.loc[frame["status"].fillna("open") != "settled"].copy()
    open_keys = {
        _open_position_key(
            str(row.get("source", "")),
            str(row.get("ticker", "")),
            str(row.get("outcome", "")),
            str(row.get("position_side", "")),
        )
        for _, row in open_rows.iterrows()
    }

    new_records = []
    skipped_existing = 0
    skipped_zero_size = 0
    use_kelly = bool(
        config
        and (
            config.get("execution", {}).get("sizing") == "quarter_kelly"
            or str(config.get("strategy_policy", {}).get("execution", {}).get("sizing", "")) == "quarter_kelly"
        )
    )
    # Also honor sizing if specified at the top-level policy execution block from strategy_policy.json
    if not use_kelly and config:
        policy_obj = config.get("_strategy_policy_payload") or {}
        if policy_obj.get("execution", {}).get("sizing") == "quarter_kelly":
            use_kelly = True

    for opportunity in opportunities:
        if use_kelly:
            from src.sizing import compute_position_size
            side_upper = str(opportunity.get("direction", "")).upper()
            trade_contracts = float(compute_position_size(
                edge=float(opportunity.get("edge") or 0.0),
                price=float(opportunity.get("market_price") or 0.0),
                side=side_upper,
                kelly_fraction=float(config.get("kelly_fraction", 0.25)),
                bankroll_dollars=float(config.get("bankroll_dollars", 100.0)),
                max_order_cost_dollars=float(
                    config.get("paper_trade_max_order_cost_dollars",
                        config.get("max_order_cost_dollars", 10.0))
                ),
                hard_cap_contracts=int(config.get("max_contracts_hard_cap", 20)),
            ))
            if trade_contracts <= 0:
                skipped_zero_size += 1
                continue
        else:
            trade_contracts = contracts

        record = paper_trade_record_from_opportunity(
            opportunity,
            scan_timestamp=scan_timestamp,
            contracts=trade_contracts,
            fee_settings=fee_settings,
        )
        key = _open_position_key(
            str(record.get("source", "")),
            str(record.get("ticker", "")),
            str(record.get("outcome", "")),
            str(record.get("position_side", "")),
        )
        if key in open_keys:
            skipped_existing += 1
            continue
        open_keys.add(key)
        new_records.append(record)

    if new_records:
        new_frame = pd.DataFrame(new_records, columns=LEDGER_COLUMNS)
        if frame.empty:
            frame = new_frame
        else:
            frame = pd.concat([frame, new_frame], ignore_index=True, sort=False)
        _write_ledger_frame(frame, ledger)

    return {
        "ledger_path": str(ledger),
        "new_trades": len(new_records),
        "skipped_existing_open_positions": skipped_existing,
        "skipped_zero_size": skipped_zero_size,
        "total_open_positions": int((frame["status"].fillna("open") != "settled").sum()),
    }


def _resolve_yes_outcome(
    actual_value_f: float,
    settlement_rule: str,
    settlement_low_f: Optional[float],
    settlement_high_f: Optional[float],
) -> bool:
    if settlement_rule == "between_inclusive":
        if settlement_low_f is None or settlement_high_f is None:
            raise ValueError("Range settlement requires both low and high bounds")
        return settlement_low_f <= actual_value_f <= settlement_high_f
    if settlement_rule == "lte":
        if settlement_high_f is None:
            raise ValueError("LTE settlement requires an upper bound")
        return actual_value_f <= settlement_high_f
    if settlement_rule == "lt":
        if settlement_high_f is None:
            raise ValueError("LT settlement requires an upper bound")
        return actual_value_f < settlement_high_f
    if settlement_rule == "gte":
        if settlement_low_f is None:
            raise ValueError("GTE settlement requires a lower bound")
        return actual_value_f >= settlement_low_f
    if settlement_rule == "gt":
        if settlement_low_f is None:
            raise ValueError("GT settlement requires a lower bound")
        return actual_value_f > settlement_low_f
    raise ValueError(f"Unsupported settlement rule: {settlement_rule}")


def _settlement_refresh_cutoff_date(
    as_of_date: str | date | datetime | None = None,
    *,
    today: str | date | datetime | None = None,
) -> str:
    today_str = _normalize_date(today)
    if today_str is None:
        today_date = datetime.now(timezone.utc).date()
    else:
        today_date = date.fromisoformat(today_str)

    cutoff = today_date - timedelta(days=1)
    as_of_date_str = _normalize_date(as_of_date)
    if as_of_date_str is None:
        return cutoff.isoformat()

    return min(cutoff, date.fromisoformat(as_of_date_str)).isoformat()


def plan_station_actuals_refresh(
    *,
    ledger_path: str | Path | None = None,
    as_of_date: str | date | datetime | None = None,
    today: str | date | datetime | None = None,
) -> dict:
    ledger = _resolve_path(ledger_path, DEFAULT_LEDGER_PATH)
    frame = _load_ledger_frame(ledger)
    cutoff_date = _settlement_refresh_cutoff_date(as_of_date=as_of_date, today=today)
    plan = {
        "ledger_path": str(ledger),
        "cutoff_date": cutoff_date,
        "needs_refresh": False,
        "open_trade_count": 0,
        "eligible_open_trade_count": 0,
        "city_windows": [],
    }

    if frame.empty:
        return plan

    open_frame = frame.loc[frame["status"].fillna("open") != "settled"].copy()
    plan["open_trade_count"] = int(len(open_frame))
    if open_frame.empty:
        return plan

    open_frame["market_date"] = open_frame["market_date"].apply(_normalize_date)
    open_frame["city"] = open_frame["city"].fillna("").astype(str).str.strip()
    eligible = open_frame.loc[
        open_frame["market_date"].notna()
        & open_frame["city"].ne("")
        & open_frame["market_date"].le(cutoff_date)
    ].copy()
    if eligible.empty:
        return plan

    city_windows = []
    for city, city_frame in eligible.groupby("city", sort=True):
        market_dates = sorted(city_frame["market_date"].dropna().unique().tolist())
        if not market_dates:
            continue
        city_windows.append(
            {
                "city": city,
                "start_date": market_dates[0],
                "end_date": cutoff_date,
                "market_dates": market_dates,
                "open_trade_count": int(len(city_frame)),
            }
        )

    plan["needs_refresh"] = bool(city_windows)
    plan["eligible_open_trade_count"] = int(len(eligible))
    plan["city_windows"] = city_windows
    return plan


def _eligible_open_trades_for_truth_check(frame: pd.DataFrame, cutoff_date: str | None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    open_frame = frame.loc[frame["status"].fillna("open") != "settled"].copy()
    if open_frame.empty:
        return open_frame

    open_frame["market_date"] = open_frame["market_date"].apply(_normalize_date)
    open_frame["city"] = open_frame["city"].fillna("").astype(str).str.strip()
    open_frame["actual_field"] = open_frame["actual_field"].fillna("").astype(str).str.strip()

    eligible = open_frame.loc[
        open_frame["market_date"].notna()
        & open_frame["city"].ne("")
        & open_frame["actual_field"].ne("")
    ].copy()
    if cutoff_date is not None:
        eligible = eligible.loc[eligible["market_date"].le(cutoff_date)].copy()
    return eligible


def _load_station_actuals_frame(
    city: str,
    station_actuals_dir: str | Path | None,
    cache: dict[str, pd.DataFrame],
) -> tuple[Path, pd.DataFrame]:
    city_path = station_actuals_path(city, base_dir=station_actuals_dir)
    cache_key = str(city_path)
    if cache_key not in cache:
        if not city_path.exists():
            cache[cache_key] = pd.DataFrame()
        else:
            cache[cache_key] = pd.read_csv(city_path)
    return city_path, cache[cache_key]


def _latest_station_actual_date(
    city: str,
    station_actuals_dir: str | Path | None,
    cache: dict[str, pd.DataFrame],
) -> Optional[str]:
    _, frame = _load_station_actuals_frame(city, station_actuals_dir, cache)
    if frame.empty or "date" not in frame.columns:
        return None

    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().date().isoformat()


def _missing_truth_blocker_details(
    frame: pd.DataFrame,
    *,
    station_actuals_dir: str | Path | None = None,
    cutoff_date: str | None = None,
) -> dict:
    eligible = _eligible_open_trades_for_truth_check(frame, cutoff_date)
    details = {
        "settlement_cutoff_date": cutoff_date,
        "eligible_open_trade_count": int(len(eligible)),
        "missing_required_truth_blocker": False,
        "missing_required_truth_trade_count": 0,
        "missing_required_truth_date_count": 0,
        "missing_required_truth_dates": [],
        "missing_required_truth_city_windows": [],
    }
    if eligible.empty:
        return details

    actual_cache: dict[str, pd.DataFrame] = {}
    missing_dates: list[dict] = []
    for (city, market_date, actual_field), group in eligible.groupby(
        ["city", "market_date", "actual_field"],
        sort=True,
    ):
        actual_value = _load_actual_value(
            str(city),
            str(market_date),
            str(actual_field),
            station_actuals_dir,
            actual_cache,
        )
        if actual_value is not None:
            continue

        missing_dates.append(
            {
                "city": str(city),
                "market_date": str(market_date),
                "actual_field": str(actual_field),
                "open_trade_count": int(len(group)),
                "latest_local_date": _latest_station_actual_date(
                    str(city),
                    station_actuals_dir,
                    actual_cache,
                ),
            }
        )

    if not missing_dates:
        return details

    windows: list[dict] = []
    missing_frame = pd.DataFrame(missing_dates)
    for city, city_frame in missing_frame.groupby("city", sort=True):
        market_dates = sorted(city_frame["market_date"].dropna().astype(str).unique().tolist())
        latest_dates = sorted(
            {
                str(value)
                for value in city_frame["latest_local_date"].dropna().astype(str).tolist()
                if value
            }
        )
        windows.append(
            {
                "city": str(city),
                "start_date": market_dates[0],
                "end_date": market_dates[-1],
                "market_dates": market_dates,
                "open_trade_count": int(pd.to_numeric(city_frame["open_trade_count"], errors="coerce").fillna(0).sum()),
                "latest_local_date": latest_dates[-1] if latest_dates else None,
            }
        )

    details.update(
        {
            "missing_required_truth_blocker": True,
            "missing_required_truth_trade_count": int(
                pd.to_numeric(missing_frame["open_trade_count"], errors="coerce").fillna(0).sum()
            ),
            "missing_required_truth_date_count": int(len(missing_dates)),
            "missing_required_truth_dates": missing_dates,
            "missing_required_truth_city_windows": windows,
        }
    )
    return details


def _build_retry_city_windows(missing_required_truth_dates: list[dict]) -> list[dict]:
    windows: list[dict] = []
    if not missing_required_truth_dates:
        return windows

    by_city: dict[str, dict[str, int]] = {}
    for item in missing_required_truth_dates:
        city = str(item.get("city", "") or "").strip()
        market_date = _normalize_date(item.get("market_date"))
        if not city or market_date is None:
            continue
        open_trade_count = int(item.get("open_trade_count", 0) or 0)
        by_city.setdefault(city, {})
        by_city[city][market_date] = by_city[city].get(market_date, 0) + open_trade_count

    for city in sorted(by_city):
        dated_counts = sorted(
            ((date.fromisoformat(market_date), count) for market_date, count in by_city[city].items()),
            key=lambda item: item[0],
        )
        if not dated_counts:
            continue

        current_dates = [dated_counts[0][0]]
        current_count = dated_counts[0][1]
        previous_date = dated_counts[0][0]

        for current_date, open_trade_count in dated_counts[1:]:
            if (current_date - previous_date).days == 1:
                current_dates.append(current_date)
                current_count += open_trade_count
            else:
                windows.append(
                    {
                        "city": city,
                        "start_date": current_dates[0].isoformat(),
                        "end_date": current_dates[-1].isoformat(),
                        "market_dates": [value.isoformat() for value in current_dates],
                        "open_trade_count": current_count,
                    }
                )
                current_dates = [current_date]
                current_count = open_trade_count
            previous_date = current_date

        windows.append(
            {
                "city": city,
                "start_date": current_dates[0].isoformat(),
                "end_date": current_dates[-1].isoformat(),
                "market_dates": [value.isoformat() for value in current_dates],
                "open_trade_count": current_count,
            }
        )

    return windows


def _run_station_actuals_refresh_pass(
    city_windows: list[dict],
    *,
    refresh_source: str,
    ncei_token: str,
    station_actuals_dir: str | Path | None,
    stations: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    refreshed_cities: list[dict] = []
    failed_cities: list[dict] = []

    for city_window in city_windows:
        city = str(city_window["city"])
        start_date = str(city_window["start_date"])
        end_date = str(city_window["end_date"])
        try:
            if refresh_source == "cdo":
                path = backfill_station_actuals(
                    city,
                    start_date,
                    end_date,
                    token=ncei_token,
                    base_dir=station_actuals_dir,
                    stations=stations,
                )
            elif refresh_source == "cli_archive":
                path = backfill_station_actuals_from_cli_archive(
                    city,
                    start=start_date,
                    end=end_date,
                    base_dir=station_actuals_dir,
                    stations=stations,
                )
            else:
                raise ValueError(f"Unsupported refresh source: {refresh_source}")

            refreshed_cities.append(
                {
                    "city": city,
                    "start_date": start_date,
                    "end_date": end_date,
                    "path": str(path),
                }
            )
        except Exception as exc:
            logger.warning(
                "Station-actual refresh failed for %s via %s (%s to %s): %s",
                city,
                refresh_source,
                start_date,
                end_date,
                exc,
            )
            failed_cities.append(
                {
                    "city": city,
                    "start_date": start_date,
                    "end_date": end_date,
                    "error": str(exc),
                }
            )

    return refreshed_cities, failed_cities


def refresh_station_actuals_for_open_trades(
    *,
    config: Optional[dict] = None,
    ledger_path: str | Path | None = None,
    station_actuals_dir: str | Path | None = None,
    as_of_date: str | date | datetime | None = None,
    today: str | date | datetime | None = None,
) -> dict:
    config = config or {}
    effective_ledger_path = ledger_path if ledger_path is not None else config.get("paper_trade_ledger_path")
    plan = plan_station_actuals_refresh(
        ledger_path=effective_ledger_path,
        as_of_date=as_of_date,
        today=today,
    )
    ledger = _resolve_path(effective_ledger_path, DEFAULT_LEDGER_PATH)
    frame = _load_ledger_frame(ledger)
    ncei_token = str(config.get("ncei_api_token", "") or "").strip()
    refresh_source = "cdo" if ncei_token else "cli_archive"
    fallback_refresh_source = "cli_archive" if refresh_source == "cdo" else None
    result = {
        **plan,
        "attempted": False,
        "refresh_source": None,
        "refreshed_cities": [],
        "failed_cities": [],
        "fallback_attempted": False,
        "fallback_refresh_source": fallback_refresh_source,
        "fallback_city_windows": [],
        "fallback_refreshed_cities": [],
        "fallback_failed_cities": [],
        **_missing_truth_blocker_details(
            frame,
            station_actuals_dir=station_actuals_dir,
            cutoff_date=plan["cutoff_date"],
        ),
    }
    if not plan["needs_refresh"]:
        return result

    stations = load_station_map()
    result["attempted"] = True
    result["refresh_source"] = refresh_source
    refreshed_cities, failed_cities = _run_station_actuals_refresh_pass(
        plan["city_windows"],
        refresh_source=refresh_source,
        ncei_token=ncei_token,
        station_actuals_dir=station_actuals_dir,
        stations=stations,
    )
    result["refreshed_cities"] = refreshed_cities
    result["failed_cities"] = failed_cities

    post_primary_details = _missing_truth_blocker_details(
        frame,
        station_actuals_dir=station_actuals_dir,
        cutoff_date=plan["cutoff_date"],
    )
    fallback_city_windows = _build_retry_city_windows(post_primary_details["missing_required_truth_dates"])
    result["fallback_city_windows"] = fallback_city_windows

    if fallback_refresh_source and fallback_city_windows:
        result["fallback_attempted"] = True
        fallback_refreshed_cities, fallback_failed_cities = _run_station_actuals_refresh_pass(
            fallback_city_windows,
            refresh_source=fallback_refresh_source,
            ncei_token=ncei_token,
            station_actuals_dir=station_actuals_dir,
            stations=stations,
        )
        result["fallback_refreshed_cities"] = fallback_refreshed_cities
        result["fallback_failed_cities"] = fallback_failed_cities

    result.update(
        _missing_truth_blocker_details(
            frame,
            station_actuals_dir=station_actuals_dir,
            cutoff_date=plan["cutoff_date"],
        )
    )
    return result


def _load_actual_value(
    city: str,
    market_date: str,
    actual_field: str,
    station_actuals_dir: str | Path | None,
    cache: dict[str, pd.DataFrame],
) -> Optional[float]:
    _, frame = _load_station_actuals_frame(city, station_actuals_dir, cache)
    if frame.empty or actual_field not in frame.columns or "date" not in frame.columns:
        return None

    matches = frame.loc[frame["date"].astype(str) == market_date]
    if matches.empty:
        return None

    value = _coerce_float(matches.iloc[-1].get(actual_field))
    return None if value is None else round(value, 2)


def _settled_profitability_metrics(frame: pd.DataFrame) -> dict:
    settled = frame.loc[frame["status"] == "settled"].copy()

    total_entry_cost = 0.0
    total_fees = 0.0
    total_pnl = 0.0
    win_rate = None
    roi = None
    if not settled.empty:
        entry_costs = pd.to_numeric(settled["entry_cost"], errors="coerce").fillna(0.0)
        fees = pd.to_numeric(settled["total_fees"], errors="coerce").fillna(0.0)
        pnls = pd.to_numeric(settled["pnl"], errors="coerce").fillna(0.0)
        total_entry_cost = float(entry_costs.sum())
        total_fees = float(fees.sum())
        total_pnl = float(pnls.sum())
        win_rate = float((pnls > 0).mean()) if len(pnls) else None
        roi = float(total_pnl / total_entry_cost) if total_entry_cost > 0 else None

    return {
        "settled_trades": int(len(settled)),
        "total_entry_cost": round(total_entry_cost, 4),
        "total_fees": round(total_fees, 4),
        "total_pnl": round(total_pnl, 4),
        "roi": None if roi is None else round(roi, 4),
        "win_rate": None if win_rate is None else round(win_rate, 4),
    }


def _forecast_calibration_source_breakdown(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}

    normalized = _apply_routing_metadata_defaults(frame)
    breakdown: dict[str, dict] = {}
    for forecast_source, source_frame in normalized.groupby("forecast_calibration_source", sort=True):
        breakdown[str(forecast_source)] = {
            "total_trades": int(len(source_frame)),
            "open_trades": int((source_frame["status"].fillna("open") != "settled").sum()),
            **_settled_profitability_metrics(source_frame),
        }

    return breakdown


def _build_summary(
    frame: pd.DataFrame,
    ledger_path: Path,
    newly_settled: int,
    blocker_details: Optional[dict] = None,
) -> dict:
    open_count = int((frame["status"].fillna("open") != "settled").sum())
    profitability = _settled_profitability_metrics(frame)

    summary = {
        "ledger_path": str(ledger_path),
        "total_trades": int(len(frame)),
        "open_trades": open_count,
        "newly_settled_trades": int(newly_settled),
        **profitability,
        "forecast_calibration_source_breakdown": _forecast_calibration_source_breakdown(frame),
    }
    if blocker_details:
        summary.update(blocker_details)
    return summary


def settle_paper_trades(
    *,
    ledger_path: str | Path | None = None,
    station_actuals_dir: str | Path | None = None,
    summary_path: str | Path | None = None,
    as_of_date: str | date | datetime | None = None,
    config: Optional[dict] = None,
) -> dict:
    ledger = _resolve_path(ledger_path, DEFAULT_LEDGER_PATH)
    summary_file = _resolve_path(summary_path, DEFAULT_SUMMARY_PATH)
    fee_settings = _resolve_fee_settings(config, load_runtime_defaults=config is None)
    frame = _apply_fee_model_to_frame(_load_ledger_frame(ledger), fee_settings)
    cutoff_date = _settlement_refresh_cutoff_date(as_of_date=as_of_date)

    if frame.empty:
        summary = _build_summary(
            frame,
            ledger,
            0,
            blocker_details=_missing_truth_blocker_details(
                frame,
                station_actuals_dir=station_actuals_dir,
                cutoff_date=cutoff_date,
            ),
        )
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    as_of_date_str = _normalize_date(as_of_date)
    actual_cache: dict[str, pd.DataFrame] = {}
    settled_now = 0
    settled_at = _normalize_iso_timestamp()

    for index, row in frame.iterrows():
        if str(row.get("status", "open")) == "settled":
            continue

        market_date = _normalize_date(row.get("market_date"))
        if market_date is None:
            continue
        if as_of_date_str is not None and market_date > as_of_date_str:
            continue

        city = str(row.get("city", ""))
        actual_field_value = row.get("actual_field")
        actual_field = ""
        if actual_field_value is not None:
            try:
                actual_field_missing = pd.isna(actual_field_value)
            except TypeError:
                actual_field_missing = False
            if not actual_field_missing and actual_field_value != "":
                actual_field = str(actual_field_value)
        if not city or not actual_field:
            continue

        actual_value_f = _load_actual_value(
            city,
            market_date,
            actual_field,
            station_actuals_dir,
            actual_cache,
        )
        if actual_value_f is None:
            continue

        settlement_low_f = _coerce_float(row.get("settlement_low_f"))
        settlement_high_f = _coerce_float(row.get("settlement_high_f"))
        yes_outcome = _resolve_yes_outcome(
            actual_value_f,
            str(row.get("settlement_rule", "")),
            settlement_low_f,
            settlement_high_f,
        )

        contracts = _coerce_float(row.get("contracts")) or 0.0
        entry_price = _coerce_float(row.get("entry_price")) or 0.0
        trade_fee_settings = _row_fee_settings(row, fee_settings)
        fee_totals = _calculate_fee_totals(contracts, trade_fee_settings)
        entry_cost = _entry_cost(entry_price, contracts, fee_totals["entry_fee"])
        payout = contracts if (str(row.get("position_side", "")) == "yes") == yes_outcome else 0.0
        pnl = payout - (entry_price * contracts) - fee_totals["total_fees"]
        roi = (pnl / entry_cost) if entry_cost > 0 else None

        frame.at[index, "status"] = "settled"
        frame.at[index, "yes_outcome"] = int(yes_outcome)
        frame.at[index, "actual_value_f"] = round(actual_value_f, 2)
        frame.at[index, "resolution_source"] = "station_actuals"
        frame.at[index, "settled_at_utc"] = settled_at
        frame.at[index, "fee_model"] = trade_fee_settings["fee_model"]
        frame.at[index, "entry_fee_per_contract"] = trade_fee_settings["entry_fee_per_contract"]
        frame.at[index, "entry_fee"] = fee_totals["entry_fee"]
        frame.at[index, "settlement_fee_per_contract"] = trade_fee_settings["settlement_fee_per_contract"]
        frame.at[index, "settlement_fee"] = fee_totals["settlement_fee"]
        frame.at[index, "total_fees"] = fee_totals["total_fees"]
        frame.at[index, "entry_cost"] = entry_cost
        frame.at[index, "payout"] = round(payout, 4)
        frame.at[index, "pnl"] = round(pnl, 4)
        frame.at[index, "roi"] = pd.NA if roi is None else round(roi, 4)
        settled_now += 1

    _write_ledger_frame(frame, ledger)
    summary = _build_summary(
        frame,
        ledger,
        settled_now,
        blocker_details=_missing_truth_blocker_details(
            frame,
            station_actuals_dir=station_actuals_dir,
            cutoff_date=cutoff_date,
        ),
    )
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_paper_trade_summary(summary: dict) -> str:
    roi = summary.get("roi")
    win_rate = summary.get("win_rate")
    roi_text = "n/a" if roi is None else f"{roi:.2%}"
    win_rate_text = "n/a" if win_rate is None else f"{win_rate:.2%}"
    message = (
        f"Paper trades: {summary.get('total_trades', 0)} total, "
        f"{summary.get('open_trades', 0)} open, "
        f"{summary.get('settled_trades', 0)} settled, "
        f"PnL {summary.get('total_pnl', 0.0):+.4f}, "
        f"ROI {roi_text}, win rate {win_rate_text}."
    )
    if summary.get("missing_required_truth_blocker"):
        windows = summary.get("missing_required_truth_city_windows", [])
        window_parts = []
        for window in windows[:3]:
            latest_local_date = window.get("latest_local_date") or "none"
            window_parts.append(
                f"{window.get('city')} {window.get('start_date')}..{window.get('end_date')} "
                f"(latest local {latest_local_date})"
            )
        if len(windows) > 3:
            window_parts.append(f"+{len(windows) - 3} more")
        window_text = "; ".join(window_parts) if window_parts else "details unavailable"
        message += (
            f" Missing truth blocker for {summary.get('missing_required_truth_trade_count', 0)} "
            f"eligible open trade(s) across {summary.get('missing_required_truth_date_count', 0)} "
            f"city/date requirement(s): {window_text}."
        )
    return message
