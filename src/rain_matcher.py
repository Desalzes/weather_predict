"""Rain market matcher for Kalshi KXRAIN binary any-rain contracts."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("weather.rain_matcher")

_BINARY_THRESHOLD_IN = 0.01
# Observed format in data/rain_market_inventory.csv:
#   KXRAINNYC-26APR21-T0 -> binary any-rain (>= 0.01 in)
# Future moderate/heavy tiers would use -T25, -T100, etc.; those are P1.5.
_BINARY_TICKER_PATTERN = re.compile(r"^KXRAIN[A-Z]{2,4}-\d{2}[A-Z]{3}\d{1,2}-T0$", re.IGNORECASE)
_BINARY_TEXT_PHRASES = {"yes", "no", "rain", "any rain"}
_BINARY_TEXT_PATTERN = re.compile(r"^\s*>?=?\s*0\.01\s*in\s*$", re.IGNORECASE)
_NON_BINARY_PATTERN = re.compile(r"(\d+\.\d+)", re.IGNORECASE)


def parse_rain_outcome(outcome: Optional[str], *, ticker: Optional[str] = None) -> Optional[dict]:
    """Parse a Kalshi KXRAIN outcome/yes_sub_title + ticker.

    Returns {"threshold_in": 0.01, "market_type": "rain_binary"} for binary
    any-rain markets. Returns None for non-binary thresholds (out of scope
    for P1).

    Precedence:
    1. If the ticker matches the `-T0` binary suffix pattern, treat as binary
       regardless of the text.
    2. Otherwise, interpret the outcome/title text.
    """
    if ticker and _BINARY_TICKER_PATTERN.match(str(ticker).strip()):
        return {"threshold_in": _BINARY_THRESHOLD_IN, "market_type": "rain_binary"}

    if outcome is None:
        return None
    normalized = str(outcome).strip().lower()

    # Exact binary phrases like "Rain in NYC", "Yes", "No" -- contain
    # "rain"/"yes"/"no" and no non-0.01 decimal threshold.
    decimal_matches = _NON_BINARY_PATTERN.findall(normalized)
    decimal_is_binary = (
        all(abs(float(m) - 0.01) < 1e-9 for m in decimal_matches)
        if decimal_matches
        else True
    )
    has_binary_word = any(phrase in normalized for phrase in _BINARY_TEXT_PHRASES)

    if _BINARY_TEXT_PATTERN.match(normalized):
        return {"threshold_in": _BINARY_THRESHOLD_IN, "market_type": "rain_binary"}
    if has_binary_word and decimal_is_binary:
        return {"threshold_in": _BINARY_THRESHOLD_IN, "market_type": "rain_binary"}

    return None


def compute_rain_yes_probability(
    calibrated_prob_any_rain: float,
    position_side: str = "yes",
) -> float:
    """Map a calibrated any-rain probability to YES-contract probability.

    For KXRAIN binary markets, the YES outcome resolves if precip_in >= 0.01,
    which is exactly the calibrated probability. For NO contracts, return
    1 - p. Clamps to [0.001, 0.999].
    """
    p = float(calibrated_prob_any_rain)
    p = max(0.001, min(0.999, p))
    if str(position_side).strip().lower() == "no":
        return 1.0 - p
    return p


def match_kalshi_rain(
    precip_forecasts: dict,
    markets: list[dict],
    *,
    calibration_manager=None,
    hrrr_data: Optional[dict] = None,
    hrrr_blend_horizon_hours: float = 12.0,
    min_edge: float = 0.15,
    now_utc: Optional[datetime] = None,
) -> list[dict]:
    """Compute opportunities for binary KXRAIN markets.

    Only processes markets whose ticker+outcome parses as rain_binary.
    Output shape mirrors matcher.match_kalshi_markets() opportunities
    (same keys), with market_category='rain'.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    opportunities: list[dict] = []

    for market in markets:
        ticker = str(market.get("ticker") or "")
        if not ticker.upper().startswith("KXRAIN"):
            continue
        parsed = parse_rain_outcome(
            market.get("outcome") or market.get("yes_sub_title") or market.get("title"),
            ticker=ticker,
        )
        if parsed is None:
            continue
        city = market.get("city")
        if not city or city not in precip_forecasts:
            continue
        market_date = market.get("market_date")
        if not market_date:
            continue

        daily = precip_forecasts[city].get("daily") or []
        forecast_row = next((d for d in daily if d.get("date") == market_date), None)
        if forecast_row is None:
            continue

        raw_prob = forecast_row.get("forecast_prob_any_rain")
        if raw_prob is None:
            continue

        forecast_blend_source = "open-meteo"
        prob_for_probability_calibration = raw_prob

        # HRRR same-day blend (within configured horizon, default 12 h)
        close_time_raw = market.get("close_time")
        hours_to_settlement = None
        if close_time_raw:
            try:
                close_dt = datetime.fromisoformat(str(close_time_raw).replace("Z", "+00:00"))
                hours_to_settlement = (close_dt - now_utc).total_seconds() / 3600.0
            except ValueError:
                hours_to_settlement = None
        if (
            hrrr_data
            and city in hrrr_data
            and hrrr_data[city] is not None  # None sentinel from fetch_hrrr_precip_multi
            and hours_to_settlement is not None
            and 0 <= hours_to_settlement <= hrrr_blend_horizon_hours
        ):
            hrrr_day = hrrr_data[city].get(market_date)
            if hrrr_day and hrrr_day.get("total_in") is not None:
                hrrr_wet = 1.0 if hrrr_day["total_in"] >= 0.01 else 0.0
                # Ramp: weight HRRR more as settlement nears, cap at 0.7
                w_hrrr = min(0.7, max(0.0, 1.0 - hours_to_settlement / hrrr_blend_horizon_hours))
                prob_for_probability_calibration = (
                    (1.0 - w_hrrr) * raw_prob + w_hrrr * hrrr_wet
                )
                forecast_blend_source = "open-meteo+hrrr"

        # Calibration (logistic bias + isotonic); fall through to raw if unavailable
        calibrated_prob = prob_for_probability_calibration
        forecast_calibration_source = "raw"
        probability_calibration_source = "raw"
        if calibration_manager is not None:
            cal = calibration_manager.calibrate_rain_probability(
                city=city,
                raw_prob=prob_for_probability_calibration,
            )
            if cal is not None:
                calibrated_prob = cal["calibrated_prob"]
                forecast_calibration_source = cal.get("forecast_calibration_source", "logistic")
                probability_calibration_source = cal.get("probability_calibration_source", "isotonic")

        # For P1 we only trade YES on the binary market (mirrors temp v4 BUY-only)
        position_side = "yes"
        our_prob = compute_rain_yes_probability(calibrated_prob, position_side=position_side)

        market_price_raw = market.get("yes_ask")
        if market_price_raw is None:
            market_price_raw = market.get("market_price")
        try:
            market_price = float(market_price_raw) if market_price_raw is not None else None
        except (TypeError, ValueError):
            market_price = None
        # If no ask is posted we can't compute edge -- skip, but log at debug
        if market_price is None:
            logger.debug("No market_price/yes_ask for %s; skipping", ticker)
            continue

        edge = our_prob - market_price
        abs_edge = abs(edge)
        if abs_edge < min_edge:
            continue

        yes_outcome_true = True  # YES resolves if it rains; we compute our_prob for that
        opportunities.append({
            "source": "kalshi",
            "ticker": ticker,
            "city": city,
            "market_type": "rain_binary",
            "market_category": "rain",
            "market_date": market_date,
            "outcome": market.get("outcome") or market.get("yes_sub_title"),
            "position_side": position_side,
            "our_probability": our_prob,
            "raw_probability": raw_prob,
            "market_price": market_price,
            "edge": edge,
            "abs_edge": abs_edge,
            "forecast_value_f": None,
            "forecast_blend_source": forecast_blend_source,
            "forecast_calibration_source": forecast_calibration_source,
            "probability_calibration_source": probability_calibration_source,
            "hours_to_settlement": hours_to_settlement,
            "volume24hr": market.get("volume24hr") or market.get("volume_24h"),
            "yes_outcome": yes_outcome_true,
        })
    return opportunities
