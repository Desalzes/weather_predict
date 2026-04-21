"""
Matches weather forecasts against Kalshi + Polymarket temperature markets.

Core logic: model forecast temp as normal distribution, compute probability
of landing in each market bucket, compare against market prices to find edge.
"""

import logging
import math
import re
import statistics
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.calibration import CalibrationManager

logger = logging.getLogger("weather.matcher")

_ENSEMBLE_SIGMA_FLOOR_F = 1.0
_ENSEMBLE_SIGMA_CAP_F = 6.0


def configure_sigma_bounds(
    floor_f: float | None = None,
    cap_f: float | None = None,
) -> None:
    """Override module-level sigma bounds from config at startup."""
    global _ENSEMBLE_SIGMA_FLOOR_F, _ENSEMBLE_SIGMA_CAP_F
    if floor_f is not None:
        _ENSEMBLE_SIGMA_FLOOR_F = float(floor_f)
    if cap_f is not None:
        _ENSEMBLE_SIGMA_CAP_F = float(cap_f)


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard normal CDF using math.erfc — no scipy needed."""
    sigma = max(float(sigma), 1e-6)
    z = (x - mu) / sigma
    return 0.5 * math.erfc(-z / math.sqrt(2))


def c_to_f(celsius: float) -> float:
    return celsius * 9 / 5 + 32


def f_to_c(fahrenheit: float) -> float:
    return (fahrenheit - 32) * 5 / 9


def compute_temperature_probability(
    forecast_value_f: float,
    outcome_low: Optional[float],
    outcome_high: Optional[float],
    uncertainty_std_f: float = 2.0,
) -> float:
    """Compute P(actual temp falls in [low, high]) given forecast as normal dist."""
    if outcome_low is not None and outcome_high is not None:
        p = _normal_cdf(outcome_high, forecast_value_f, uncertainty_std_f) - \
            _normal_cdf(outcome_low, forecast_value_f, uncertainty_std_f)
    elif outcome_low is not None:
        p = 1.0 - _normal_cdf(outcome_low, forecast_value_f, uncertainty_std_f)
    elif outcome_high is not None:
        p = _normal_cdf(outcome_high, forecast_value_f, uncertainty_std_f)
    else:
        return 0.0
    return max(0.0, min(1.0, p))


def _parse_market_date(ticker: str) -> Optional[str]:
    """Extract the calendar date from a Kalshi ticker.

    KXHIGHTDAL-26MAR28-T65 -> '2026-03-28'
    """
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{1,2})-", ticker.upper())
    if not m:
        return None
    year_short = int(m.group(1))
    month_str = m.group(2)
    day = int(m.group(3))
    months = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    month = months.get(month_str)
    if not month:
        return None
    year = 2000 + year_short
    return f"{year}-{month:02d}-{day:02d}"


def _kalshi_threshold_yes_probability(
    direction: str,
    threshold: float,
    forecast_value_f: float,
    uncertainty_std_f: float,
) -> float:
    """Compute YES probability from the live threshold title direction."""
    if direction == "below":
        return _normal_cdf(threshold, forecast_value_f, uncertainty_std_f)
    if direction == "above":
        return 1.0 - _normal_cdf(threshold + 0.99, forecast_value_f, uncertainty_std_f)
    return 0.0


def _kalshi_threshold_direction(market: dict) -> Optional[str]:
    title = str(market.get("title", "")).lower()
    yes_sub_title = str(market.get("yes_sub_title", "")).lower()

    if "<" in title or "or below" in yes_sub_title:
        return "below"
    if ">" in title or "or above" in yes_sub_title:
        return "above"
    return None


def _kalshi_threshold_outcome_label(direction: str, threshold: float) -> str:
    if direction == "below":
        return f"<{threshold:.0f}°F"
    if direction == "above":
        return f">{threshold:.0f}°F"
    return f"{threshold:.0f}°F"


def _is_bucket_market(ticker: str) -> bool:
    """Return True if the ticker's last segment starts with B (bucket/range)."""
    last_seg = ticker.split("-")[-1]
    return last_seg.upper().startswith("B")


def _settlement_rule_from_bounds(
    low: Optional[float],
    high: Optional[float],
) -> str:
    if low is not None and high is not None:
        return "between_inclusive"
    if low is not None:
        return "gte"
    if high is not None:
        return "lte"
    raise ValueError("Settlement bounds require at least one bound")


def _kalshi_bucket_bounds_from_threshold(bucket_threshold: float) -> tuple[float, float]:
    """Return continuous bounds for a Kalshi bucket threshold like ``66.5``."""
    threshold = float(bucket_threshold)
    return threshold - 1.0, threshold + 1.0


def _should_apply_probability_calibration(
    forecast_blend_source: str,
    hours_to_settlement: Optional[float],
    blend_horizon_hours: float,
) -> bool:
    """Apply isotonic calibration only in the day-ahead regime it was trained on."""
    if forecast_blend_source != "open-meteo":
        return False
    if hours_to_settlement is None:
        return True
    return float(hours_to_settlement) >= float(blend_horizon_hours)


def _extract_calendar_day_temps(
    forecast: dict, target_date: str
) -> list[float]:
    """Extract only the hourly temps (Celsius) for a specific calendar date."""
    return _extract_calendar_day_series(
        forecast.get("hourly", {}),
        "temperature_2m",
        target_date,
    )


def _extract_calendar_day_series(
    hourly_data: dict,
    field_name: str,
    target_date: str,
) -> list[float]:
    """Extract one hourly series for a specific calendar date."""
    if not target_date:
        return []

    times = hourly_data.get("time", [])
    values = hourly_data.get(field_name, [])

    day_temps = []
    for i, t in enumerate(times):
        if i >= len(values):
            break
        if t.startswith(target_date):
            day_temps.append(values[i])

    return day_temps


def _ensemble_sigma_for_date(
    ensemble_forecast: Optional[dict],
    target_date: str,
    market_type: str,
) -> Optional[float]:
    """Compute sigma from ensemble-member daily highs/lows for one date."""
    if not ensemble_forecast or not target_date or market_type not in {"high", "low"}:
        return None

    hourly = ensemble_forecast.get("hourly", {})
    member_keys = sorted(
        key for key in hourly
        if key.startswith("temperature_2m_member")
    )
    if len(member_keys) < 2:
        return None

    member_extremes_f = []
    for key in member_keys:
        member_temps_c = _extract_calendar_day_series(hourly, key, target_date)
        if not member_temps_c:
            continue

        member_temps_f = [c_to_f(temp_c) for temp_c in member_temps_c]
        member_extremes_f.append(
            max(member_temps_f) if market_type == "high" else min(member_temps_f)
        )

    if len(member_extremes_f) < 2:
        return None

    sigma_f = statistics.pstdev(member_extremes_f)
    return max(_ENSEMBLE_SIGMA_FLOOR_F, min(_ENSEMBLE_SIGMA_CAP_F, sigma_f))


def _resolve_temperature_uncertainty(
    ensemble_data: Optional[dict],
    city: str,
    target_date: Optional[str],
    market_type: str,
    fallback_sigma_f: float,
) -> tuple[float, str]:
    if ensemble_data and target_date:
        sigma_f = _ensemble_sigma_for_date(
            ensemble_data.get(city),
            target_date,
            market_type,
        )
        if sigma_f is not None:
            return sigma_f, "ensemble"
    return fallback_sigma_f, "config"


def _apply_calibration(
    calibration_manager: Optional["CalibrationManager"],
    city: str,
    market_type: str,
    forecast_value_f: float,
    sigma_f: float,
 ) -> tuple[float, str]:
    corrected_forecast_f = float(forecast_value_f)
    forecast_calibration_source = "raw"

    if calibration_manager is not None:
        corrected_forecast_f, forecast_calibration_source = calibration_manager.correct_forecast(
            city,
            market_type,
            forecast_value_f,
            sigma_f,
        )

    return corrected_forecast_f, forecast_calibration_source


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _doy_from_date(target_date: Optional[str], fallback_now: Optional[datetime] = None) -> int:
    """Return day-of-year (1–366) for an ISO date string.

    On parse failure, falls back to today's DOY (seasonally current) rather
    than 1 so that NGR's sin/cos(doy) features aren't silently biased to
    Jan 1 when a date string is malformed.
    """
    if target_date:
        try:
            return datetime.fromisoformat(target_date).timetuple().tm_yday
        except ValueError:
            logger.debug("Could not parse DOY from %r, falling back to today", target_date)
    now = fallback_now or datetime.now(timezone.utc)
    return now.timetuple().tm_yday


def _lead_hours(close_time: str, now_utc: Optional[datetime]) -> float:
    """Return hours from now_utc to close_time; floor at 0.0; default 24.0 on parse failure."""
    close_dt = _parse_iso_datetime(close_time)
    if close_dt is None:
        return 24.0
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (close_dt.astimezone(timezone.utc) - current.astimezone(timezone.utc)).total_seconds() / 3600.0)


def _apply_hrrr_blend(
    hrrr_data: Optional[dict],
    city: str,
    target_date: Optional[str],
    market_type: str,
    close_time: str,
    open_meteo_forecast_f: float,
    timezone_name: Optional[str],
    blend_horizon_hours: float,
    now_utc: Optional[datetime],
) -> tuple[float, str, Optional[float], float, Optional[float]]:
    """Blend Open-Meteo with HRRR when the market is near settlement."""
    if not hrrr_data or city not in hrrr_data or market_type not in {"high", "low"}:
        return float(open_meteo_forecast_f), "open-meteo", None, 0.0, None
    if not target_date or blend_horizon_hours <= 0:
        return float(open_meteo_forecast_f), "open-meteo", None, 0.0, None

    close_dt = _parse_iso_datetime(close_time)
    if close_dt is None:
        return float(open_meteo_forecast_f), "open-meteo", None, 0.0, None

    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)

    hours_to_settlement = (close_dt.astimezone(timezone.utc) - current).total_seconds() / 3600.0
    if hours_to_settlement < 0 or hours_to_settlement > blend_horizon_hours:
        return float(open_meteo_forecast_f), "open-meteo", None, 0.0, hours_to_settlement

    try:
        from src.fetch_hrrr import get_hrrr_high_low
    except Exception:
        return float(open_meteo_forecast_f), "open-meteo", None, 0.0, hours_to_settlement

    hrrr_high_f, hrrr_low_f = get_hrrr_high_low(hrrr_data.get(city), target_date, timezone_name)
    hrrr_forecast_f = hrrr_high_f if market_type == "high" else hrrr_low_f
    if hrrr_forecast_f is None:
        return float(open_meteo_forecast_f), "open-meteo", None, 0.0, hours_to_settlement

    hrrr_weight = max(0.3, 1.0 - hours_to_settlement / blend_horizon_hours)
    hrrr_weight = max(0.0, min(1.0, hrrr_weight))
    blended_forecast_f = (1.0 - hrrr_weight) * open_meteo_forecast_f + hrrr_weight * float(hrrr_forecast_f)

    return blended_forecast_f, "open-meteo+hrrr", float(hrrr_forecast_f), hrrr_weight, hours_to_settlement


def match_kalshi_markets(
    forecasts: dict,
    kalshi_markets: list[dict],
    min_edge: float = 0.05,
    uncertainty_std_f: float = 2.0,
    ensemble_data: Optional[dict] = None,
    calibration_manager: Optional["CalibrationManager"] = None,
    hrrr_data: Optional[dict] = None,
    hrrr_blend_horizon_hours: float = 18.0,
    now_utc: Optional[datetime] = None,
    use_ngr_calibration: bool = False,
) -> list[dict]:
    """Match forecasts against Kalshi weather markets."""
    opps = []
    for m in kalshi_markets:
        city = m.get("city")
        mtype = m.get("type")
        threshold = m.get("threshold")
        ticker = m.get("ticker", "")
        if not city or not mtype or threshold is None:
            continue

        forecast = forecasts.get(city)
        if not forecast:
            continue

        market_date = _parse_market_date(ticker)
        if market_date:
            temps_c = _extract_calendar_day_temps(forecast, market_date)
        else:
            temps_c = []

        if not temps_c:
            logger.debug(
                "No forecast data for %s on %s, skipping %s",
                city, market_date, ticker,
            )
            continue

        temps_f = [c_to_f(t) for t in temps_c]
        is_bucket = _is_bucket_market(ticker)

        if mtype == "high":
            open_meteo_forecast_value = max(temps_f)
        elif mtype == "low":
            open_meteo_forecast_value = min(temps_f)
        else:
            continue

        ensemble_sigma_f, sigma_source = _resolve_temperature_uncertainty(
            ensemble_data,
            city,
            market_date,
            mtype,
            uncertainty_std_f,
        )

        raw_forecast_value, forecast_blend_source, hrrr_forecast_value, hrrr_weight, hours_to_settlement = _apply_hrrr_blend(
            hrrr_data,
            city,
            market_date,
            mtype,
            str(m.get("close_time", "")),
            open_meteo_forecast_value,
            forecast.get("timezone"),
            hrrr_blend_horizon_hours,
            now_utc,
        )

        if use_ngr_calibration and calibration_manager is not None and hasattr(calibration_manager, "predict_distribution"):
            lead_h = _lead_hours(str(m.get("close_time", "")), now_utc)
            doy = _doy_from_date(market_date)
            forecast_value, sigma_f, forecast_calibration_source = calibration_manager.predict_distribution(
                city, mtype, raw_forecast_value, ensemble_sigma_f, lead_h, doy,
            )
        else:
            sigma_f = ensemble_sigma_f
            forecast_value, forecast_calibration_source = _apply_calibration(
                calibration_manager,
                city,
                mtype,
                raw_forecast_value,
                sigma_f,
            )

        if is_bucket:
            bucket_low, bucket_high = _kalshi_bucket_bounds_from_threshold(threshold)
            raw_prob = compute_temperature_probability(
                forecast_value,
                bucket_low,
                bucket_high,
                sigma_f,
            )
            our_prob = compute_temperature_probability(
                forecast_value,
                bucket_low,
                bucket_high,
                sigma_f,
            )
            outcome_label = f"{math.floor(threshold):.0f}-{math.ceil(threshold):.0f}\u00b0F"
            settlement_rule = "between_inclusive"
            settlement_low_f = float(math.floor(threshold))
            settlement_high_f = float(math.ceil(threshold))
        else:
            threshold_direction = _kalshi_threshold_direction(m)
            if threshold_direction is None:
                logger.debug("Could not infer Kalshi threshold direction for %s", ticker)
                continue
            raw_prob = _kalshi_threshold_yes_probability(
                threshold_direction,
                threshold,
                forecast_value,
                sigma_f,
            )
            our_prob = raw_prob
            outcome_label = _kalshi_threshold_outcome_label(threshold_direction, threshold)
            settlement_rule = "lte" if threshold_direction == "below" else "gt"
            settlement_low_f = float(threshold) if threshold_direction == "above" else None
            settlement_high_f = float(threshold) if threshold_direction == "below" else None

        probability_calibration_source = "raw"
        if calibration_manager is not None and _should_apply_probability_calibration(
            forecast_blend_source,
            hours_to_settlement,
            hrrr_blend_horizon_hours,
        ):
            our_prob, probability_calibration_source = calibration_manager.calibrate_probability(
                city,
                mtype,
                our_prob,
            )

        our_prob = max(0.001, min(0.999, our_prob))
        market_price = m.get("last_price", 0)
        if market_price <= 0:
            market_price = (m.get("yes_bid", 0) + m.get("yes_ask", 0)) / 2
        if market_price <= 0:
            continue

        edge = our_prob - market_price
        hours_available = len(temps_c)

        opps.append({
            "source": "kalshi",
            "ticker": ticker,
            "market_question": m.get("title", ""),
            "outcome": outcome_label,
            "market_date": market_date,
            "is_bucket": is_bucket,
            "our_probability": round(our_prob, 4),
            "market_price": round(market_price, 4),
            "edge": round(edge, 4),
            "abs_edge": round(abs(edge), 4),
            "direction": "BUY" if edge > 0 else "SELL",
            "forecast_value_f": round(forecast_value, 1),
            "open_meteo_forecast_value_f": round(open_meteo_forecast_value, 1),
            "raw_forecast_value_f": round(raw_forecast_value, 1),
            "forecast_blend_source": forecast_blend_source,
            "hrrr_forecast_value_f": round(hrrr_forecast_value, 1) if hrrr_forecast_value is not None else None,
            "hrrr_weight": round(hrrr_weight, 3),
            "hours_to_settlement": round(hours_to_settlement, 2) if hours_to_settlement is not None else None,
            "uncertainty_std_f": round(sigma_f, 2),
            "uncertainty_source": sigma_source,
            "raw_probability": round(raw_prob, 4),
            "forecast_calibration_source": forecast_calibration_source,
            "probability_calibration_source": probability_calibration_source,
            "forecast_hours": hours_available,
            "volume24hr": m.get("volume_24h", 0),
            "city": city,
            "market_type": mtype,
            "actual_field": "tmax_f" if mtype == "high" else "tmin_f",
            "settlement_rule": settlement_rule,
            "settlement_low_f": settlement_low_f,
            "settlement_high_f": settlement_high_f,
            "yes_bid": m.get("yes_bid", 0),
            "yes_ask": m.get("yes_ask", 0),
        })

    filtered = [o for o in opps if o["abs_edge"] >= min_edge]
    filtered.sort(key=lambda x: x["abs_edge"], reverse=True)
    logger.info(
        "Kalshi: %d opportunities (edge >= %.0f%%) from %d markets",
        len(filtered), min_edge * 100, len(kalshi_markets),
    )
    return filtered


def match_polymarket_markets(
    forecasts: dict,
    poly_markets: list[dict],
    min_edge: float = 0.05,
    uncertainty_std_f: float = 2.0,
    ensemble_data: Optional[dict] = None,
    calibration_manager: Optional["CalibrationManager"] = None,
    hrrr_data: Optional[dict] = None,
    hrrr_blend_horizon_hours: float = 18.0,
    now_utc: Optional[datetime] = None,
    use_ngr_calibration: bool = False,
) -> list[dict]:
    """Match forecasts against Polymarket weather markets."""
    from src.fetch_polymarket import _parse_outcome_range

    opps = []
    for market in poly_markets:
        city = market.get("city")
        mtype = market.get("market_type", "")

        if not city or city not in forecasts:
            matched = False
            for loc_name in forecasts:
                if city and city.lower() in loc_name.lower():
                    city = loc_name
                    matched = True
                    break
            if not matched:
                continue

        forecast = forecasts.get(city)
        if not forecast:
            continue

        from datetime import datetime, timezone
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        temps_c = _extract_calendar_day_temps(forecast, today_str)
        if not temps_c:
            temps_c = forecast.get("hourly", {}).get("temperature_2m", [])[:24]
        if not temps_c:
            continue

        temps_f = [c_to_f(t) for t in temps_c]

        if mtype == "daily_high":
            open_meteo_forecast_value = max(temps_f)
            sigma_market_type = "high"
        elif mtype == "daily_low":
            open_meteo_forecast_value = min(temps_f)
            sigma_market_type = "low"
        else:
            continue

        ensemble_sigma_f, sigma_source = _resolve_temperature_uncertainty(
            ensemble_data,
            city,
            today_str,
            sigma_market_type,
            uncertainty_std_f,
        )

        raw_forecast_value, forecast_blend_source, hrrr_forecast_value, hrrr_weight, hours_to_settlement = _apply_hrrr_blend(
            hrrr_data,
            city,
            today_str,
            sigma_market_type,
            str(market.get("endDate", "")),
            open_meteo_forecast_value,
            forecast.get("timezone"),
            hrrr_blend_horizon_hours,
            now_utc,
        )

        if use_ngr_calibration and calibration_manager is not None and hasattr(calibration_manager, "predict_distribution"):
            lead_h = _lead_hours(str(market.get("endDate", "")), now_utc)
            doy = _doy_from_date(today_str)
            poly_forecast_value, sigma_f, poly_forecast_calibration_source = calibration_manager.predict_distribution(
                city, sigma_market_type, raw_forecast_value, ensemble_sigma_f, lead_h, doy,
            )
        else:
            sigma_f = ensemble_sigma_f
            poly_forecast_value = None
            poly_forecast_calibration_source = None

        outcomes = market.get("outcomes", [])
        prices = market.get("outcomePrices", [])
        if not outcomes or not prices or len(outcomes) != len(prices):
            continue

        end_dt = _parse_iso_datetime(str(market.get("endDate", "")))
        market_date = end_dt.date().isoformat() if end_dt is not None else today_str

        for outcome_str, market_price in zip(outcomes, prices):
            low, high = _parse_outcome_range(outcome_str)
            if low is None and high is None:
                continue
            settlement_rule = _settlement_rule_from_bounds(low, high)

            if poly_forecast_value is not None:
                forecast_value = poly_forecast_value
                forecast_calibration_source = poly_forecast_calibration_source
            else:
                forecast_value, forecast_calibration_source = _apply_calibration(
                    calibration_manager,
                    city,
                    sigma_market_type,
                    raw_forecast_value,
                    sigma_f,
                )
            raw_prob = compute_temperature_probability(forecast_value, low, high, sigma_f)
            our_prob = raw_prob
            probability_calibration_source = "raw"
            if calibration_manager is not None and _should_apply_probability_calibration(
                forecast_blend_source,
                hours_to_settlement,
                hrrr_blend_horizon_hours,
            ):
                our_prob, probability_calibration_source = calibration_manager.calibrate_probability(
                    city,
                    sigma_market_type,
                    our_prob,
                )
            edge = our_prob - market_price

            opps.append({
                "source": "polymarket",
                "ticker": market.get("slug", ""),
                "market_question": market["question"],
                "outcome": outcome_str,
                "market_date": market_date,
                "our_probability": round(our_prob, 4),
                "market_price": round(market_price, 4),
                "edge": round(edge, 4),
                "abs_edge": round(abs(edge), 4),
                "direction": "BUY" if edge > 0 else "SELL",
                "forecast_value_f": round(forecast_value, 1),
                "open_meteo_forecast_value_f": round(open_meteo_forecast_value, 1),
                "raw_forecast_value_f": round(raw_forecast_value, 1),
                "forecast_blend_source": forecast_blend_source,
                "hrrr_forecast_value_f": round(hrrr_forecast_value, 1) if hrrr_forecast_value is not None else None,
                "hrrr_weight": round(hrrr_weight, 3),
                "hours_to_settlement": round(hours_to_settlement, 2) if hours_to_settlement is not None else None,
                "uncertainty_std_f": round(sigma_f, 2),
                "uncertainty_source": sigma_source,
                "raw_probability": round(raw_prob, 4),
                "forecast_calibration_source": forecast_calibration_source,
                "probability_calibration_source": probability_calibration_source,
                "forecast_hours": len(temps_c),
                "volume24hr": market.get("volume24hr", 0),
                "city": city,
                "market_type": mtype,
                "actual_field": "tmax_f" if sigma_market_type == "high" else "tmin_f",
                "settlement_rule": settlement_rule,
                "settlement_low_f": low,
                "settlement_high_f": high,
            })

    filtered = [o for o in opps if o["abs_edge"] >= min_edge]
    filtered.sort(key=lambda x: x["abs_edge"], reverse=True)
    logger.info(
        "Polymarket: %d opportunities (edge >= %.0f%%) from %d markets",
        len(filtered), min_edge * 100, len(poly_markets),
    )
    return filtered


def scan_all(
    forecasts: dict,
    kalshi_markets: list[dict] = None,
    poly_markets: list[dict] = None,
    min_edge: float = 0.05,
    uncertainty_std_f: float = 2.0,
    ensemble_data: Optional[dict] = None,
    calibration_manager: Optional["CalibrationManager"] = None,
    hrrr_data: Optional[dict] = None,
    hrrr_blend_horizon_hours: float = 18.0,
    now_utc: Optional[datetime] = None,
    use_ngr_calibration: bool = False,
) -> list[dict]:
    all_opps = []

    if kalshi_markets:
        all_opps.extend(
            match_kalshi_markets(
                forecasts,
                kalshi_markets,
                min_edge,
                uncertainty_std_f,
                ensemble_data=ensemble_data,
                calibration_manager=calibration_manager,
                hrrr_data=hrrr_data,
                hrrr_blend_horizon_hours=hrrr_blend_horizon_hours,
                now_utc=now_utc,
                use_ngr_calibration=use_ngr_calibration,
            )
        )

    if poly_markets:
        all_opps.extend(
            match_polymarket_markets(
                forecasts,
                poly_markets,
                min_edge,
                uncertainty_std_f,
                ensemble_data=ensemble_data,
                calibration_manager=calibration_manager,
                hrrr_data=hrrr_data,
                hrrr_blend_horizon_hours=hrrr_blend_horizon_hours,
                now_utc=now_utc,
                use_ngr_calibration=use_ngr_calibration,
            )
        )

    all_opps.sort(key=lambda x: x["abs_edge"], reverse=True)
    return all_opps


def format_report(opportunities: list[dict]) -> str:
    """Format opportunities as a readable table."""
    if not opportunities:
        return "No opportunities found."

    lines = []
    header = (
        f"{'Src':<10} {'City':<16} {'Type':<7} {'Outcome':<14} "
        f"{'Date':<11} {'Fcst':>6} {'Hrs':>4} {'Ours':>6} {'Mkt':>6} {'Edge':>7} {'Dir':>5} {'Vol':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for o in opportunities:
        src = o.get("source", "?")[:9]
        mdate = o.get("market_date", "today")[:10]
        mtype = o.get("market_type", "")
        if o.get("is_bucket"):
            mtype += "/B"
        hours = o.get("forecast_hours", "?")
        fv = o.get("forecast_value_f")
        fv_str = f"{fv:>5.1f}\u00b0" if isinstance(fv, (int, float)) else f"{'--':>6}"
        vol = o.get("volume24hr")
        try:
            vol_str = f"{int(vol):>10,}" if vol is not None else f"{'--':>10}"
        except (TypeError, ValueError):
            vol_str = f"{'--':>10}"
        outcome = str(o.get("outcome") or "")
        lines.append(
            f"{src:<10} {o.get('city', ''):<16} {mtype:<7} "
            f"{outcome:<14} {mdate:<11} {fv_str} "
            f"{hours:>4} {o['our_probability']:>5.0%} {o['market_price']:>5.0%} "
            f"{o['edge']:>+6.0%} {o['direction']:>5} {vol_str}"
        )

    return "\n".join(lines)
