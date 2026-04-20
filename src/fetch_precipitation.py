"""Open-Meteo precipitation fetcher, parallel to fetch_forecasts for temperature."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import requests

logger = logging.getLogger("weather.fetch_precipitation")

_DETERMINISTIC_URL = "https://api.open-meteo.com/v1/forecast"
_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"


def _mm_to_in(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / 25.4
    except (TypeError, ValueError):
        return None


def _percent_to_fraction(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return None


def fetch_precipitation_multi(
    locations: Iterable[dict],
    forecast_hours: int = 72,
    models: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Fetch deterministic precipitation for each location.

    Returns: {city_name: {"daily": [{"date", "forecast_prob_any_rain",
    "forecast_amount_in"}, ...]}}
    """
    results: dict[str, dict] = {}
    for loc in locations:
        name = loc.get("name")
        if not name:
            continue
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "daily": "precipitation_sum,precipitation_probability_max",
            "forecast_days": max(1, int(forecast_hours / 24)),
            "timezone": "UTC",
        }
        if models:
            params["models"] = ",".join(models)
        try:
            r = requests.get(_DETERMINISTIC_URL, params=params, timeout=20)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            logger.warning("Precip fetch failed for %s: %s", name, exc)
            continue
        daily = payload.get("daily", {})
        dates = daily.get("time", []) or []
        amounts = daily.get("precipitation_sum", []) or []
        probs = daily.get("precipitation_probability_max", []) or []
        daily_rows = []
        for i, d in enumerate(dates):
            daily_rows.append({
                "date": d,
                "forecast_prob_any_rain": _percent_to_fraction(probs[i] if i < len(probs) else None),
                "forecast_amount_in": _mm_to_in(amounts[i] if i < len(amounts) else None),
            })
        results[name] = {"daily": daily_rows}
    return results


def fetch_precipitation_ensemble_multi(
    locations: Iterable[dict],
    forecast_hours: int = 72,
) -> dict[str, dict]:
    """Fetch ensemble-member precipitation per location.

    Returns wet-day fraction and amount std across members for each date.
    """
    results: dict[str, dict] = {}
    for loc in locations:
        name = loc.get("name")
        if not name:
            continue
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "hourly": "precipitation",
            "forecast_days": max(1, int(forecast_hours / 24)),
            "timezone": "UTC",
            "models": "icon_seamless",
        }
        try:
            r = requests.get(_ENSEMBLE_URL, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            logger.warning("Ensemble precip fetch failed for %s: %s", name, exc)
            continue
        results[name] = _summarize_ensemble_precip(payload)
    return results


def _summarize_ensemble_precip(payload: dict) -> dict:
    """Compute per-date wet-fraction and amount std across ensemble members."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    # Each member comes back as a key like "precipitation_member01".
    # Open-Meteo also sometimes emits a bare "precipitation" series (the
    # deterministic/control run) alongside members; exclude it — counting
    # it as a member would inflate member_count and bias wet_fraction.
    member_keys = [k for k in hourly.keys() if k.startswith("precipitation_member")]
    if not times or not member_keys:
        return {"daily": []}

    # Bucket hourly rows by YYYY-MM-DD
    from collections import defaultdict
    date_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(times):
        date_to_indices[str(t)[:10]].append(i)

    daily_rows = []
    for d, indices in date_to_indices.items():
        member_totals = []
        for key in member_keys:
            series = hourly.get(key) or []
            total_mm = sum((series[i] or 0.0) for i in indices if i < len(series))
            member_totals.append(total_mm)
        if not member_totals:
            continue
        wet_count = sum(1 for m in member_totals if m >= (0.01 * 25.4))  # 0.01 in → mm
        wet_fraction = wet_count / len(member_totals)
        mean_mm = sum(member_totals) / len(member_totals)
        var_mm = sum((m - mean_mm) ** 2 for m in member_totals) / max(1, len(member_totals))
        std_mm = var_mm ** 0.5
        daily_rows.append({
            "date": d,
            "ensemble_wet_fraction": wet_fraction,
            "ensemble_amount_mean_in": _mm_to_in(mean_mm),
            "ensemble_amount_std_in": _mm_to_in(std_mm),
            "member_count": len(member_totals),
        })
    return {"daily": daily_rows}


def _extract_hrrr_apcp_series(location: dict, fxx: int, **kwargs) -> list[dict]:
    """Return [{valid_time, apcp_kg_m2}, ...] from the latest HRRR run.

    Uses Herbie the same way fetch_hrrr does, but requests the APCP variable.
    Mirrors the fetch_hrrr import-failure handling (raises RuntimeError if
    Herbie is unavailable so the caller can degrade gracefully).
    """
    try:
        from herbie import Herbie
    except ImportError as exc:
        raise RuntimeError("Herbie not installed — HRRR precip unavailable") from exc

    series: list[dict] = []
    for hour in range(0, fxx + 1):
        try:
            H = Herbie(model="hrrr", product="sfc", fxx=hour)
            ds = H.xarray(":APCP:surface")
            val_kg_m2 = float(ds.sel(
                latitude=location["lat"], longitude=location["lon"], method="nearest"
            ).values)
        except Exception as exc:
            logger.debug("HRRR APCP extract failed at f%02d: %s", hour, exc)
            continue
        series.append({
            "valid_time": str(ds.valid_time.values),
            "apcp_kg_m2": val_kg_m2,
        })
    return series


def fetch_hrrr_precip_multi(
    locations: Iterable[dict],
    fxx: int = 18,
    **kwargs,
) -> dict[str, dict]:
    """Fetch HRRR accumulated precipitation for each location.

    Returns: {city_name: {date: {"total_in": float}, ...}}

    HRRR APCP is reset at the start of each run, so "total accumulation" for
    a given date is the max APCP value observed on that date minus the first.
    One kg/m^2 of liquid water equals ~0.0394 in (1 mm = 0.0394 in).
    """
    results: dict[str, dict] = {}
    for loc in locations:
        name = loc.get("name")
        if not name:
            continue
        try:
            series = _extract_hrrr_apcp_series(loc, fxx=fxx, **kwargs)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("HRRR precip failed for %s: %s", name, exc)
            continue

        from collections import defaultdict
        by_date: dict[str, list[float]] = defaultdict(list)
        for row in series:
            date_str = str(row["valid_time"])[:10]
            by_date[date_str].append(row["apcp_kg_m2"])

        daily_totals: dict[str, dict] = {}
        for date_str, kg_values in by_date.items():
            if not kg_values:
                continue
            total_mm = max(kg_values) - min(kg_values)  # 1 kg/m^2 ≈ 1 mm
            daily_totals[date_str] = {"total_in": _mm_to_in(total_mm)}
        results[name] = daily_totals
    return results
