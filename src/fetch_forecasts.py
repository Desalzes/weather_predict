"""
Fetches weather forecast data from Open-Meteo (free, no API key required).
"""

import logging
from datetime import date, datetime
from typing import Optional

import requests

logger = logging.getLogger("weather.forecasts")

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

_HOURLY_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "precipitation_probability",
    "cloudcover",
    "windspeed_10m",
    "weathercode",
    "relative_humidity_2m",
    "snowfall",
    "rain",
]

_ENSEMBLE_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
]


def _location_name(location: dict) -> str:
    return location.get("name", f"{location.get('lat')},{location.get('lon')}")


def _normalize_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def fetch_hourly_forecast(
    latitude: float,
    longitude: float,
    hours: int = 72,
) -> Optional[dict]:
    """Fetch hourly weather forecast from the Open-Meteo API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(_HOURLY_VARIABLES),
        "forecast_hours": hours,
        "models": "best_match",
        "timezone": "auto",
    }

    try:
        response = requests.get(_FORECAST_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching hourly forecast for (%.4f, %.4f)", latitude, longitude)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("Request error fetching hourly forecast for (%.4f, %.4f): %s", latitude, longitude, exc)
        return None
    except ValueError as exc:
        logger.warning("JSON decode error for hourly forecast (%.4f, %.4f): %s", latitude, longitude, exc)
        return None

    return {
        "location": {"lat": latitude, "lon": longitude},
        "hourly": data.get("hourly", {}),
        "units": data.get("hourly_units", {}),
        "timezone": data.get("timezone"),
        "timezone_abbreviation": data.get("timezone_abbreviation"),
    }


def fetch_ensemble_forecast(
    latitude: float,
    longitude: float,
    hours: int = 72,
) -> Optional[dict]:
    """Fetch ensemble forecast from Open-Meteo's ensemble API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(_ENSEMBLE_VARIABLES),
        "forecast_hours": hours,
        "timezone": "auto",
    }

    try:
        response = requests.get(_ENSEMBLE_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching ensemble forecast for (%.4f, %.4f)", latitude, longitude)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning("Request error fetching ensemble forecast for (%.4f, %.4f): %s", latitude, longitude, exc)
        return None
    except ValueError as exc:
        logger.warning("JSON decode error for ensemble forecast (%.4f, %.4f): %s", latitude, longitude, exc)
        return None

    return data


def fetch_ensemble_multi_location(
    locations: list,
    hours: int = 72,
) -> dict:
    """Fetch ensemble forecasts for multiple named locations in one request."""
    if not locations:
        return {}

    params = {
        "latitude": ",".join(str(loc["lat"]) for loc in locations),
        "longitude": ",".join(str(loc["lon"]) for loc in locations),
        "hourly": ",".join(_ENSEMBLE_VARIABLES),
        "forecast_hours": hours,
        "timezone": "auto",
    }

    try:
        response = requests.get(_ENSEMBLE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching ensemble forecast batch for %d locations", len(locations))
        return {_location_name(loc): None for loc in locations}
    except requests.exceptions.RequestException as exc:
        logger.warning("Request error fetching ensemble forecast batch: %s", exc)
        return {_location_name(loc): None for loc in locations}
    except ValueError as exc:
        logger.warning("JSON decode error for ensemble forecast batch: %s", exc)
        return {_location_name(loc): None for loc in locations}

    payloads = data if isinstance(data, list) else [data]
    results = {}

    for idx, loc in enumerate(locations):
        name = _location_name(loc)
        if idx >= len(payloads):
            logger.warning("Missing ensemble payload for %s in batch response", name)
            results[name] = None
            continue

        payload = payloads[idx]
        if not isinstance(payload, dict):
            logger.warning(
                "Unexpected ensemble payload type for %s: %s",
                name,
                type(payload).__name__,
            )
            results[name] = None
            continue

        results[name] = {
            "location": {"name": name, "lat": loc["lat"], "lon": loc["lon"]},
            "hourly": payload.get("hourly", {}),
            "units": payload.get("hourly_units", {}),
            "timezone": payload.get("timezone"),
            "timezone_abbreviation": payload.get("timezone_abbreviation"),
        }

    return results


def fetch_multi_location(
    locations: list,
    hours: int = 72,
) -> dict:
    """Fetch hourly forecasts for multiple named locations."""
    results = {}
    for loc in locations:
        name = _location_name(loc)
        lat = loc["lat"]
        lon = loc["lon"]
        logger.debug("Fetching forecast for %s (%.4f, %.4f)", name, lat, lon)
        results[name] = fetch_hourly_forecast(lat, lon, hours=hours)
    return results


def fetch_previous_run_forecast(
    latitude: float,
    longitude: float,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    *,
    lead_days: int = 1,
    model: str = "best_match",
) -> Optional[dict]:
    """Fetch archived previous-day forecasts for a date range.

    Returns both:
    - Hourly temperature at lead N (``temperature_2m_previous_day{N}``)
    - Hourly precipitation at lead N (``precipitation_previous_day{N}``),
      aggregated to per-day sums by this function
    - Daily max precipitation probability (``precipitation_probability_max``,
      plain daily var — the Previous Runs API does not support a
      ``*_previous_day{N}`` suffix on daily variables, so this is the
      historical forecast's probability output for each date in the range)

    The daily precipitation block is normalized into ``result["daily"]`` with
    ``precipitation_sum_in`` (mm-total aggregated from hourly, converted to
    inches) and ``precipitation_probability_max`` (kept as percent, 0-100).
    """
    if lead_days < 1:
        raise ValueError("lead_days must be >= 1")

    lead = int(lead_days)
    hourly_temp = f"temperature_2m_previous_day{lead}"
    hourly_precip = f"precipitation_previous_day{lead}"
    daily_precip_prob = "precipitation_probability_max"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join([hourly_temp, hourly_precip]),
        "daily": daily_precip_prob,
        "start_date": _normalize_date(start_date),
        "end_date": _normalize_date(end_date),
        "models": model,
        "timezone": "auto",
    }

    try:
        response = requests.get(_PREVIOUS_RUNS_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching previous-run forecast for (%.4f, %.4f)", latitude, longitude)
        return None
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "Request error fetching previous-run forecast for (%.4f, %.4f): %s",
            latitude,
            longitude,
            exc,
        )
        return None
    except ValueError as exc:
        logger.warning(
            "JSON decode error for previous-run forecast (%.4f, %.4f): %s",
            latitude,
            longitude,
            exc,
        )
        return None

    daily_raw = data.get("daily", {}) or {}
    daily_times = list(daily_raw.get("time", []) or [])
    precip_prob = list(daily_raw.get(daily_precip_prob, []) or [])

    hourly_raw = data.get("hourly", {}) or {}
    hourly_times = list(hourly_raw.get("time", []) or [])
    hourly_precip = list(hourly_raw.get(hourly_precip, []) or [])

    precip_sum_by_date: dict[str, float] = {}
    for i, t in enumerate(hourly_times):
        if i >= len(hourly_precip):
            break
        value = hourly_precip[i]
        if value is None:
            continue
        date_key = str(t)[:10]
        precip_sum_by_date[date_key] = precip_sum_by_date.get(date_key, 0.0) + float(value)

    precip_in = [
        precip_sum_by_date[d] / 25.4 if d in precip_sum_by_date else None
        for d in daily_times
    ]

    return {
        "location": {"lat": latitude, "lon": longitude},
        "hourly": data.get("hourly", {}),
        "units": data.get("hourly_units", {}),
        "timezone": data.get("timezone"),
        "timezone_abbreviation": data.get("timezone_abbreviation"),
        "model": model,
        "lead_days": lead,
        "daily": {
            "time": daily_times,
            "precipitation_sum_in": precip_in,
            # percent 0-100; archive layer converts to fraction
            "precipitation_probability_max": precip_prob,
        },
    }
