"""
Optional HRRR point forecast fetchers via Herbie.

The implementation is intentionally defensive:
- HRRR is only used when Herbie/cfgrib/xarray are installed
- failures return None / empty dict instead of breaking the scan
- data are fetched only for 2 m temperature at specific points
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("weather.hrrr")

_TMP_SEARCH = ":TMP:2 m"
_DEFAULT_AVAILABILITY_LAG_HOURS = 2
_DEFAULT_SAVE_DIR = Path(__file__).resolve().parents[1] / "data" / "hrrr_cache"
_DEFAULT_TIMEOUT_SECONDS = 300
_DEFAULT_MAX_RETRIES_PER_HOUR = 2
_DEFAULT_RETRY_DELAY_SECONDS = 5
_MULTI_CACHE: dict[str, object] = {
    "run_time": None,
    "fxx": -1,
    "locations": {},
    "results": {},
}


def _get_herbie_class():
    try:
        from herbie import Herbie
    except ImportError as exc:
        raise RuntimeError(
            "HRRR support requires herbie-data plus its xarray/cfgrib stack to be installed"
        ) from exc

    return Herbie


def _to_utc(value: Optional[datetime] = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _resolve_init_time(
    now: Optional[datetime] = None,
    availability_lag_hours: int = _DEFAULT_AVAILABILITY_LAG_HOURS,
) -> datetime:
    current = _to_utc(now).replace(minute=0, second=0, microsecond=0)
    return current - timedelta(hours=max(0, availability_lag_hours))


def _to_herbie_datetime(value: datetime) -> datetime:
    """Herbie compares against naive pandas timestamps, so pass naive UTC."""
    return _to_utc(value).replace(tzinfo=None)


def _configure_utf8_console() -> None:
    """Herbie's point-picker prints Unicode progress messages on Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                continue


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _temp_to_f(value: float, units: str = "") -> float:
    units_lower = (units or "").lower()
    temp = float(value)

    if "kelvin" in units_lower or units_lower == "k" or temp > 200:
        return (temp - 273.15) * 9.0 / 5.0 + 32.0
    if "celsius" in units_lower or units_lower in {"c", "degc"}:
        return temp * 9.0 / 5.0 + 32.0
    if "fahrenheit" in units_lower or units_lower in {"f", "degf"}:
        return temp

    # Herbie/cfgrib typically expose HRRR 2 m temperature in Kelvin.
    return (temp - 273.15) * 9.0 / 5.0 + 32.0


def _extract_dataset_values(dataset) -> tuple[np.ndarray, str]:
    if hasattr(dataset, "data_vars"):
        data_vars = list(dataset.data_vars.values())
        if not data_vars:
            raise ValueError("HRRR dataset did not contain any data variables")
        data_array = data_vars[0]
    else:
        data_array = dataset

    values = np.asarray(data_array.values).reshape(-1)
    if values.size == 0:
        raise ValueError("HRRR dataset variable was empty")

    return values.astype(float), str(data_array.attrs.get("units", ""))


def _normalize_locations(locations: list) -> list[dict]:
    normalized = []
    for loc in locations:
        normalized.append(
            {
                "name": loc.get("name", f"{loc.get('lat')},{loc.get('lon')}"),
                "lat": float(loc["lat"]),
                "lon": float(loc["lon"]),
            }
        )
    return normalized


def _location_signature(loc: dict) -> tuple[str, float, float]:
    return (str(loc["name"]), round(float(loc["lat"]), 4), round(float(loc["lon"]), 4))


def _get_cached_multi_result(run_time: datetime, normalized_locations: list[dict], fxx: int) -> Optional[dict]:
    if _MULTI_CACHE.get("run_time") != run_time.isoformat():
        return None
    if int(_MULTI_CACHE.get("fxx", -1)) < int(fxx):
        return None

    cached_locations = _MULTI_CACHE.get("locations", {})
    cached_results = _MULTI_CACHE.get("results", {})
    if not isinstance(cached_locations, dict) or not isinstance(cached_results, dict):
        return None

    requested_names = []
    for loc in normalized_locations:
        signature = _location_signature(loc)
        if cached_locations.get(loc["name"]) != signature:
            return None
        requested_names.append(loc["name"])

    return {name: cached_results.get(name) for name in requested_names}


def _store_cached_multi_result(run_time: datetime, normalized_locations: list[dict], fxx: int, results: dict) -> None:
    _MULTI_CACHE["run_time"] = run_time.isoformat()
    _MULTI_CACHE["fxx"] = int(fxx)
    _MULTI_CACHE["locations"] = {loc["name"]: _location_signature(loc) for loc in normalized_locations}
    _MULTI_CACHE["results"] = dict(results)


def _fetch_hrrr_point_values(
    Herbie,
    run_time_utc: datetime,
    points: pd.DataFrame,
    lead_hour: int,
) -> tuple[np.ndarray, str]:
    herbie_run_time = _to_herbie_datetime(run_time_utc)
    H = Herbie(
        herbie_run_time,
        model="hrrr",
        product="sfc",
        fxx=lead_hour,
        save_dir=_DEFAULT_SAVE_DIR,
        verbose=False,
    )
    ds = H.xarray(search=_TMP_SEARCH, remove_grib=False)
    matched = ds.herbie.pick_points(points, method="nearest")
    return _extract_dataset_values(matched)


def fetch_hrrr_point_forecast(
    lat: float,
    lon: float,
    fxx: int = 18,
    *,
    now: Optional[datetime] = None,
    init_time: Optional[datetime] = None,
) -> Optional[dict]:
    """Fetch latest HRRR 2 m temperature timeseries for a single point."""
    result = fetch_hrrr_multi(
        [{"name": "__point__", "lat": lat, "lon": lon}],
        fxx=fxx,
        now=now,
        init_time=init_time,
    )
    return result.get("__point__")


def fetch_hrrr_multi(
    locations: list,
    fxx: int = 18,
    *,
    now: Optional[datetime] = None,
    init_time: Optional[datetime] = None,
    timeout_seconds: Optional[int] = None,
    max_retries_per_hour: Optional[int] = None,
    retry_delay_seconds: Optional[float] = None,
) -> dict:
    """Fetch HRRR point forecasts for multiple named locations.

    Args:
        timeout_seconds: Overall timeout for the entire fetch. Default 300s.
        max_retries_per_hour: Retries per lead hour on transient failure. Default 2.
        retry_delay_seconds: Delay between retries. Default 5s.
    """
    import time as _time

    results = {}
    if not locations:
        return results

    timeout = int(timeout_seconds or _DEFAULT_TIMEOUT_SECONDS)
    retries = int(max_retries_per_hour if max_retries_per_hour is not None else _DEFAULT_MAX_RETRIES_PER_HOUR)
    retry_delay = float(retry_delay_seconds if retry_delay_seconds is not None else _DEFAULT_RETRY_DELAY_SECONDS)

    Herbie = _get_herbie_class()
    run_time = _to_utc(init_time) if init_time is not None else _resolve_init_time(now=now)
    normalized_locations = _normalize_locations(locations)
    cached = _get_cached_multi_result(run_time, normalized_locations, fxx)
    if cached is not None:
        logger.debug(
            "Using in-memory HRRR cache for %d location(s) through f%02d.",
            len(normalized_locations),
            int(fxx),
        )
        return cached

    points = pd.DataFrame(
        {
            "longitude": [loc["lon"] for loc in normalized_locations],
            "latitude": [loc["lat"] for loc in normalized_locations],
        }
    )
    _configure_utf8_console()
    _DEFAULT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for loc in normalized_locations:
        logger.debug("Fetching HRRR for %s (%.4f, %.4f)", loc["name"], loc["lat"], loc["lon"])
        results[loc["name"]] = {
            "times": [],
            "temperature_2m_f": [],
            "run_time": run_time.isoformat(),
            "location": {"lat": loc["lat"], "lon": loc["lon"]},
        }

    deadline = _time.monotonic() + timeout
    consecutive_failures = 0

    for lead_hour in range(0, max(0, int(fxx)) + 1):
        if _time.monotonic() > deadline:
            logger.warning(
                "HRRR fetch timed out after %ds at f%02d/%02d; returning partial results.",
                timeout,
                lead_hour,
                fxx,
            )
            break

        scalars = None
        units = ""
        for attempt in range(1 + retries):
            try:
                scalars, units = _fetch_hrrr_point_values(Herbie, run_time, points, lead_hour)
                consecutive_failures = 0
                break
            except RuntimeError:
                raise
            except Exception as exc:
                if attempt < retries and _time.monotonic() < deadline:
                    logger.debug("HRRR fetch retry %d/%d for f%02d: %s", attempt + 1, retries, lead_hour, exc)
                    _time.sleep(min(retry_delay, max(deadline - _time.monotonic(), 0)))
                else:
                    logger.debug("HRRR fetch failed for f%02d after %d attempt(s): %s", lead_hour, attempt + 1, exc)
                    consecutive_failures += 1

        if scalars is None:
            if consecutive_failures >= 3:
                logger.warning(
                    "HRRR: %d consecutive lead-hour failures; aborting remaining hours.",
                    consecutive_failures,
                )
                break
            continue

        if scalars.size != len(normalized_locations):
            logger.debug(
                "HRRR point count mismatch for f%02d: expected %d, got %d",
                lead_hour,
                len(normalized_locations),
                scalars.size,
            )
            continue

        valid_time = run_time + timedelta(hours=lead_hour)
        valid_stamp = valid_time.isoformat()
        for idx, loc in enumerate(normalized_locations):
            scalar = scalars[idx]
            if np.isnan(scalar):
                continue
            results[loc["name"]]["times"].append(valid_stamp)
            results[loc["name"]]["temperature_2m_f"].append(round(_temp_to_f(scalar, units), 2))

    for loc in normalized_locations:
        name = loc["name"]
        if not results[name]["temperature_2m_f"]:
            results[name] = None

    _store_cached_multi_result(run_time, normalized_locations, fxx, results)
    return results


def get_hrrr_high_low(
    hrrr_data: dict,
    target_date: str,
    timezone_name: Optional[str] = "UTC",
) -> tuple[Optional[float], Optional[float]]:
    """Extract predicted daily high/low from an HRRR timeseries."""
    if not hrrr_data or not target_date:
        return None, None

    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None  # type: ignore[assignment]

    if timezone_name and ZoneInfo is not None:
        try:
            tzinfo = ZoneInfo(timezone_name)
        except Exception:
            tzinfo = timezone.utc
    else:
        tzinfo = timezone.utc

    day_temps: list[float] = []
    for stamp, temp_f in zip(hrrr_data.get("times", []), hrrr_data.get("temperature_2m_f", [])):
        valid_time = _parse_iso_datetime(str(stamp))
        if valid_time is None:
            continue
        local_date = valid_time.astimezone(tzinfo).date().isoformat()
        if local_date == target_date:
            day_temps.append(float(temp_f))

    if not day_temps:
        return None, None

    return max(day_temps), min(day_temps)
