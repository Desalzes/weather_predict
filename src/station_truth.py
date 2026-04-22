"""
Station truth ingestion and forecast archiving for Weather Signals.

Provides:
- NWS CLI parsing for official daily station highs/lows
- NCEI CDO historical daily fetches for backfills
- local CSV persistence for station actuals and forecast archives
- a training-set builder that joins archived forecasts with saved actuals
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("weather.station_truth")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIONS_PATH = PROJECT_ROOT / "stations.json"
DATA_DIR = PROJECT_ROOT / "data"
STATION_ACTUALS_DIR = DATA_DIR / "station_actuals"
FORECAST_ARCHIVE_DIR = DATA_DIR / "forecast_archive"
PRECIP_ARCHIVE_DIR = DATA_DIR / "precip_archive"
CALIBRATION_MODELS_DIR = DATA_DIR / "calibration_models"

_PRECIP_ARCHIVE_COLUMNS = [
    "as_of_utc",
    "date",
    "forecast_prob_any_rain",
    "forecast_amount_in",
    "ensemble_wet_fraction",
    "ensemble_amount_std_in",
    "forecast_model",
    "forecast_lead_days",
    "forecast_source",
]

_CLI_URL = "https://forecast.weather.gov/product.php"
_CDO_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

_EMPTY_TRAINING_COLUMNS = [
    "date",
    "forecast_high_f",
    "actual_high_f",
    "forecast_low_f",
    "actual_low_f",
    "ensemble_std_f",
    "ensemble_high_std_f",
    "ensemble_low_std_f",
    "forecast_lead_days",
    "as_of_utc",
]


def _prepare_forecast_archive_frame(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Normalize archived forecast metadata before training joins."""
    frame = forecasts.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["as_of_utc"] = pd.to_datetime(
        frame["as_of_utc"],
        utc=True,
        errors="coerce",
        format="ISO8601",
    )

    if "forecast_source" not in frame.columns:
        frame["forecast_source"] = "live_scan"
    else:
        frame["forecast_source"] = frame["forecast_source"].fillna("").astype(str).str.strip()
        frame.loc[frame["forecast_source"] == "", "forecast_source"] = "live_scan"

    if "forecast_lead_days" not in frame.columns:
        frame["forecast_lead_days"] = pd.NA
    frame["forecast_lead_days"] = pd.to_numeric(frame["forecast_lead_days"], errors="coerce")

    # Prefer explicit previous-run forecasts and ignore later same-day scans that
    # can leak settlement information into the calibration set.
    frame["training_priority"] = 0
    previous_run_mask = (
        frame["forecast_source"].eq("open_meteo_previous_runs")
        | frame["forecast_lead_days"].ge(1).fillna(False)
    )
    frame.loc[previous_run_mask, "training_priority"] = 1
    return frame


def ensure_data_directories() -> dict[str, Path]:
    """Ensure the Phase 2+ data directories exist."""
    paths = {
        "data": DATA_DIR,
        "station_actuals": STATION_ACTUALS_DIR,
        "forecast_archive": FORECAST_ARCHIVE_DIR,
        "precip_archive": PRECIP_ARCHIVE_DIR,
        "calibration_models": CALIBRATION_MODELS_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_station_map(path: Path | str = STATIONS_PATH) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_station_info(city: str, stations: Optional[dict[str, dict]] = None) -> dict:
    station_map = stations or load_station_map()
    if city not in station_map:
        raise KeyError(f"Unknown station mapping for city: {city}")
    return station_map[city]


def _slugify_city(city: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", city.lower()).strip("_")
    return slug or "unknown_city"


def station_actuals_path(city: str, base_dir: Optional[Path | str] = None) -> Path:
    directory = Path(base_dir) if base_dir is not None else STATION_ACTUALS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slugify_city(city)}.csv"


def forecast_archive_path(city: str, base_dir: Optional[Path | str] = None) -> Path:
    directory = Path(base_dir) if base_dir is not None else FORECAST_ARCHIVE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slugify_city(city)}.csv"


def _extract_preformatted_text(page_html: str) -> str:
    match = re.search(r"<pre[^>]*>(.*?)</pre>", page_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("NWS CLI page did not contain a report <pre> block")
    return unescape(match.group(1)).replace("\r\n", "\n").replace("\r", "\n")


def _extract_section(report_text: str, start_header: str, end_headers: list[str]) -> str:
    start = report_text.find(start_header)
    if start == -1:
        return ""

    end = len(report_text)
    for end_header in end_headers:
        idx = report_text.find(end_header, start + len(start_header))
        if idx != -1:
            end = min(end, idx)

    return report_text[start:end]


def _search_value(section_text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, section_text, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _parse_cli_measurement(token: Optional[str], *, allow_trace: bool = False) -> tuple[Optional[float], bool]:
    if token is None:
        return None, False

    value = token.strip().upper()
    if value == "MM":
        return None, False
    if value == "T":
        return (0.0 if allow_trace else None), True

    return float(value), False


def _parse_cli_summary_date(report_text: str) -> str:
    match = re.search(r"CLIMATE SUMMARY FOR ([A-Z]+ \d{1,2} \d{4})", report_text)
    if not match:
        raise ValueError("Could not find observation date in CLI report")
    dt = datetime.strptime(match.group(1).title(), "%B %d %Y")
    return dt.date().isoformat()


def _resolve_cli_station(station: str, stations: Optional[dict[str, dict]] = None) -> str:
    value = station.strip()
    if not value:
        raise ValueError("Station cannot be blank")

    station_map = stations or load_station_map()
    if value in station_map:
        cli_station = station_map[value].get("cli")
        if cli_station:
            return str(cli_station).upper()

    upper_value = value.upper()
    for info in station_map.values():
        if upper_value in {
            str(info.get("cli", "")).upper(),
            str(info.get("icao", "")).upper(),
            str(info.get("ghcnd", "")).upper(),
        }:
            cli_station = info.get("cli")
            if cli_station:
                return str(cli_station).upper()

    return upper_value


def parse_cli_report_text(
    report_text: str,
    *,
    cli_station: Optional[str] = None,
    source_url: Optional[str] = None,
) -> dict:
    """Parse an extracted CLI report into a normalized daily record."""
    observation_date = _parse_cli_summary_date(report_text)
    temp_section = _extract_section(report_text, "TEMPERATURE (F)", ["PRECIPITATION (IN)"])
    precip_section = _extract_section(report_text, "PRECIPITATION (IN)", ["SNOWFALL (IN)", "DEGREE DAYS"])

    tmax_f, _ = _parse_cli_measurement(
        _search_value(temp_section, r"^\s*MAXIMUM\s+(-?\d+(?:\.\d+)?|MM)\b")
    )
    tmin_f, _ = _parse_cli_measurement(
        _search_value(temp_section, r"^\s*MINIMUM\s+(-?\d+(?:\.\d+)?|MM)\b")
    )
    precip_in, precip_trace = _parse_cli_measurement(
        _search_value(precip_section, r"^\s*YESTERDAY\s+([0-9.]+|T|MM)\b"),
        allow_trace=True,
    )

    return {
        "date": observation_date,
        "tmax_f": tmax_f,
        "tmin_f": tmin_f,
        "precip_in": precip_in,
        "precip_trace": precip_trace,
        "cli_station": cli_station,
        "source_url": source_url,
    }


def _fetch_cli_response(station: str, *, version: Optional[int] = None) -> requests.Response:
    cli_station = _resolve_cli_station(station)
    params = {
        "site": "NWS",
        "issuedby": cli_station,
        "product": "CLI",
    }
    if version is not None:
        params["version"] = int(version)
        params["format"] = "txt"
        params["glossary"] = "0"

    response = requests.get(_CLI_URL, params=params, timeout=30)
    response.raise_for_status()
    return response


def fetch_cli_report(station: str, *, version: Optional[int] = None) -> dict:
    """Fetch and parse the latest or a versioned NWS CLI report."""
    cli_station = _resolve_cli_station(station)
    response = _fetch_cli_response(cli_station, version=version)
    report_text = _extract_preformatted_text(response.text)
    return parse_cli_report_text(report_text, cli_station=cli_station, source_url=response.url)


def available_cli_versions(station: str) -> list[int]:
    """Return available archived version numbers for a CLI station page."""
    response = _fetch_cli_response(station)
    versions = {1}
    for match in re.findall(r"version=(\d+)", response.text):
        versions.add(int(match))
    return sorted(versions)


def _normalize_station_id(ghcnd_id: str) -> str:
    if ":" in ghcnd_id:
        return ghcnd_id
    return f"GHCND:{ghcnd_id}"


def _normalize_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _cdo_temp_tenths_c_to_f(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return round((float(value) / 10.0) * 9.0 / 5.0 + 32.0, 2)


def _cdo_precip_tenths_mm_to_in(value: Optional[float]) -> Optional[float]:
    """Convert NCEI CDO PRCP value (tenths of mm) to inches."""
    if value is None or pd.isna(value):
        return None
    # PRCP is reported in tenths of a millimeter; 1 inch = 25.4 mm = 254 tenths of mm.
    # Rounding to 3 decimals (not 2) preserves single-tenth-mm readings (~0.004 in)
    # that 2-decimal rounding would truncate to zero.
    return round(float(value) / 254.0, 3)


def fetch_historical_daily(
    ghcnd_id: str,
    start: str | date | datetime,
    end: str | date | datetime,
    token: str,
) -> pd.DataFrame:
    """Fetch GHCND daily TMAX/TMIN/PRCP via NOAA CDO and return Fahrenheit/inch values.

    Returns a DataFrame with columns: ``date`` (YYYY-MM-DD string),
    ``tmax_f`` and ``tmin_f`` (Fahrenheit), and ``precip_in`` (inches).
    Note that CDO does not report trace precipitation — only numeric PRCP totals —
    so ``precip_trace`` is populated only from the CLI path, never from this function.
    """
    if not token:
        raise ValueError("NCEI API token is required")

    params = {
        "datasetid": "GHCND",
        "stationid": _normalize_station_id(ghcnd_id),
        "datatypeid": "TMAX,TMIN,PRCP",
        "startdate": _normalize_date(start),
        "enddate": _normalize_date(end),
        "limit": 1000,
        "offset": 1,
    }
    headers = {"token": token}
    rows: list[dict] = []

    while True:
        payload = None
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = requests.get(_CDO_URL, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                payload = response.json()
                last_error = None
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= 3:
                    break
                logger.warning(
                    "NCEI CDO request failed for %s (attempt %d/3): %s",
                    ghcnd_id,
                    attempt,
                    exc,
                )
                time_module.sleep(float(attempt))

        if payload is None:
            raise RuntimeError(f"NCEI CDO request failed for {ghcnd_id}: {last_error}")

        results = payload.get("results", [])
        rows.extend(results)

        resultset = payload.get("metadata", {}).get("resultset", {})
        count = int(resultset.get("count", len(rows)))
        offset = int(resultset.get("offset", params["offset"]))
        limit = int(resultset.get("limit", params["limit"]))

        if offset + limit > count or not results:
            break
        params["offset"] = offset + limit

    if not rows:
        return pd.DataFrame(columns=["date", "tmax_f", "tmin_f", "precip_in"])

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    pivot = (
        frame.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for column in ("TMAX", "TMIN", "PRCP"):
        if column not in pivot.columns:
            pivot[column] = pd.NA

    pivot["tmax_f"] = pivot["TMAX"].apply(_cdo_temp_tenths_c_to_f)
    pivot["tmin_f"] = pivot["TMIN"].apply(_cdo_temp_tenths_c_to_f)
    pivot["precip_in"] = pivot["PRCP"].apply(_cdo_precip_tenths_mm_to_in)

    return (
        pivot[["date", "tmax_f", "tmin_f", "precip_in"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def save_station_actuals(city: str, actuals: pd.DataFrame | dict, base_dir: Optional[Path | str] = None) -> Path:
    """Upsert daily station actual rows for a city into its CSV."""
    path = station_actuals_path(city, base_dir=base_dir)
    if isinstance(actuals, dict):
        frame = pd.DataFrame([actuals])
    else:
        frame = actuals.copy()

    if frame.empty:
        return path

    required = {"date", "tmax_f", "tmin_f"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Station actuals are missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)

    if path.exists():
        existing = pd.read_csv(path)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date.astype(str)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    else:
        combined = frame

    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    combined.to_csv(path, index=False)
    return path


def update_station_actuals_from_cli(city: str, base_dir: Optional[Path | str] = None) -> Path:
    """Fetch the latest CLI report for a city and upsert it into local storage."""
    report = fetch_cli_report(city)
    report["city"] = city
    return save_station_actuals(city, report, base_dir=base_dir)


def backfill_station_actuals(
    city: str,
    start: str | date | datetime,
    end: str | date | datetime,
    token: str,
    base_dir: Optional[Path | str] = None,
    stations: Optional[dict[str, dict]] = None,
) -> Path:
    """Backfill a city's historical daily station actuals from NOAA CDO."""
    station = get_station_info(city, stations=stations)
    daily = fetch_historical_daily(station["ghcnd"], start, end, token)
    daily["city"] = city
    daily["source"] = "cdo"
    return save_station_actuals(city, daily, base_dir=base_dir)


def backfill_station_actuals_from_cli_archive(
    city: str,
    start: Optional[str | date | datetime] = None,
    end: Optional[str | date | datetime] = None,
    *,
    base_dir: Optional[Path | str] = None,
    stations: Optional[dict[str, dict]] = None,
) -> Path:
    """Backfill a city's recent daily station actuals from NWS CLI archived versions."""
    _ = get_station_info(city, stations=stations)
    start_str = _normalize_date(start) if start is not None else None
    end_str = _normalize_date(end) if end is not None else None

    rows_by_date: dict[str, dict] = {}
    for version in available_cli_versions(city):
        try:
            row = fetch_cli_report(city, version=version)
        except Exception as exc:
            logger.debug("CLI archive fetch failed for %s version %s: %s", city, version, exc)
            continue

        row_date = str(row["date"])
        if start_str and row_date < start_str:
            continue
        if end_str and row_date > end_str:
            continue

        row["city"] = city
        row["source"] = "cli_archive"
        row["archive_version"] = int(version)
        rows_by_date[row_date] = row

    if not rows_by_date:
        return station_actuals_path(city, base_dir=base_dir)

    frame = pd.DataFrame(sorted(rows_by_date.values(), key=lambda item: item["date"]))
    return save_station_actuals(city, frame, base_dir=base_dir)


def _extract_day_series(hourly_data: dict, field_name: str, target_date: str) -> list[float]:
    times = hourly_data.get("time", [])
    values = hourly_data.get(field_name, [])
    day_values = []
    for idx, stamp in enumerate(times):
        if idx >= len(values):
            break
        if str(stamp).startswith(target_date):
            day_values.append(values[idx])
    return day_values


def _c_to_f(celsius: float) -> float:
    return float(celsius) * 9.0 / 5.0 + 32.0


def _member_spread_for_date(ensemble_forecast: Optional[dict], target_date: str, mode: str) -> Optional[float]:
    if not ensemble_forecast:
        return None

    hourly = ensemble_forecast.get("hourly", {})
    member_keys = sorted(key for key in hourly if key.startswith("temperature_2m_member"))
    if len(member_keys) < 2:
        return None

    extremes = []
    for key in member_keys:
        temps = _extract_day_series(hourly, key, target_date)
        if not temps:
            continue
        temps_f = [_c_to_f(value) for value in temps]
        extremes.append(max(temps_f) if mode == "high" else min(temps_f))

    if len(extremes) < 2:
        return None

    return round(statistics.pstdev(extremes), 2)


def archive_forecast_snapshot(
    city: str,
    forecast: dict,
    ensemble_forecast: Optional[dict] = None,
    *,
    as_of: Optional[str | datetime] = None,
    base_dir: Optional[Path | str] = None,
) -> Path:
    """Archive daily high/low forecast summaries for one city."""
    path = forecast_archive_path(city, base_dir=base_dir)
    hourly = forecast.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])

    if not times or not temps:
        return path

    as_of_utc = as_of
    if isinstance(as_of_utc, datetime):
        as_of_utc = as_of_utc.astimezone(timezone.utc).isoformat()
    if as_of_utc is None:
        as_of_utc = datetime.now(timezone.utc).isoformat()

    per_day: dict[str, list[float]] = {}
    for idx, stamp in enumerate(times):
        if idx >= len(temps):
            break
        target_date = str(stamp)[:10]
        per_day.setdefault(target_date, []).append(_c_to_f(temps[idx]))

    rows = []
    for target_date, temps_f in sorted(per_day.items()):
        rows.append({
            "as_of_utc": as_of_utc,
            "date": target_date,
            "forecast_high_f": round(max(temps_f), 2),
            "forecast_low_f": round(min(temps_f), 2),
            "ensemble_high_std_f": _member_spread_for_date(ensemble_forecast, target_date, "high"),
            "ensemble_low_std_f": _member_spread_for_date(ensemble_forecast, target_date, "low"),
            "forecast_source": "live_scan",
        })

    if not rows:
        return path

    frame = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    else:
        combined = frame

    combined = combined.drop_duplicates(subset=["as_of_utc", "date"], keep="last")
    combined = combined.sort_values(["as_of_utc", "date"])
    combined.to_csv(path, index=False)
    return path


def archive_previous_run_forecast(
    city: str,
    forecast: dict,
    *,
    lead_days: int = 1,
    model: Optional[str] = None,
    start_date: Optional[str | date | datetime] = None,
    end_date: Optional[str | date | datetime] = None,
    base_dir: Optional[Path | str] = None,
) -> Path:
    """Archive one previous-run hourly forecast payload as daily high/low rows."""
    path = forecast_archive_path(city, base_dir=base_dir)
    hourly = forecast.get("hourly", {})
    variable = f"temperature_2m_previous_day{int(lead_days)}"
    times = hourly.get("time", [])
    temps = hourly.get(variable, [])

    if not times or not temps:
        return path

    start_str = _normalize_date(start_date) if start_date is not None else None
    end_str = _normalize_date(end_date) if end_date is not None else None

    per_day: dict[str, list[float]] = {}
    for idx, stamp in enumerate(times):
        if idx >= len(temps):
            break
        value = temps[idx]
        if pd.isna(value):
            continue
        target_date = str(stamp)[:10]
        if start_str and target_date < start_str:
            continue
        if end_str and target_date > end_str:
            continue
        per_day.setdefault(target_date, []).append(_c_to_f(value))

    rows = []
    for target_date, temps_f in sorted(per_day.items()):
        as_of_date = datetime.combine(
            datetime.fromisoformat(target_date).date() - timedelta(days=int(lead_days)),
            time(hour=12),
            tzinfo=timezone.utc,
        )
        rows.append(
            {
                "as_of_utc": as_of_date.isoformat(),
                "date": target_date,
                "forecast_high_f": round(max(temps_f), 2),
                "forecast_low_f": round(min(temps_f), 2),
                "ensemble_high_std_f": pd.NA,
                "ensemble_low_std_f": pd.NA,
                "forecast_model": model or forecast.get("model"),
                "forecast_lead_days": int(lead_days),
                "forecast_source": "open_meteo_previous_runs",
            }
        )

    if not rows:
        return path

    frame = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    else:
        combined = frame

    combined = combined.drop_duplicates(subset=["as_of_utc", "date"], keep="last")
    combined = combined.sort_values(["as_of_utc", "date"])
    combined.to_csv(path, index=False)
    return path


def archive_previous_run_precipitation(
    city: str,
    snapshot: dict,
    *,
    base_dir: Optional[Path | str] = None,
) -> Path:
    """Archive a single-day precipitation snapshot from a previous-run forecast.

    Expected keys on ``snapshot``:
    - ``date`` (YYYY-MM-DD)
    - ``precipitation_sum_in`` (inches, may be None)
    - ``precipitation_probability_max`` (percent 0-100, may be None)
    - ``lead_days`` (optional, int)
    - ``as_of_utc`` (optional, ISO-8601 string; derived from date/lead_days when absent)
    - ``forecast_source`` (optional; defaults to "open_meteo_previous_runs")
    - ``forecast_model`` (optional; defaults to "best_match")
    """
    directory = Path(base_dir) if base_dir is not None else PRECIP_ARCHIVE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slugify_city(city)}.csv"

    prob = snapshot.get("precipitation_probability_max")
    try:
        prob_fraction = None if prob is None else float(prob) / 100.0
    except (TypeError, ValueError):
        prob_fraction = None

    amount_in = snapshot.get("precipitation_sum_in")
    try:
        amount_value = None if amount_in is None else float(amount_in)
    except (TypeError, ValueError):
        amount_value = None

    # Derive as_of_utc from date + lead_days when callers don't supply it, so
    # every archived row lands on the same canonical (as_of_utc, date) dedup
    # key regardless of which code path produced the snapshot. Mirrors the
    # derivation inside ``archive_previous_run_forecast``.
    as_of_utc = snapshot.get("as_of_utc")
    if as_of_utc is None:
        target_date = snapshot.get("date")
        lead_days_raw = snapshot.get("lead_days")
        if target_date is not None and lead_days_raw is not None:
            try:
                as_of_dt = datetime.combine(
                    datetime.fromisoformat(str(target_date)).date()
                    - timedelta(days=int(lead_days_raw)),
                    time(hour=12),
                    tzinfo=timezone.utc,
                )
                as_of_utc = as_of_dt.isoformat()
            except (TypeError, ValueError):
                as_of_utc = None

    row = {
        "as_of_utc": as_of_utc,
        "date": snapshot.get("date"),
        "forecast_prob_any_rain": prob_fraction,
        "forecast_amount_in": amount_value,
        "ensemble_wet_fraction": None,
        "ensemble_amount_std_in": None,
        "forecast_model": snapshot.get("forecast_model", "best_match"),
        "forecast_lead_days": snapshot.get("lead_days"),
        "forecast_source": snapshot.get("forecast_source", "open_meteo_previous_runs"),
    }

    frame = pd.DataFrame([row], columns=_PRECIP_ARCHIVE_COLUMNS)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    else:
        combined = frame

    combined = combined.drop_duplicates(subset=["as_of_utc", "date"], keep="last")
    combined = combined.sort_values(["as_of_utc", "date"])
    combined.to_csv(path, index=False)
    return path


def build_rain_training_set(
    city: str,
    actuals_dir: Optional[Path | str] = None,
    precip_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Join archived precipitation forecasts with station precip actuals.

    Returns a DataFrame with columns:
      date, raw_prob, forecast_amount_in, ensemble_wet_fraction,
      ensemble_amount_std_in, actual_precip_in, actual_wet_0_1,
      forecast_lead_days, as_of_utc.
    """
    actuals_dir = Path(actuals_dir) if actuals_dir is not None else STATION_ACTUALS_DIR
    precip_dir = Path(precip_dir) if precip_dir is not None else PRECIP_ARCHIVE_DIR
    slug = _slugify_city(city)
    actuals_path = actuals_dir / f"{slug}.csv"
    precip_path = precip_dir / f"{slug}.csv"
    empty_cols = [
        "date", "raw_prob", "forecast_amount_in", "ensemble_wet_fraction",
        "ensemble_amount_std_in", "actual_precip_in", "actual_wet_0_1",
        "forecast_lead_days", "as_of_utc",
    ]
    if not actuals_path.exists() or not precip_path.exists():
        return pd.DataFrame(columns=empty_cols)

    actuals = pd.read_csv(actuals_path)
    precip = pd.read_csv(precip_path)
    if precip.empty or actuals.empty:
        return pd.DataFrame(columns=empty_cols)

    # Prefer earliest lead-1 forecast per date (mirrors temp pipeline discipline).
    precip["date"] = pd.to_datetime(precip["date"]).dt.strftime("%Y-%m-%d")
    precip = precip.sort_values(["date", "forecast_lead_days", "as_of_utc"])
    precip = precip.drop_duplicates("date", keep="first")

    actuals["date"] = pd.to_datetime(actuals["date"]).dt.strftime("%Y-%m-%d")
    if "precip_in" not in actuals.columns:
        return pd.DataFrame(columns=empty_cols)

    joined = precip.merge(actuals[["date", "precip_in"]], on="date", how="inner")
    if joined.empty:
        return pd.DataFrame(columns=empty_cols)

    joined["actual_precip_in"] = pd.to_numeric(joined["precip_in"], errors="coerce")
    joined["actual_wet_0_1"] = (joined["actual_precip_in"] >= 0.01).astype("Int64")

    return joined.rename(
        columns={"forecast_prob_any_rain": "raw_prob"}
    )[[
        "date", "raw_prob", "forecast_amount_in", "ensemble_wet_fraction",
        "ensemble_amount_std_in", "actual_precip_in", "actual_wet_0_1",
        "forecast_lead_days", "as_of_utc",
    ]].reset_index(drop=True)


def build_training_set(
    city: str,
    days: int = 90,
    *,
    station_actuals_dir: Optional[Path | str] = None,
    forecast_archive_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Join archived forecasts with station actuals for a city."""
    actuals_file = station_actuals_path(city, base_dir=station_actuals_dir)
    archive_file = forecast_archive_path(city, base_dir=forecast_archive_dir)

    if not actuals_file.exists() or not archive_file.exists():
        return pd.DataFrame(columns=_EMPTY_TRAINING_COLUMNS)

    actuals = pd.read_csv(actuals_file)
    forecasts = _prepare_forecast_archive_frame(pd.read_csv(archive_file))
    if actuals.empty or forecasts.empty:
        return pd.DataFrame(columns=_EMPTY_TRAINING_COLUMNS)

    actuals["date"] = pd.to_datetime(actuals["date"])
    forecasts = forecasts.loc[forecasts["training_priority"] > 0].copy()
    if forecasts.empty:
        return pd.DataFrame(columns=_EMPTY_TRAINING_COLUMNS)

    latest_forecast = (
        forecasts.sort_values(["date", "training_priority", "as_of_utc"])
        .drop_duplicates(subset=["date"], keep="last")
        .copy()
    )

    merged = latest_forecast.merge(
        actuals[["date", "tmax_f", "tmin_f"]],
        on="date",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=_EMPTY_TRAINING_COLUMNS)

    cutoff = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=days))
    merged = merged.loc[merged["date"] >= cutoff].copy()
    if merged.empty:
        return pd.DataFrame(columns=_EMPTY_TRAINING_COLUMNS)

    merged["ensemble_std_f"] = merged[["ensemble_high_std_f", "ensemble_low_std_f"]].max(axis=1, skipna=True)
    merged = merged.rename(
        columns={
            "tmax_f": "actual_high_f",
            "tmin_f": "actual_low_f",
        }
    )

    columns = _EMPTY_TRAINING_COLUMNS.copy()
    result = merged[columns].copy()
    result["date"] = result["date"].dt.date.astype(str)
    result["forecast_lead_days"] = pd.to_numeric(result["forecast_lead_days"], errors="coerce").astype("Int64")
    result["as_of_utc"] = result["as_of_utc"].astype(str)
    return result.reset_index(drop=True)
