"""
Backfill recent training data for Weather Signals calibration.

This script fills both sides of the training join:
- station actuals from NOAA/NWS sources
- archived one-day-ahead forecasts from Open-Meteo Previous Runs

Usage:
    python backfill_training_data.py
    python backfill_training_data.py --days 21 --city "New York"
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config import load_app_config, resolve_config_path
from src.fetch_forecasts import fetch_previous_run_forecast
from src.logging_setup import configure_logging, set_log_level
from src.station_truth import (
    CALIBRATION_MODELS_DIR,
    FORECAST_ARCHIVE_DIR,
    PRECIP_ARCHIVE_DIR,
    STATION_ACTUALS_DIR,
    archive_previous_run_forecast,
    archive_previous_run_precipitation,
    backfill_station_actuals,
    backfill_station_actuals_from_cli_archive,
    build_training_set,
    ensure_data_directories,
    load_station_map,
)

_CONFIG_PATH = resolve_config_path()
log = configure_logging(
    "weather.backfill_training",
    config_path=_CONFIG_PATH,
    log_filename="backfill_training_data.log",
    level=logging.INFO,
)


def load_config(config_path: Path | str = _CONFIG_PATH) -> dict:
    return load_app_config(config_path)


def _infer_date_window(days: int) -> tuple[str, str]:
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=max(0, int(days) - 1))
    return start_date.isoformat(), end_date.isoformat()


def _load_csv_dates(path: Path) -> tuple[str | None, str | None, int]:
    if not path.exists():
        return None, None, 0

    frame = pd.read_csv(path)
    if frame.empty or "date" not in frame.columns:
        return None, None, int(len(frame))

    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None, None, int(len(frame))

    return dates.min().date().isoformat(), dates.max().date().isoformat(), int(len(frame))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Weather Signals training data")
    parser.add_argument("--config", default=str(_CONFIG_PATH), help="Config path")
    parser.add_argument("--city", help="Backfill one city only")
    parser.add_argument("--days", type=int, default=30, help="Requested backfill window in days")
    parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--lead-days", type=int, default=1, help="Forecast lead to archive from previous runs")
    parser.add_argument("--forecast-model", default="best_match", help="Open-Meteo previous-runs model")
    parser.add_argument("--station-actuals-dir", help="Override station actuals directory")
    parser.add_argument("--forecast-archive-dir", help="Override forecast archive directory")
    parser.add_argument("--precip-archive-dir", help="Override precipitation archive directory")
    parser.add_argument("--train-window-days", type=int, default=365, help="Window used when reporting overlap")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        set_log_level(logging.DEBUG)

    config = load_config(args.config)
    ensure_data_directories()

    station_actuals_dir = Path(args.station_actuals_dir or STATION_ACTUALS_DIR)
    forecast_archive_dir = Path(args.forecast_archive_dir or FORECAST_ARCHIVE_DIR)
    precip_archive_dir = Path(args.precip_archive_dir or PRECIP_ARCHIVE_DIR)
    station_map = load_station_map()
    cities = [args.city] if args.city else list(station_map.keys())
    start_date, end_date = args.start, args.end
    if not start_date or not end_date:
        start_date, end_date = _infer_date_window(args.days)

    ncei_token = str(config.get("ncei_api_token", "") or "").strip()
    log.info(
        "Backfilling %d cities from %s to %s using %s actuals and %s previous-run forecasts (lead=%d day).",
        len(cities),
        start_date,
        end_date,
        "NCEI CDO" if ncei_token else "NWS CLI archive",
        args.forecast_model,
        args.lead_days,
    )

    total_overlap_rows = 0
    failed_cities: list[str] = []

    for city in cities:
        station_info = station_map[city]
        log.info("%s: starting backfill", city)

        try:
            if ncei_token:
                actuals_path = backfill_station_actuals(
                    city,
                    start_date,
                    end_date,
                    token=ncei_token,
                    base_dir=station_actuals_dir,
                    stations=station_map,
                )
            else:
                actuals_path = backfill_station_actuals_from_cli_archive(
                    city,
                    start=start_date,
                    end=end_date,
                    base_dir=station_actuals_dir,
                    stations=station_map,
                )

            actual_start, actual_end, actual_rows = _load_csv_dates(actuals_path)
            if not actual_start or not actual_end:
                log.info("%s: no station actuals available in requested range", city)
                continue

            forecast = fetch_previous_run_forecast(
                station_info["lat"],
                station_info["lon"],
                actual_start,
                actual_end,
                lead_days=args.lead_days,
                model=args.forecast_model,
            )
            if forecast is None:
                log.info("%s: previous-run forecast fetch failed", city)
                continue

            archive_path = archive_previous_run_forecast(
                city,
                forecast,
                lead_days=args.lead_days,
                model=args.forecast_model,
                start_date=actual_start,
                end_date=actual_end,
                base_dir=forecast_archive_dir,
            )
            archive_start, archive_end, archive_rows = _load_csv_dates(archive_path)

            # Archive precipitation snapshots alongside the temperature archive.
            daily_block = forecast.get("daily", {}) if isinstance(forecast, dict) else {}
            daily_times = list(daily_block.get("time", []) or [])
            precip_sum_series = list(daily_block.get("precipitation_sum_in", []) or [])
            precip_prob_series = list(daily_block.get("precipitation_probability_max", []) or [])
            precip_rows_written = 0
            for idx, target_date in enumerate(daily_times):
                # Intentional: clip precip-forecast archive to the temperature-actuals
                # window so training joins stay aligned. Revisit when PRCP-specific
                # actuals windows diverge from TMAX/TMIN coverage.
                if actual_start and str(target_date) < actual_start:
                    continue
                if actual_end and str(target_date) > actual_end:
                    continue
                precip_sum_in = precip_sum_series[idx] if idx < len(precip_sum_series) else None
                precip_prob = precip_prob_series[idx] if idx < len(precip_prob_series) else None
                if precip_sum_in is None and precip_prob is None:
                    continue
                # as_of_utc is derived inside archive_previous_run_precipitation
                # from (date, lead_days); no need to compute it here.
                snapshot = {
                    "date": str(target_date),
                    "precipitation_sum_in": precip_sum_in,
                    "precipitation_probability_max": precip_prob,
                    "lead_days": int(args.lead_days),
                    "forecast_model": args.forecast_model,
                    "forecast_source": "open_meteo_previous_runs",
                }
                archive_previous_run_precipitation(
                    city=city,
                    snapshot=snapshot,
                    base_dir=precip_archive_dir,
                )
                precip_rows_written += 1
            if precip_rows_written:
                log.info("%s: archived %d precipitation snapshots", city, precip_rows_written)

            training_df = build_training_set(
                city,
                days=args.train_window_days,
                station_actuals_dir=station_actuals_dir,
                forecast_archive_dir=forecast_archive_dir,
            )
            overlap_rows = int(len(training_df))
            total_overlap_rows += overlap_rows

            log.info(
                "%s: actuals=%d (%s to %s), archive=%d (%s to %s), overlap=%d",
                city,
                actual_rows,
                actual_start,
                actual_end,
                archive_rows,
                archive_start,
                archive_end,
                overlap_rows,
            )
        except Exception as exc:
            failed_cities.append(city)
            log.warning("%s: backfill failed: %s", city, exc)
            continue

    log.info("Backfill complete: %d total overlapping training row(s)", total_overlap_rows)
    if failed_cities:
        log.warning("Cities with backfill failures: %s", ", ".join(failed_cities))


if __name__ == "__main__":
    main()
