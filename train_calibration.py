"""
Train per-city calibration models from archived forecasts and station actuals.

Usage:
    python train_calibration.py
    python train_calibration.py --city "New York" --days 60
"""

import argparse
import logging
from pathlib import Path

from src.config import load_app_config, resolve_config_path
from src.calibration import train_city_models
from src.logging_setup import configure_logging, set_log_level
from src.station_truth import (
    CALIBRATION_MODELS_DIR,
    FORECAST_ARCHIVE_DIR,
    STATION_ACTUALS_DIR,
    build_training_set,
    ensure_data_directories,
    load_station_map,
)

_CONFIG_PATH = resolve_config_path()
log = configure_logging(
    "weather.train_calibration",
    config_path=_CONFIG_PATH,
    log_filename="train_calibration.log",
    level=logging.INFO,
)


def load_config(config_path: Path | str = _CONFIG_PATH) -> dict:
    return load_app_config(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Weather Signals calibration models")
    parser.add_argument("--config", default=str(_CONFIG_PATH), help="Config path")
    parser.add_argument("--city", help="Train one city only")
    parser.add_argument("--days", type=int, help="Rolling training window in days")
    parser.add_argument("--min-rows", type=int, default=10, help="Minimum overlapping rows required")
    parser.add_argument("--model-dir", help="Override model output directory")
    parser.add_argument("--station-actuals-dir", help="Override station actuals directory")
    parser.add_argument("--forecast-archive-dir", help="Override forecast archive directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        set_log_level(logging.DEBUG)

    config = load_config(args.config)
    ensure_data_directories()

    days = args.days or int(config.get("calibration_window_days", 90))
    model_dir = Path(args.model_dir or config.get("calibration_model_dir", CALIBRATION_MODELS_DIR))
    station_actuals_dir = Path(args.station_actuals_dir or STATION_ACTUALS_DIR)
    forecast_archive_dir = Path(args.forecast_archive_dir or FORECAST_ARCHIVE_DIR)

    station_map = load_station_map()
    cities = [args.city] if args.city else list(station_map.keys())

    total_city_models = 0
    total_isotonic_models = 0
    total_ngr_models = 0
    skipped_cities = 0

    log.info("Training calibration models for %d cities (window=%d days)...", len(cities), days)

    for city in cities:
        training_df = build_training_set(
            city,
            days=days,
            station_actuals_dir=station_actuals_dir,
            forecast_archive_dir=forecast_archive_dir,
        )

        if training_df.empty:
            skipped_cities += 1
            log.info("%s: no overlapping forecast/archive history yet, skipping", city)
            continue

        results = train_city_models(
            city,
            training_df,
            model_dir=model_dir,
            min_training_rows=args.min_rows,
        )

        city_trained = 0
        city_isotonic = 0
        city_ngr = 0
        for market_type, outcome in results.items():
            if outcome.get("trained_emos"):
                city_trained += 1
                total_city_models += 1
            if outcome.get("trained_isotonic"):
                city_isotonic += 1
                total_isotonic_models += 1
            if outcome.get("trained_ngr"):
                city_ngr += 1
                total_ngr_models += 1

            log.info(
                "%s %s: status=%s rows=%s ngr_crps=%s reason=%s",
                city,
                market_type,
                outcome.get("status"),
                outcome.get("rows"),
                outcome.get("ngr_training_crps"),
                outcome.get("reason", ""),
            )

        if city_trained == 0:
            skipped_cities += 1
            log.info("%s: insufficient overlapping rows for training", city)
        else:
            log.info(
                "%s: trained %d EMOS, %d isotonic, %d NGR",
                city, city_trained, city_isotonic, city_ngr,
            )

    log.info(
        "Training complete: %d EMOS, %d isotonic, %d NGR, %d skipped",
        total_city_models,
        total_isotonic_models,
        total_ngr_models,
        skipped_cities,
    )


if __name__ == "__main__":
    main()
