"""Train per-city rain calibration models (logistic + isotonic).

Usage:
    python train_rain_calibration.py
    python train_rain_calibration.py --city "New York" --window-days 60
"""

from __future__ import annotations

import argparse
import logging

from src.config import resolve_config_path
from src.logging_setup import configure_logging
from src.rain_calibration import (
    IsotonicRainCalibrator,
    LogisticRainCalibrator,
    _rain_model_path,
)
from src.station_truth import (
    build_rain_training_set,
    ensure_data_directories,
    load_station_map,
)

_CONFIG_PATH = str(resolve_config_path())
log = configure_logging(
    "weather.train_rain",
    config_path=_CONFIG_PATH,
    log_filename="train_rain_calibration.log",
    level=logging.INFO,
)


def train_city(city: str, window_days: int = 90) -> dict:
    training = build_rain_training_set(city=city)
    training = training.dropna(subset=["raw_prob", "actual_wet_0_1"])
    if len(training) < 30:
        log.warning("%s: only %d usable rows; skipping", city, len(training))
        return {"city": city, "trained": False, "rows": len(training)}

    recent = training.tail(window_days).reset_index(drop=True)
    raw = recent["raw_prob"].astype(float).values
    outcomes = recent["actual_wet_0_1"].astype(int).values

    logistic = LogisticRainCalibrator(city=city)
    logistic.fit(raw, outcomes)
    logistic.save(_rain_model_path(city, "logistic"))

    logistic_preds = [logistic.predict(p) for p in raw]
    isotonic = IsotonicRainCalibrator(city=city)
    isotonic.fit(logistic_preds, outcomes)
    isotonic.save(_rain_model_path(city, "isotonic"))

    log.info("%s: trained logistic+isotonic on %d rows", city, len(recent))
    return {"city": city, "trained": True, "rows": len(recent)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Weather Signals rain calibration models")
    parser.add_argument("--city", help="Single city (default: all in stations.json)")
    parser.add_argument("--window-days", type=int, default=90)
    args = parser.parse_args()

    ensure_data_directories()
    cities = [args.city] if args.city else list(load_station_map().keys())
    trained = 0
    skipped = 0
    for city in cities:
        result = train_city(city, window_days=args.window_days)
        if result.get("trained"):
            trained += 1
        else:
            skipped += 1

    log.info(
        "Rain calibration training complete: %d trained, %d skipped (of %d cities)",
        trained, skipped, len(cities),
    )


if __name__ == "__main__":
    main()
