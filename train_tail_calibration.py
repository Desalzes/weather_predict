"""Train per-pair tail calibration models.

Pairs:
  - threshold: every (city, market_type, direction) where the training set
    has >= 30 tail-region rows (raw_prob < 0.25 or > 0.75) and >= 2 actual
    tail events on each side of the raw_prob distribution.
  - bucket: every (city, market_type) with >= 30 training rows and >= 2
    in-bucket actuals over the window.

For the P2 initial rollout we use a single representative threshold per
(city, market_type) - the median of a default threshold grid per market_type.
Future work would sweep observed Kalshi market thresholds; the median is
sufficient to produce one calibrator per (city, market_type, direction).

Output artifacts in data/calibration_models/ (filenames set by the calibrator
classes' save()):
  {slug}_{mtype}_{direction}_tail_logistic.pkl
  {slug}_{mtype}_{direction}_tail_isotonic.pkl
  {slug}_{mtype}_{direction}_tail_meta.pkl
  {slug}_{mtype}_bucket_logistic.pkl
  {slug}_{mtype}_bucket_isotonic.pkl
  {slug}_{mtype}_bucket_meta.pkl
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_app_config, resolve_config_path
from src.logging_setup import configure_logging
from src.station_truth import (
    CALIBRATION_MODELS_DIR,
    _slugify_city,
    ensure_data_directories,
    load_station_map,
)
from src.tail_calibration import (
    BucketDistributionalCalibrator,
    TailBinaryCalibrator,
)
from src.tail_training_data import (
    build_bucket_training_set,
    build_tail_training_set,
)

_CONFIG_PATH = str(resolve_config_path())
log = configure_logging(
    "weather.train_tail",
    config_path=_CONFIG_PATH,
    log_filename="train_tail_calibration.log",
    level=logging.INFO,
)

# Default threshold sweep per market_type (degrees Fahrenheit). Kalshi markets
# typically post thresholds within ~15 F of the climatological mean. We take
# the median of each grid as the representative threshold for P2 initial
# rollout - one fit per (city, market_type, direction).
_DEFAULT_HIGH_THRESHOLDS = list(range(40, 105, 5))  # 40, 45, ..., 100
_DEFAULT_LOW_THRESHOLDS = list(range(10, 80, 5))    # 10, 15, ..., 75


def _tail_region_counts(df) -> tuple[int, int, int]:
    """Return (tail_region_total, tail_events_low, tail_events_high)."""
    if df.empty:
        return 0, 0, 0
    low_mask = df["raw_prob"] < 0.25
    high_mask = df["raw_prob"] > 0.75
    tail_mask = low_mask | high_mask
    tail_events_low = int(df.loc[low_mask, "actual_exceeded_0_1"].sum())
    tail_events_high = int(df.loc[high_mask, "actual_exceeded_0_1"].sum())
    return int(tail_mask.sum()), tail_events_low, tail_events_high


def train_threshold(
    city: str,
    market_type: str,
    direction: str,
    threshold: float,
    window_days: int,
    uncertainty_std_f: float,
) -> dict:
    df = build_tail_training_set(
        city=city, market_type=market_type,
        direction=direction, threshold=threshold,
        uncertainty_std_f=uncertainty_std_f,
    )
    if df.empty:
        return {"status": "no_data"}

    df = df.tail(window_days).reset_index(drop=True)
    df = df.dropna(subset=["raw_prob", "actual_exceeded_0_1"])

    tail_total, tail_lo_events, tail_hi_events = _tail_region_counts(df)
    # Sample-size gate: >=30 tail-region rows, >=2 events on each side
    if tail_total < 30 or tail_lo_events < 2 or tail_hi_events < 2:
        return {
            "status": "insufficient_data",
            "tail_total": tail_total,
            "tail_lo_events": tail_lo_events,
            "tail_hi_events": tail_hi_events,
        }

    cal = TailBinaryCalibrator(city=city, market_type=market_type, direction=direction)
    cal.fit(df["raw_prob"].values, df["actual_exceeded_0_1"].astype(int).values)

    prefix = CALIBRATION_MODELS_DIR / f"{_slugify_city(city)}_{market_type}_{direction}_tail"
    cal.save(prefix)
    return {"status": "trained", "rows": int(len(df)), "threshold": threshold}


def train_bucket(
    city: str,
    market_type: str,
    bucket_low: float,
    bucket_high: float,
    window_days: int,
    uncertainty_std_f: float,
) -> dict:
    df = build_bucket_training_set(
        city=city, market_type=market_type,
        bucket_low=bucket_low, bucket_high=bucket_high,
        uncertainty_std_f=uncertainty_std_f,
    )
    if df.empty:
        return {"status": "no_data"}
    df = df.tail(window_days).reset_index(drop=True)
    df = df.dropna(subset=["raw_bucket_prob", "actual_in_bucket_0_1"])
    if len(df) < 30 or int(df["actual_in_bucket_0_1"].sum()) < 2:
        return {"status": "insufficient_data", "rows": int(len(df))}

    cal = BucketDistributionalCalibrator(city=city, market_type=market_type)
    cal.fit(df["raw_bucket_prob"].values, df["actual_in_bucket_0_1"].astype(int).values)

    prefix = CALIBRATION_MODELS_DIR / f"{_slugify_city(city)}_{market_type}_bucket"
    cal.save(prefix)
    return {"status": "trained", "rows": int(len(df))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", help="Single city (default: all in stations.json)")
    parser.add_argument("--window-days", type=int, default=180)
    args = parser.parse_args()

    config = load_app_config(_CONFIG_PATH)
    window = int(config.get("tail_calibration_window_days", args.window_days))
    # Mirror the matcher's serving default (src/matcher.py::match_kalshi_markets)
    # so archive rows with NaN ensemble sigma fall back to the same value the
    # live path would use — otherwise training silently drops those rows and
    # the calibrator sees a recency-biased 23% sample.
    uncertainty_std_f = float(config.get("uncertainty_std_f", 2.0))
    ensure_data_directories()

    cities = [args.city] if args.city else list(load_station_map().keys())
    for city in cities:
        for mtype, thresholds in (
            ("high", _DEFAULT_HIGH_THRESHOLDS),
            ("low", _DEFAULT_LOW_THRESHOLDS),
        ):
            # Median threshold for the representative fit
            threshold = float(thresholds[len(thresholds) // 2])
            for direction in ("above", "below"):
                result = train_threshold(
                    city, mtype, direction, threshold, window,
                    uncertainty_std_f,
                )
                log.info(
                    "%s %s %s@%.0fF tail: %s",
                    city, mtype, direction, threshold,
                    result.get("status"),
                )
            # 1F-wide bucket centered on the threshold
            bucket_result = train_bucket(
                city, mtype, threshold - 0.5, threshold + 0.5, window,
                uncertainty_std_f,
            )
            log.info(
                "%s %s bucket[%0.1f-%0.1f]: %s",
                city, mtype, threshold - 0.5, threshold + 0.5,
                bucket_result.get("status"),
            )


if __name__ == "__main__":
    main()
