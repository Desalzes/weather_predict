"""Chronological-holdout evaluation of tail calibration.

Per (city, market_type, direction, threshold) pair, splits the training set
chronologically (train = 1..N-holdout, holdout = last holdout_days), fits
tail + baselines on train, scores log-loss on the tail region of holdout.

Tail region = raw_prob < 0.25 or raw_prob > 0.75. Pairs qualify for unblock
only if the tail-calibrated log-loss strictly beats BOTH climatology and
raw-isotonic baselines on the tail-region holdout.

Output: data/evaluation_reports/tail_eval_YYYY-MM-DD.json with per-pair
results and qualifies_for_unblock bool.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import load_app_config, resolve_config_path
from src.rain_calibration import IsotonicRainCalibrator
from src.station_truth import load_station_map
from src.tail_calibration import (
    BucketDistributionalCalibrator,
    TailBinaryCalibrator,
)
from src.tail_training_data import (
    build_bucket_training_set,
    build_tail_training_set,
)

_CONFIG_PATH = str(resolve_config_path())

_DEFAULT_HIGH_THRESHOLDS = [60.0, 70.0, 80.0, 90.0]
_DEFAULT_LOW_THRESHOLDS = [20.0, 30.0, 40.0, 50.0]
_TAIL_LO = 0.25
_TAIL_HI = 0.75

# Minimum log-loss improvement over isotonic/climatology to qualify for
# unblock. Prevents noise qualifications where the tail calibrator's
# logistic stage degenerates to approximate identity — seen in the first
# scorecard where ~60% of "qualifying" pairs had dIso < 1e-4 nats.
# Threshold chosen at 1e-3 nats (~0.1% relative log-loss improvement).
_MIN_IMPROVEMENT_NATS = 1e-3


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


def _tail_mask(raw_probs: np.ndarray) -> np.ndarray:
    return (raw_probs < _TAIL_LO) | (raw_probs > _TAIL_HI)


def evaluate_threshold_pair(
    city: str, market_type: str, direction: str, threshold: float,
    days: int, holdout_days: int, uncertainty_std_f: float,
) -> dict:
    df = build_tail_training_set(
        city=city, market_type=market_type,
        direction=direction, threshold=threshold,
        uncertainty_std_f=uncertainty_std_f,
    ).dropna(subset=["raw_prob", "actual_exceeded_0_1"])
    df = df.tail(days).reset_index(drop=True)
    if len(df) < holdout_days + 20:
        return {
            "city": city, "market_type": market_type, "direction": direction,
            "threshold": threshold,
            "status": "insufficient_data",
            "rows": int(len(df)),
        }

    train = df.iloc[:-holdout_days]
    holdout = df.iloc[-holdout_days:]

    raw = holdout["raw_prob"].astype(float).values
    y = holdout["actual_exceeded_0_1"].astype(int).values

    # Baseline: existing isotonic on raw probs (no tail chain)
    existing_iso = IsotonicRainCalibrator(city=city)
    existing_iso.fit(train["raw_prob"].values, train["actual_exceeded_0_1"].astype(int).values)
    iso_preds = np.array([existing_iso.predict(p) for p in raw])

    # Tail chain
    tail_cal = TailBinaryCalibrator(city=city, market_type=market_type, direction=direction)
    tail_cal.fit(train["raw_prob"].values, train["actual_exceeded_0_1"].astype(int).values)
    tail_preds = np.array([tail_cal.predict(p) for p in raw])

    climatology = float(train["actual_exceeded_0_1"].astype(int).mean()) * np.ones_like(y, dtype=float)

    tail_idx = _tail_mask(raw)
    if tail_idx.sum() == 0:
        return {
            "city": city, "market_type": market_type, "direction": direction,
            "threshold": threshold,
            "status": "no_tail_rows_in_holdout",
            "rows": int(len(holdout)),
        }

    ll_raw = _log_loss(raw[tail_idx], y[tail_idx])
    ll_iso = _log_loss(iso_preds[tail_idx], y[tail_idx])
    ll_tail = _log_loss(tail_preds[tail_idx], y[tail_idx])
    ll_clim = _log_loss(climatology[tail_idx], y[tail_idx])

    qualifies = (
        ll_tail < ll_clim - _MIN_IMPROVEMENT_NATS
        and ll_tail < ll_iso - _MIN_IMPROVEMENT_NATS
    )
    return {
        "city": city, "market_type": market_type, "direction": direction,
        "threshold": threshold, "status": "ok",
        "tail_region_n": int(tail_idx.sum()),
        "log_loss_raw": ll_raw,
        "log_loss_isotonic": ll_iso,
        "log_loss_tail": ll_tail,
        "log_loss_climatology": ll_clim,
        "qualifies_for_unblock": bool(qualifies),
    }


def evaluate_bucket_pair(
    city: str, market_type: str, bucket_low: float, bucket_high: float,
    days: int, holdout_days: int, uncertainty_std_f: float,
) -> dict:
    df = build_bucket_training_set(
        city=city, market_type=market_type,
        bucket_low=bucket_low, bucket_high=bucket_high,
        uncertainty_std_f=uncertainty_std_f,
    ).dropna(subset=["raw_bucket_prob", "actual_in_bucket_0_1"])
    df = df.tail(days).reset_index(drop=True)
    if len(df) < holdout_days + 20:
        return {
            "city": city, "market_type": market_type,
            "bucket_low": bucket_low, "bucket_high": bucket_high,
            "status": "insufficient_data",
            "rows": int(len(df)),
        }

    train = df.iloc[:-holdout_days]
    holdout = df.iloc[-holdout_days:]
    raw = holdout["raw_bucket_prob"].astype(float).values
    y = holdout["actual_in_bucket_0_1"].astype(int).values

    existing_iso = IsotonicRainCalibrator(city=city)
    existing_iso.fit(train["raw_bucket_prob"].values, train["actual_in_bucket_0_1"].astype(int).values)
    iso_preds = np.array([existing_iso.predict(p) for p in raw])

    bucket_cal = BucketDistributionalCalibrator(city=city, market_type=market_type)
    bucket_cal.fit(train["raw_bucket_prob"].values, train["actual_in_bucket_0_1"].astype(int).values)
    bucket_preds = np.array([bucket_cal.predict(p) for p in raw])

    climatology = float(train["actual_in_bucket_0_1"].astype(int).mean()) * np.ones_like(y, dtype=float)

    ll_raw = _log_loss(raw, y)
    ll_iso = _log_loss(iso_preds, y)
    ll_bucket = _log_loss(bucket_preds, y)
    ll_clim = _log_loss(climatology, y)

    qualifies = (
        ll_bucket < ll_clim - _MIN_IMPROVEMENT_NATS
        and ll_bucket < ll_iso - _MIN_IMPROVEMENT_NATS
    )
    return {
        "city": city, "market_type": market_type,
        "bucket_low": bucket_low, "bucket_high": bucket_high,
        "status": "ok",
        "rows": int(len(holdout)),
        "log_loss_raw": ll_raw,
        "log_loss_isotonic": ll_iso,
        "log_loss_tail": ll_bucket,
        "log_loss_climatology": ll_clim,
        "qualifies_for_unblock": bool(qualifies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--holdout-days", type=int, default=30)
    args = parser.parse_args()

    config = load_app_config(_CONFIG_PATH)
    uncertainty_std_f = float(config.get("uncertainty_std_f", 2.0))

    cities = list(load_station_map().keys())
    threshold_results = []
    bucket_results = []

    for city in cities:
        for mtype, thresholds in (
            ("high", _DEFAULT_HIGH_THRESHOLDS),
            ("low", _DEFAULT_LOW_THRESHOLDS),
        ):
            for threshold in thresholds:
                for direction in ("above", "below"):
                    threshold_results.append(
                        evaluate_threshold_pair(
                            city, mtype, direction, threshold,
                            days=args.days, holdout_days=args.holdout_days,
                            uncertainty_std_f=uncertainty_std_f,
                        )
                    )
                bucket_results.append(
                    evaluate_bucket_pair(
                        city, mtype, threshold - 0.5, threshold + 0.5,
                        days=args.days, holdout_days=args.holdout_days,
                        uncertainty_std_f=uncertainty_std_f,
                    )
                )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {"days": args.days, "holdout_days": args.holdout_days,
                 "uncertainty_std_f": uncertainty_std_f},
        "threshold_pairs": threshold_results,
        "bucket_pairs": bucket_results,
    }
    out = (
        Path("data/evaluation_reports")
        / f"tail_eval_{datetime.now(timezone.utc).date().isoformat()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")

    qualifying = [r for r in threshold_results if r.get("qualifies_for_unblock")]
    qualifying += [r for r in bucket_results if r.get("qualifies_for_unblock")]
    print(f"qualifying pairs: {len(qualifying)}")


if __name__ == "__main__":
    main()
