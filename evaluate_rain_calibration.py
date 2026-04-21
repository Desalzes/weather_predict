"""Chronological-holdout evaluation of rain calibration.

Fits logistic + isotonic on a chronological training split for each city,
then scores calibrated, raw, and climatology probabilities on the held-out
tail window. Brier score and log-loss are reported; a JSON scorecard is
written to ``data/evaluation_reports/``.

Usage:
    python evaluate_rain_calibration.py
    python evaluate_rain_calibration.py --days 400 --holdout-days 30
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.rain_calibration import IsotonicRainCalibrator, LogisticRainCalibrator
from src.station_truth import build_rain_training_set, load_station_map


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


def evaluate_city(city: str, days: int, holdout_days: int) -> dict:
    training = build_rain_training_set(city=city).dropna(
        subset=["raw_prob", "actual_wet_0_1"]
    )
    training = training.tail(days).reset_index(drop=True)
    if len(training) < holdout_days + 20:
        return {"city": city, "usable_rows": int(len(training)), "status": "insufficient_data"}

    train_slice = training.iloc[:-holdout_days]
    holdout = training.iloc[-holdout_days:]

    logistic = LogisticRainCalibrator(city=city)
    logistic.fit(train_slice["raw_prob"].values, train_slice["actual_wet_0_1"].astype(int).values)
    logistic_preds_train = np.array([logistic.predict(p) for p in train_slice["raw_prob"].values])
    isotonic = IsotonicRainCalibrator(city=city)
    isotonic.fit(logistic_preds_train, train_slice["actual_wet_0_1"].astype(int).values)

    y = holdout["actual_wet_0_1"].astype(int).values
    raw = holdout["raw_prob"].astype(float).values
    cal = np.array([isotonic.predict(logistic.predict(p)) for p in raw])
    climatology = float(train_slice["actual_wet_0_1"].astype(int).mean()) * np.ones_like(y, dtype=float)

    return {
        "city": city,
        "status": "ok",
        "holdout_n": int(len(y)),
        "train_n": int(len(train_slice)),
        "brier_raw": _brier(raw, y),
        "brier_calibrated": _brier(cal, y),
        "brier_climatology": _brier(climatology, y),
        "log_loss_raw": _log_loss(raw, y),
        "log_loss_calibrated": _log_loss(cal, y),
        "log_loss_climatology": _log_loss(climatology, y),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological-holdout evaluation for rain calibration")
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--holdout-days", type=int, default=30)
    args = parser.parse_args()

    cities = list(load_station_map().keys())
    per_city = [evaluate_city(c, days=args.days, holdout_days=args.holdout_days) for c in cities]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {"days": args.days, "holdout_days": args.holdout_days},
        "cities": per_city,
    }
    out = Path("data/evaluation_reports") / f"rain_eval_{datetime.now(timezone.utc).date().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
