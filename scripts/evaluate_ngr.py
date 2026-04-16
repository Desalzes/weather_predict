"""Evaluate NGR calibration on holdout data.

Loads trained NGR models and the per-city training sets, splits chronologically,
and reports CRPS / MAE / Brier on the holdout window.

Usage:
    python scripts/evaluate_ngr.py
    python scripts/evaluate_ngr.py --days 400 --holdout-days 30 --city Austin
    python scripts/evaluate_ngr.py --output data/evaluation_reports/ngr_holdout.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_app_config, resolve_config_path
from src.ngr import NGRCalibrator, build_ngr_features, gaussian_crps
from src.station_truth import (
    CALIBRATION_MODELS_DIR,
    FORECAST_ARCHIVE_DIR,
    STATION_ACTUALS_DIR,
    _slugify_city,
    build_training_set,
    load_station_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
log = logging.getLogger("weather.evaluate_ngr")


def _split_chronologically(df: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "date" not in df.columns:
        return df, df.iloc[0:0]
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"])
    cutoff = df["_date"].max() - timedelta(days=holdout_days)
    train = df.loc[df["_date"] <= cutoff].drop(columns=["_date"])
    holdout = df.loc[df["_date"] > cutoff].drop(columns=["_date"])
    return train, holdout


def _evaluate_pair(
    model: NGRCalibrator,
    holdout_df: pd.DataFrame,
    market_type: str,
) -> dict:
    feats = build_ngr_features(holdout_df, market_type)
    if feats.empty:
        return {"pair": f"{model.city} {market_type}", "n": 0, "available": False}

    mus = np.empty(len(feats))
    sigmas = np.empty(len(feats))
    for i, row in enumerate(feats.itertuples(index=False)):
        mu, sigma = model.predict(row.forecast_f, row.spread_f, row.lead_h, int(row.doy))
        mus[i] = mu
        sigmas[i] = sigma

    y = feats["actual_f"].to_numpy(dtype=float)
    forecast_raw = feats["forecast_f"].to_numpy(dtype=float)

    crps_ngr = float(np.mean(gaussian_crps(mus, sigmas, y)))
    crps_raw = float(np.mean(gaussian_crps(forecast_raw, np.maximum(feats["spread_f"].to_numpy(), 1.0), y)))

    mae_ngr = float(np.mean(np.abs(y - mus)))
    mae_raw = float(np.mean(np.abs(y - forecast_raw)))

    # Brier on "above median forecast" as a simple stable threshold
    threshold = float(np.median(forecast_raw))
    prob_ngr = 1.0 - norm.cdf(threshold, loc=mus, scale=sigmas)
    prob_raw = 1.0 - norm.cdf(threshold, loc=forecast_raw, scale=np.maximum(feats["spread_f"].to_numpy(), 1.0))
    outcome = (y > threshold).astype(float)
    brier_ngr = float(np.mean((prob_ngr - outcome) ** 2))
    brier_raw = float(np.mean((prob_raw - outcome) ** 2))

    return {
        "pair": f"{model.city} {market_type}",
        "n": int(len(feats)),
        "available": True,
        "crps_ngr": crps_ngr,
        "crps_raw": crps_raw,
        "mae_ngr": mae_ngr,
        "mae_raw": mae_raw,
        "brier_ngr": brier_ngr,
        "brier_raw": brier_raw,
        "crps_delta": crps_ngr - crps_raw,
        "brier_delta": brier_ngr - brier_raw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate NGR calibration on holdout")
    parser.add_argument("--config", default=str(resolve_config_path()))
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument("--city", help="Evaluate one city only")
    parser.add_argument("--model-dir", default=str(CALIBRATION_MODELS_DIR))
    parser.add_argument("--station-actuals-dir", default=str(STATION_ACTUALS_DIR))
    parser.add_argument("--forecast-archive-dir", default=str(FORECAST_ARCHIVE_DIR))
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    load_app_config(args.config)  # validate config but we don't need anything from it here

    model_dir = Path(args.model_dir)
    station_map = load_station_map()
    cities = [args.city] if args.city else list(station_map.keys())

    results = []
    for city in cities:
        training_df = build_training_set(
            city,
            days=args.days,
            station_actuals_dir=Path(args.station_actuals_dir),
            forecast_archive_dir=Path(args.forecast_archive_dir),
        )
        if training_df.empty:
            log.info("%s: no training data", city)
            continue

        _, holdout = _split_chronologically(training_df, args.holdout_days)
        if holdout.empty:
            log.info("%s: no holdout rows", city)
            continue

        for market_type in ("high", "low"):
            ngr_path = model_dir / f"{_slugify_city(city)}_{market_type}_ngr.pkl"
            if not ngr_path.exists():
                log.info("%s %s: no NGR model", city, market_type)
                continue
            model = NGRCalibrator.load(ngr_path)
            pair_result = _evaluate_pair(model, holdout, market_type)
            results.append(pair_result)
            if pair_result.get("available"):
                log.info(
                    "%s: n=%d CRPS ngr=%.3f raw=%.3f (delta=%+.3f) Brier ngr=%.4f raw=%.4f",
                    pair_result["pair"],
                    pair_result["n"],
                    pair_result["crps_ngr"],
                    pair_result["crps_raw"],
                    pair_result["crps_delta"],
                    pair_result["brier_ngr"],
                    pair_result["brier_raw"],
                )

    # Aggregate
    available = [r for r in results if r.get("available")]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pairs_evaluated": len(available),
        "pairs_better_crps": sum(1 for r in available if r["crps_delta"] < 0),
        "pairs_better_brier": sum(1 for r in available if r["brier_delta"] < 0),
        "mean_crps_ngr": float(np.mean([r["crps_ngr"] for r in available])) if available else None,
        "mean_crps_raw": float(np.mean([r["crps_raw"] for r in available])) if available else None,
        "mean_brier_ngr": float(np.mean([r["brier_ngr"] for r in available])) if available else None,
        "mean_brier_raw": float(np.mean([r["brier_raw"] for r in available])) if available else None,
        "per_pair": results,
    }

    log.info(
        "Summary: %d/%d pairs better CRPS, %d better Brier; mean CRPS ngr=%.3f raw=%.3f",
        summary["pairs_better_crps"],
        summary["pairs_evaluated"],
        summary["pairs_better_brier"],
        summary["mean_crps_ngr"] or 0.0,
        summary["mean_crps_raw"] or 0.0,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Wrote report to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
