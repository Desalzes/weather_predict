"""Append every scored opportunity to a daily CSV archive for post-hoc evaluation.

This is write-only — it does NOT feed back into trade selection. It exists
so calibration can be evaluated on hundreds of labeled (prob, outcome) pairs
per day instead of the 0-3 from actual paper trades.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("weather.opportunity_log")

DEFAULT_ARCHIVE_DIR = Path("data/opportunity_archive")

OPPORTUNITY_ARCHIVE_COLUMNS = [
    "scan_id",
    "recorded_at_utc",
    "source",
    "ticker",
    "city",
    "market_type",
    "market_date",
    "outcome",
    "our_probability",
    "raw_probability",
    "market_price",
    "edge",
    "abs_edge",
    "forecast_value_f",
    "raw_forecast_value_f",
    "uncertainty_std_f",
    "forecast_blend_source",
    "forecast_calibration_source",
    "probability_calibration_source",
    "hours_to_settlement",
    "volume24hr",
    "yes_outcome",
    "actual_value_f",
    "settled_at_utc",
]


def _archive_path_for_date(archive_dir: Path, date_str: str) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir / f"{date_str}.csv"


def _row_from_opportunity(scan_id: str, opp: dict) -> dict:
    return {
        "scan_id": scan_id,
        "recorded_at_utc": scan_id,
        "source": opp.get("source"),
        "ticker": opp.get("ticker"),
        "city": opp.get("city"),
        "market_type": opp.get("market_type"),
        "market_date": opp.get("market_date"),
        "outcome": opp.get("outcome"),
        "our_probability": opp.get("our_probability"),
        "raw_probability": opp.get("raw_probability"),
        "market_price": opp.get("market_price"),
        "edge": opp.get("edge"),
        "abs_edge": opp.get("abs_edge"),
        "forecast_value_f": opp.get("forecast_value_f"),
        "raw_forecast_value_f": opp.get("raw_forecast_value_f"),
        "uncertainty_std_f": opp.get("uncertainty_std_f"),
        "forecast_blend_source": opp.get("forecast_blend_source"),
        "forecast_calibration_source": opp.get("forecast_calibration_source"),
        "probability_calibration_source": opp.get("probability_calibration_source"),
        "hours_to_settlement": opp.get("hours_to_settlement"),
        "volume24hr": opp.get("volume24hr"),
        "yes_outcome": None,
        "actual_value_f": None,
        "settled_at_utc": None,
    }


def log_opportunities(
    scan_id: str,
    opportunities: list[dict],
    archive_dir: Path | str = DEFAULT_ARCHIVE_DIR,
) -> Optional[Path]:
    if not opportunities:
        return None
    archive_dir = Path(archive_dir)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    path = _archive_path_for_date(archive_dir, today)

    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OPPORTUNITY_ARCHIVE_COLUMNS)
        if write_header:
            writer.writeheader()
        for opp in opportunities:
            writer.writerow(_row_from_opportunity(scan_id, opp))
    logger.info("Logged %d opportunities to %s", len(opportunities), path)
    return path
