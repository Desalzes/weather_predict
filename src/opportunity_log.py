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


def _parse_outcome_bounds(outcome: str) -> tuple[Optional[float], Optional[float]]:
    """Extract bounds from outcome labels like '>75°F', '<63°F', '66-67°F'."""
    if not outcome:
        return None, None
    cleaned = outcome.replace("°F", "").replace("°", "").strip()
    if cleaned.startswith(">"):
        return float(cleaned[1:]), None
    if cleaned.startswith("<"):
        return None, float(cleaned[1:])
    if "-" in cleaned:
        low_s, high_s = cleaned.split("-", 1)
        return float(low_s), float(high_s)
    return None, None


def _yes_outcome(low: Optional[float], high: Optional[float], actual_f: float, market_type: str) -> int:
    if low is not None and high is not None:
        return 1 if low <= actual_f <= high else 0
    if low is not None:
        return 1 if actual_f > low else 0   # threshold ">"
    if high is not None:
        return 1 if actual_f < high else 0  # threshold "<"
    return 0


def settle_opportunity_archive(
    archive_dir: Path | str,
    station_actuals: dict,
) -> int:
    """Join archive rows with station truth; fill yes_outcome/actual_value_f in-place.

    station_actuals: {city: pd.DataFrame with columns date, tmax_f, tmin_f}.
    Returns number of rows updated.
    """
    archive_dir = Path(archive_dir)
    if not archive_dir.exists():
        return 0

    now_iso = datetime.utcnow().isoformat() + "Z"
    total_updated = 0

    for csv_path in sorted(archive_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path, dtype={"yes_outcome": "object", "actual_value_f": "object", "settled_at_utc": "object"})
        if frame.empty:
            continue
        frame_updated = False

        for idx, row in frame.iterrows():
            if not pd.isna(row.get("yes_outcome")):
                continue
            city = row["city"]
            actuals_df = station_actuals.get(city)
            if actuals_df is None or actuals_df.empty:
                continue
            match = actuals_df.loc[actuals_df["date"].astype(str) == str(row["market_date"])]
            if match.empty:
                continue
            actual_col = "tmax_f" if row["market_type"] == "high" else "tmin_f"
            actual_val = match.iloc[0][actual_col]
            if pd.isna(actual_val):
                continue
            low, high = _parse_outcome_bounds(str(row["outcome"]))
            if low is None and high is None:
                continue
            yo = _yes_outcome(low, high, float(actual_val), row["market_type"])
            frame.at[idx, "yes_outcome"] = yo
            frame.at[idx, "actual_value_f"] = float(actual_val)
            frame.at[idx, "settled_at_utc"] = now_iso
            frame_updated = True
            total_updated += 1

        if frame_updated:
            frame.to_csv(csv_path, index=False)

    logger.info("Settled %d archived opportunities across %s", total_updated, archive_dir)
    return total_updated
