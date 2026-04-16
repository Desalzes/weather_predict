"""Standalone settlement pass for the opportunity archive.

Loads station actuals per city from data/station_actuals and fills
yes_outcome / actual_value_f for archive rows whose market_date has
truth data. Safe to run repeatedly (idempotent).

Usage:
    python scripts/settle_opportunity_archive.py
    python scripts/settle_opportunity_archive.py --archive-dir data/opportunity_archive
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_app_config, resolve_config_path
from src.opportunity_log import settle_opportunity_archive
from src.station_truth import STATION_ACTUALS_DIR, load_station_map, _slugify_city

_CONFIG_PATH = resolve_config_path()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("weather.settle_opportunity_archive")


def load_station_actuals(station_actuals_dir: Path) -> dict:
    """Return {city_name: DataFrame[date, tmax_f, tmin_f]} for all known cities."""
    station_map = load_station_map()
    result = {}
    for city in station_map.keys():
        path = station_actuals_dir / f"{_slugify_city(city)}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if {"date", "tmax_f", "tmin_f"}.issubset(df.columns):
                result[city] = df
        except Exception as exc:
            log.warning("Could not load station actuals for %s: %s", city, exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Settle opportunity archive against station actuals")
    parser.add_argument("--config", default=str(_CONFIG_PATH))
    parser.add_argument("--archive-dir", help="Override archive directory")
    parser.add_argument("--station-actuals-dir", help="Override station actuals directory")
    args = parser.parse_args()

    config = load_app_config(args.config)
    archive_dir = Path(args.archive_dir or config.get("opportunity_archive_dir", "data/opportunity_archive"))
    actuals_dir = Path(args.station_actuals_dir or STATION_ACTUALS_DIR)

    if not archive_dir.exists():
        log.info("No archive directory at %s; nothing to settle", archive_dir)
        return 0

    actuals = load_station_actuals(actuals_dir)
    log.info("Loaded station actuals for %d cities", len(actuals))

    count = settle_opportunity_archive(archive_dir=archive_dir, station_actuals=actuals)
    log.info("Settled %d archive rows", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
