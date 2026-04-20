"""One-off discovery: dump Kalshi KXRAIN market inventory to CSV.

Output: data/rain_market_inventory.csv (ticker, market_question, city,
market_date, outcome, close_time, yes_ask, volume_24h). Drives rain-policy
threshold choices and rain_matcher parser shape.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fetch_kalshi import fetch_weather_markets  # noqa: E402


def main() -> None:
    markets = fetch_weather_markets(pages=5)
    rain = [m for m in markets if str(m.get("ticker", "")).startswith("KXRAIN")]
    out = Path("data/rain_market_inventory.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ticker",
        "market_question",
        "city",
        "market_date",
        "outcome",
        "close_time",
        "yes_ask",
        "volume_24h",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for m in rain:
            w.writerow({k: m.get(k) for k in fields})
    print(f"wrote {len(rain)} rows to {out}")


if __name__ == "__main__":
    main()
