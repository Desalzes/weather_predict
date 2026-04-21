"""One-shot smoke test: force enable_rain_vertical=true and run one scan.

Runs the full scanner pipeline with the rain vertical flipped on, paper-
trading off so we don't pollute the ledger. Proves wiring is correct:
fetch → calibrate → match → policy-filter produces valid opportunity dicts.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\smoke_test_rain_vertical.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import load_config, run_scan  # noqa: E402


def main() -> None:
    config = load_config()
    config["enable_rain_vertical"] = True
    config["rain_watchlist"] = ["New York"]
    # Keep paper-trading off so the smoke test does not write ledger rows.
    config["enable_paper_trading"] = False
    # Keep DeepSeek off so we don't burn a review budget on a smoke run.
    config["enable_deepseek_worker"] = False

    result = run_scan(config=config)

    rain_opps = [o for o in result["opportunities"] if o.get("market_category") == "rain"]
    temp_opps = [o for o in result["opportunities"] if o.get("market_category") != "rain"]
    print()
    print(f"total opportunities: {len(result['opportunities'])}")
    print(f"  temperature:       {len(temp_opps)}")
    print(f"  rain:              {len(rain_opps)}")
    print()
    for o in rain_opps[:5]:
        ticker = o.get("ticker", "?")
        city = o.get("city", "?")
        edge = o.get("edge")
        our_p = o.get("our_probability")
        mkt_p = o.get("market_price")
        print(
            f"  {ticker:<30} {city:<14} "
            f"our={our_p:.3f} mkt={mkt_p:.3f} edge={edge:+.3f}"
        )


if __name__ == "__main__":
    main()
