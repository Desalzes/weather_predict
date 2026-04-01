"""
Fetch active weather markets from Kalshi using the local KalshiClient.
"""

import logging
import re
import time
from typing import Optional

import requests

logger = logging.getLogger("weather.kalshi")

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_CACHE_TTL = 300
_cache: dict = {"markets": None, "ts": 0.0}

KALSHI_CITY_CODES = {
    "NYC": "New York",
    "LAX": "Los Angeles",
    "CHI": "Chicago",
    "MIA": "Miami",
    "HOU": "Houston",
    "PHX": "Phoenix",
    "DEN": "Denver",
    "SEA": "Seattle",
    "PHL": "Philadelphia",
    "DFW": "Dallas",
    "DAL": "Dallas",
    "AUS": "Austin",
    "BOS": "Boston",
    "ATL": "Atlanta",
    "DCA": "Washington DC",
    "DC": "Washington DC",
    "SFO": "San Francisco",
    "LAS": "Las Vegas",
    "MSP": "Minneapolis",
    "MIN": "Minneapolis",
    "SAT": "San Antonio",
    "SATX": "San Antonio",
    "MSY": "New Orleans",
    "NOLA": "New Orleans",
    "OKC": "Oklahoma City",
}

_WEATHER_PREFIXES = ("KXHIGHT", "KXLOWT", "KXRAIN", "KXSNOW")


def _get_client():
    """Get the local KalshiClient for authenticated requests."""
    try:
        from src.kalshi_client import KalshiClient
        return KalshiClient()
    except Exception as exc:
        logger.debug("Could not load KalshiClient: %s", exc)
        return None


def _parse_temp_ticker(ticker: str) -> Optional[dict]:
    """Parse a Kalshi temperature ticker into components."""
    tk = ticker.upper()

    for prefix, mtype in [("KXHIGHT", "high"), ("KXLOWT", "low")]:
        if not tk.startswith(prefix):
            continue
        m = re.match(
            rf"{prefix}(\w{{2,4}})-(\d{{2}}[A-Z]{{3}}\d{{1,2}})-[TB](\d+\.?\d*)",
            tk,
        )
        if m:
            city_code = m.group(1)
            return {
                "type": mtype,
                "city_code": city_code,
                "city": KALSHI_CITY_CODES.get(city_code, city_code),
                "date_raw": m.group(2),
                "threshold": float(m.group(3)),
            }

    return None


def _fetch_via_client(client, pages: int, per_page: int) -> set:
    weather_tickers: set = set()
    cursor = None

    for page in range(pages):
        try:
            params = {"limit": per_page}
            if cursor:
                params["cursor"] = cursor

            resp = client.session.get(
                f"{client.base_url}/markets/trades", params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Client fetch failed on page %d: %s", page, exc)
            break

        for trade in data.get("trades", []):
            tk = trade.get("ticker", "")
            if any(tk.startswith(p) for p in _WEATHER_PREFIXES):
                weather_tickers.add(tk)

        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.2)

    return weather_tickers


def _fetch_via_requests(pages: int, per_page: int) -> set:
    weather_tickers: set = set()
    cursor = None

    for page in range(pages):
        params: dict = {"limit": per_page}
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(
                f"{KALSHI_BASE}/markets/trades", params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch trades page %d: %s", page, exc)
            break

        for trade in data.get("trades", []):
            tk = trade.get("ticker", "")
            if any(tk.startswith(p) for p in _WEATHER_PREFIXES):
                weather_tickers.add(tk)

        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.2)

    return weather_tickers


def _fetch_market_details(ticker: str, client=None) -> Optional[dict]:
    try:
        if client:
            resp = client.session.get(f"{client.base_url}/markets/{ticker}", timeout=10)
        else:
            resp = requests.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        return body.get("market", body)
    except Exception as exc:
        logger.warning("Failed to fetch market %s: %s", ticker, exc)
        return None


def fetch_weather_markets(pages: int = 5, per_page: int = 1000) -> list[dict]:
    """Discover active weather markets by scanning recent Kalshi trades."""
    now = time.monotonic()
    if _cache["markets"] is not None and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["markets"]

    logger.info("Discovering weather markets from recent Kalshi trades...")

    client = _get_client()
    if client:
        logger.info("Using authenticated KalshiClient")
        weather_tickers = _fetch_via_client(client, pages, per_page)
    else:
        logger.info("Falling back to unauthenticated requests")
        weather_tickers = _fetch_via_requests(pages, per_page)

    logger.info("Found %d unique weather tickers from trades", len(weather_tickers))

    markets = []
    for ticker in sorted(weather_tickers):
        mdata = _fetch_market_details(ticker, client)
        if not mdata:
            continue

        parsed = _parse_temp_ticker(ticker)

        last_price = float(mdata.get("last_price_dollars", 0) or mdata.get("last_price", 0) or 0)
        yes_ask = float(mdata.get("yes_ask_dollars", 0) or mdata.get("yes_ask", 0) or 0)
        yes_bid = float(mdata.get("yes_bid_dollars", 0) or mdata.get("yes_bid", 0) or 0)

        markets.append({
            "source": "kalshi",
            "ticker": ticker,
            "title": mdata.get("title", ""),
            "yes_sub_title": mdata.get("yes_sub_title", ""),
            "no_sub_title": mdata.get("no_sub_title", ""),
            "type": parsed["type"] if parsed else "unknown",
            "city": parsed["city"] if parsed else None,
            "city_code": parsed["city_code"] if parsed else None,
            "threshold": parsed["threshold"] if parsed else None,
            "last_price": last_price,
            "yes_ask": yes_ask,
            "yes_bid": yes_bid,
            "volume_24h": int(float(mdata.get("volume_24h_fp", 0) or mdata.get("volume_24h", 0) or 0)),
            "close_time": mdata.get("close_time", ""),
            "status": mdata.get("status", ""),
            "result": mdata.get("result", ""),
        })
        time.sleep(0.2)

    markets = [m for m in markets if m["status"] not in ("closed", "settled", "finalized")]
    markets.sort(key=lambda m: (m.get("city") or "", m.get("type") or "", m.get("threshold") or 0))

    logger.info("Fetched details for %d active weather markets", len(markets))
    _cache["markets"] = markets
    _cache["ts"] = time.monotonic()
    return markets


def group_markets_by_city(markets: list[dict]) -> dict[str, dict]:
    grouped: dict = {}
    for m in markets:
        city = m.get("city")
        mtype = m.get("type")
        if not city or not mtype:
            continue
        grouped.setdefault(city, {}).setdefault(mtype, []).append(m)

    for city_data in grouped.values():
        for mtype_list in city_data.values():
            mtype_list.sort(key=lambda x: x.get("threshold") or 0)

    return grouped
