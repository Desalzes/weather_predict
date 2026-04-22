"""
Fetch active weather markets from Kalshi using the local KalshiClient.
"""

import logging
import re
import time
from datetime import datetime
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


_RAIN_TICKER_PATTERN = re.compile(
    r"^KXRAIN([A-Z]{2,4})-(\d{2}[A-Z]{3}\d{1,2})-T0$",
    re.IGNORECASE,
)


def _city_to_codes_map() -> dict[str, list[str]]:
    """Reverse KALSHI_CITY_CODES into {city_name: [code1, code2, ...]}.

    Several cities have multiple aliases (DFW/DAL for Dallas, NOLA/MSY for
    New Orleans, etc.). Rain-series discovery only needs one hit per unique
    city — the first code in each list is used as the primary.
    """
    by_city: dict[str, list[str]] = {}
    for code, city in KALSHI_CITY_CODES.items():
        by_city.setdefault(city, []).append(code)
    return by_city


def _parse_rain_ticker_date(ticker: str) -> Optional[str]:
    """Parse a KXRAIN{CODE}-{YYMMMDD}-T0 ticker into ISO market_date (YYYY-MM-DD).

    Returns None if the ticker does not match the expected pattern or the
    date portion is malformed.
    """
    m = _RAIN_TICKER_PATTERN.match(ticker.strip())
    if not m:
        return None
    date_raw = m.group(2)
    try:
        parsed = datetime.strptime(date_raw.upper(), "%y%b%d")
    except ValueError:
        return None
    return parsed.strftime("%Y-%m-%d")


def _coerce_float_or_none(value) -> Optional[float]:
    """Preserve None/empty string as None; otherwise parse as float. Used to
    keep blank `yes_ask`/`yes_bid` on untraded markets from collapsing to 0."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_rain_market(mdata: dict, city: str) -> Optional[dict]:
    """Map a raw /markets response entry into the dict shape consumed by
    src.rain_matcher.match_kalshi_rain. Returns None if the ticker can't be
    parsed (logs at debug)."""
    ticker = str(mdata.get("ticker") or "")
    market_date = _parse_rain_ticker_date(ticker)
    if market_date is None:
        logger.debug("Skipping rain market with unparsable ticker %r", ticker)
        return None

    yes_ask = _coerce_float_or_none(
        mdata.get("yes_ask_dollars") if mdata.get("yes_ask_dollars") is not None else mdata.get("yes_ask")
    )
    yes_bid = _coerce_float_or_none(
        mdata.get("yes_bid_dollars") if mdata.get("yes_bid_dollars") is not None else mdata.get("yes_bid")
    )
    last_price = _coerce_float_or_none(
        mdata.get("last_price_dollars") if mdata.get("last_price_dollars") is not None else mdata.get("last_price")
    )

    volume_24h_raw = mdata.get("volume_24h_fp")
    if volume_24h_raw is None:
        volume_24h_raw = mdata.get("volume_24h")
    try:
        volume_24h = int(float(volume_24h_raw)) if volume_24h_raw not in (None, "") else 0
    except (TypeError, ValueError):
        volume_24h = 0

    return {
        "source": "kalshi",
        "ticker": ticker,
        "title": mdata.get("title", ""),
        "yes_sub_title": mdata.get("yes_sub_title", ""),
        "no_sub_title": mdata.get("no_sub_title", ""),
        "type": "rain",
        "market_type": "rain_binary",
        "market_category": "rain",
        "city": city,
        "market_date": market_date,
        "last_price": last_price,
        "yes_ask": yes_ask,
        "yes_bid": yes_bid,
        "volume_24h": volume_24h,
        "volume24hr": volume_24h,
        "close_time": mdata.get("close_time", ""),
        "status": mdata.get("status", ""),
        "result": mdata.get("result", ""),
    }


def _fetch_rain_markets_for_series(
    series_ticker: str,
    *,
    client,
    limit: int,
) -> list[dict]:
    """GET /markets?series_ticker={series}&status=open&limit={limit}.

    Returns a list of raw market dicts (empty list on HTTP errors or empty
    response). Uses the authenticated client session when available so we
    inherit its retry/throttle behavior; falls back to plain requests.
    """
    url = f"{KALSHI_BASE}/markets"
    params = {
        "series_ticker": series_ticker,
        "status": "open",
        "limit": limit,
    }
    try:
        if client is not None:
            resp = client.session.get(url, params=params, timeout=15)
        else:
            resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Rain market fetch failed for %s: %s", series_ticker, exc)
        return []
    except Exception as exc:  # pragma: no cover - defensive against session-level errors
        logger.warning("Rain market fetch failed for %s: %s", series_ticker, exc)
        return []

    try:
        body = resp.json()
    except ValueError as exc:
        logger.warning("Rain market response for %s not JSON: %s", series_ticker, exc)
        return []

    return body.get("markets") or []


def fetch_kalshi_rain_markets(
    cities: Optional[list[str]] = None,
    limit: int = 50,
) -> list[dict]:
    """Discover open Kalshi KXRAIN markets per-city via /markets?series_ticker=.

    The default market-discovery path (`fetch_weather_markets`) scans
    /markets/trades, which only returns markets that have traded recently.
    Many KXRAIN listings never trade, so they never surface there. This
    function hits /markets?series_ticker=KXRAIN{CODE}&status=open per city
    instead, so untraded rain listings still enter the rain matcher.

    Args:
        cities: Optional whitelist of city names (e.g. ["New York"]). When
            None, every unique city in KALSHI_CITY_CODES is queried. Unknown
            city names are silently skipped (no API call).
        limit: `limit` query-param passed to /markets. Defaults to 50 —
            rain series have <5 open markets each in practice so this is
            generous.

    Returns:
        List of market dicts in the shape consumed by
        src.rain_matcher.match_kalshi_rain (ticker, city, market_date,
        close_time, yes_ask, yes_bid, volume_24h, etc.). yes_ask is
        preserved as None when the API returns a blank value so downstream
        filtering still treats untraded markets as "no ask posted".
    """
    code_by_city = _city_to_codes_map()

    if cities is None:
        target_cities = sorted(code_by_city.keys())
    else:
        target_cities = []
        for name in cities:
            if name in code_by_city:
                target_cities.append(name)
            else:
                logger.debug("Skipping unknown rain city %r (no KALSHI code)", name)

    if not target_cities:
        return []

    client = _get_client()
    if client is not None:
        logger.debug("Rain market discovery using authenticated KalshiClient")
    else:
        logger.debug("Rain market discovery using unauthenticated requests")

    all_markets: list[dict] = []
    for city in target_cities:
        # Use the first (primary) code for each city — aliases would trigger
        # duplicate API calls for the same series.
        code = code_by_city[city][0]
        series_ticker = f"KXRAIN{code}"
        raw_markets = _fetch_rain_markets_for_series(
            series_ticker, client=client, limit=limit,
        )
        if not raw_markets:
            logger.debug("No open rain markets for %s (%s)", city, series_ticker)
        for raw in raw_markets:
            normalized = _normalize_rain_market(raw, city)
            if normalized is None:
                continue
            # Defensive: drop closed/settled even if the query said status=open
            if normalized.get("status") in ("closed", "settled", "finalized"):
                continue
            all_markets.append(normalized)
        time.sleep(0.2)

    logger.info(
        "Discovered %d open rain markets across %d city(ies).",
        len(all_markets),
        len(target_cities),
    )
    return all_markets


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
