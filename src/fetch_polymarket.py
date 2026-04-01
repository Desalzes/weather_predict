"""
Fetch weather markets from Polymarket's Gamma API.
"""

import json
import logging
import re
import time
from typing import Optional

import requests

logger = logging.getLogger("weather.polymarket")

_GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
_CACHE_TTL_SECONDS = 300
_cache: dict = {"markets": None, "fetched_at": 0.0}

CITY_ALIASES = {
    "nyc": "New York",
    "new york": "New York",
    "new york city": "New York",
    "la": "Los Angeles",
    "los angeles": "Los Angeles",
    "chicago": "Chicago",
    "miami": "Miami",
    "houston": "Houston",
    "phoenix": "Phoenix",
    "denver": "Denver",
    "seattle": "Seattle",
    "philadelphia": "Philadelphia",
    "dallas": "Dallas",
    "austin": "Austin",
    "boston": "Boston",
    "atlanta": "Atlanta",
    "washington dc": "Washington DC",
    "san francisco": "San Francisco",
    "las vegas": "Las Vegas",
    "minneapolis": "Minneapolis",
    "san antonio": "San Antonio",
    "new orleans": "New Orleans",
    "oklahoma city": "Oklahoma City",
}


def _parse_market_type(question: str) -> dict:
    q = question.lower().strip()
    result = {"type": "unknown", "city": None, "raw_city": None}

    m = re.search(r"highest temperature in (.+?)(?:\s+today|\s+tomorrow|\?)", q)
    if m:
        result["type"] = "daily_high"
        result["raw_city"] = m.group(1).strip()
        result["city"] = CITY_ALIASES.get(result["raw_city"].lower(), result["raw_city"].title())
        return result

    m = re.search(r"lowest temperature in (.+?)(?:\s+today|\s+tomorrow|\?)", q)
    if m:
        result["type"] = "daily_low"
        result["raw_city"] = m.group(1).strip()
        result["city"] = CITY_ALIASES.get(result["raw_city"].lower(), result["raw_city"].title())
        return result

    m = re.search(r"(?:will it )?rain in (.+?)(?:\s+today|\?)", q)
    if m:
        result["type"] = "rain_today"
        result["raw_city"] = m.group(1).strip()
        result["city"] = CITY_ALIASES.get(result["raw_city"].lower(), result["raw_city"].title())
        return result

    m = re.search(r"rain in (.+?)\s+this month", q)
    if m:
        result["type"] = "rain_month"
        result["raw_city"] = m.group(1).strip()
        result["city"] = CITY_ALIASES.get(result["raw_city"].lower(), result["raw_city"].title())
        return result

    if "hottest" in q or "coldest" in q or "climate" in q:
        result["type"] = "climate"
        return result

    return result


def _parse_outcome_range(outcome: str) -> tuple[Optional[float], Optional[float]]:
    """Parse a Polymarket temperature outcome string into (low, high) range in deg F."""
    outcome = outcome.strip()

    m = re.match(r"(\d+)°?\s*(?:or above|\+|or higher)", outcome, re.IGNORECASE)
    if m:
        return (float(m.group(1)), None)

    m = re.match(r"(\d+)°?\s*(?:or below|or lower)", outcome, re.IGNORECASE)
    if m:
        return (None, float(m.group(1)) + 0.99)

    m = re.match(r"(\d+)°?\s*to\s*(\d+)°?", outcome, re.IGNORECASE)
    if m:
        return (float(m.group(1)), float(m.group(2)) + 0.99)

    return (None, None)


def fetch_weather_markets() -> list[dict]:
    """Fetch weather-related markets from Polymarket Gamma API."""
    now = time.monotonic()
    if _cache["markets"] is not None and (now - _cache["fetched_at"]) < _CACHE_TTL_SECONDS:
        return _cache["markets"]

    logger.info("Fetching weather markets from Gamma API")
    weather_markets = []
    tag_searches = [
        {"tag_slug": "daily-temperature"},
        {"tag_slug": "high-temperature"},
        {"tag_slug": "low-temperature"},
        {"tag_slug": "snow-and-rain"},
        {"tag_slug": "climate-change"},
        {"tag_slug": "climate"},
    ]
    text_searches = [
        "highest temperature",
        "lowest temperature",
        "rain today",
        "hottest",
    ]
    seen_ids = set()

    for params_extra in tag_searches:
        try:
            params = {"closed": "false", "limit": 100, "order": "volume24hr", "ascending": "false"}
            params.update(params_extra)
            resp = requests.get(_GAMMA_MARKETS_URL, params=params, timeout=15)
            resp.raise_for_status()
            markets = resp.json()
        except requests.RequestException as exc:
            logger.warning("Gamma API tag search %s failed: %s", params_extra, exc)
            continue

        for m in markets:
            mid = m.get("id") or m.get("conditionId", "")
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            parsed = _parse_market_type(m.get("question", ""))
            if parsed["type"] == "unknown":
                continue

            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except (json.JSONDecodeError, TypeError):
                    prices = []

            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except (json.JSONDecodeError, TypeError):
                    outcomes = []

            weather_markets.append({
                "source": "polymarket",
                "id": mid,
                "question": m.get("question", ""),
                "slug": m.get("slug", ""),
                "market_type": parsed["type"],
                "city": parsed["city"],
                "raw_city": parsed["raw_city"],
                "outcomes": outcomes,
                "outcomePrices": [float(p) for p in prices] if prices else [],
                "volume24hr": float(m.get("volume24hr", 0) or 0),
                "endDate": m.get("endDate", ""),
                "conditionId": m.get("conditionId", ""),
                "clobTokenIds": m.get("clobTokenIds", ""),
            })

    for term in text_searches:
        try:
            resp = requests.get(
                _GAMMA_MARKETS_URL,
                params={"closed": "false", "limit": 50, "order": "volume24hr", "ascending": "false", "tag": term},
                timeout=15,
            )
            resp.raise_for_status()
            markets = resp.json()
        except requests.RequestException as exc:
            logger.warning("Gamma API search for '%s' failed: %s", term, exc)
            continue

        for m in markets:
            mid = m.get("id") or m.get("conditionId", "")
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            parsed = _parse_market_type(m.get("question", ""))
            if parsed["type"] == "unknown":
                continue

            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except (json.JSONDecodeError, TypeError):
                    prices = []

            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except (json.JSONDecodeError, TypeError):
                    outcomes = []

            weather_markets.append({
                "source": "polymarket",
                "id": mid,
                "question": m.get("question", ""),
                "slug": m.get("slug", ""),
                "market_type": parsed["type"],
                "city": parsed["city"],
                "raw_city": parsed["raw_city"],
                "outcomes": outcomes,
                "outcomePrices": [float(p) for p in prices] if prices else [],
                "volume24hr": float(m.get("volume24hr", 0) or 0),
                "endDate": m.get("endDate", ""),
                "conditionId": m.get("conditionId", ""),
                "clobTokenIds": m.get("clobTokenIds", ""),
            })

    logger.info("Found %d Polymarket weather markets", len(weather_markets))
    _cache["markets"] = weather_markets
    _cache["fetched_at"] = time.monotonic()
    return weather_markets
