"""
Optional GOES satellite cloud-fraction fetcher via goes2go.

Phase 5 of the upgrade plan. Uses GOES-16/18 ABI imagery to detect
cloud cover near forecast cities and adjust daytime high-temp forecasts
downward when unexpected clouds block solar heating.

The implementation is intentionally defensive:
- GOES is only used when goes2go is installed
- failures return None instead of breaking the scan
- data are fetched for the latest available CONUS scan
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("weather.goes")

_DEFAULT_RADIUS_KM = 25.0
_CLOUD_ADJUSTMENT_MAX_F = -5.0  # max negative adjustment for full cloud cover
_PEAK_HEATING_HOURS_LOCAL = (10, 15)  # 10am-3pm local time


def _get_goes_class():
    """Import goes2go.GOES; raises RuntimeError if not installed."""
    try:
        from goes2go import GOES
    except ImportError as exc:
        raise RuntimeError(
            "GOES support requires goes2go to be installed: pip install goes2go"
        ) from exc
    return GOES


def fetch_cloud_fraction(
    lat: float,
    lon: float,
    radius_km: float = _DEFAULT_RADIUS_KM,
    *,
    band: int = 13,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """Fetch latest GOES ABI data and compute cloud fraction near a point.

    Args:
        lat: Latitude of the point.
        lon: Longitude of the point.
        radius_km: Radius around the point to sample pixels.
        band: ABI band number. 13 = 10.3um IR (day+night), 2 = visible (day only).
        now: Current time for recency check. Defaults to UTC now.

    Returns:
        dict with cloud_fraction (0-1), clear_sky_fraction, scan_time, band,
        pixel_count, and source. Returns None on failure.
    """
    # TODO: Implement once goes2go is installed
    # 1. GOES = _get_goes_class()
    # 2. G = GOES(satellite=16, product="ABI-L2-MCMIPC", domain="C")
    # 3. ds = G.nearesttime(now or datetime.now(timezone.utc))
    # 4. Extract bounding box around (lat, lon) with radius_km
    # 5. Compute cloud fraction from Cloud Mask (BCM) or brightness temp
    # 6. Return structured dict

    logger.debug(
        "GOES cloud fraction not yet implemented (lat=%.4f, lon=%.4f, band=%d)",
        lat, lon, band,
    )
    return None


def fetch_cloud_fraction_multi(
    locations: list[dict],
    *,
    band: int = 13,
    now: Optional[datetime] = None,
    radius_km: float = _DEFAULT_RADIUS_KM,
) -> dict[str, Optional[dict]]:
    """Fetch cloud fraction for multiple named locations.

    Args:
        locations: List of dicts with 'name', 'lat', 'lon' keys.

    Returns:
        Dict mapping location name to cloud_fraction result (or None).
    """
    results: dict[str, Optional[dict]] = {}
    for loc in locations:
        name = loc.get("name", f"{loc.get('lat')},{loc.get('lon')}")
        try:
            results[name] = fetch_cloud_fraction(
                float(loc["lat"]),
                float(loc["lon"]),
                radius_km=radius_km,
                band=band,
                now=now,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.debug("GOES fetch failed for %s: %s", name, exc)
            results[name] = None
    return results


def compute_cloud_adjustment_f(
    cloud_fraction: float,
    local_hour: int,
    max_adjustment_f: float = _CLOUD_ADJUSTMENT_MAX_F,
) -> float:
    """Compute a temperature adjustment in Fahrenheit based on cloud fraction.

    The adjustment is strongest during peak heating hours (10am-3pm local)
    and zero outside that window.

    Args:
        cloud_fraction: 0.0 (clear) to 1.0 (fully cloudy).
        local_hour: Hour of day in local timezone (0-23).
        max_adjustment_f: Maximum negative adjustment at 100% cloud cover
            during peak heating. Default -5F.

    Returns:
        Negative adjustment in Fahrenheit (0.0 if outside peak hours or clear).
    """
    if cloud_fraction <= 0.0:
        return 0.0

    peak_start, peak_end = _PEAK_HEATING_HOURS_LOCAL
    if local_hour < peak_start or local_hour > peak_end:
        return 0.0

    # Scale linearly: 100% cloud cover -> max_adjustment_f during peak
    # Reduce effect at edges of the heating window
    peak_mid = (peak_start + peak_end) / 2.0
    peak_half_width = (peak_end - peak_start) / 2.0
    hour_weight = 1.0 - abs(local_hour - peak_mid) / peak_half_width

    return max_adjustment_f * min(cloud_fraction, 1.0) * max(hour_weight, 0.0)


def get_goes_forecast_adjustment(
    goes_data: Optional[dict],
    local_hour: int,
    max_adjustment_f: float = _CLOUD_ADJUSTMENT_MAX_F,
) -> tuple[float, str]:
    """Get the forecast adjustment from GOES cloud data.

    Returns:
        Tuple of (adjustment_f, source_label). If GOES data is unavailable,
        returns (0.0, "no_goes_data").
    """
    if goes_data is None:
        return 0.0, "no_goes_data"

    cloud_fraction = float(goes_data.get("cloud_fraction", 0.0))
    adjustment = compute_cloud_adjustment_f(cloud_fraction, local_hour, max_adjustment_f)

    if adjustment == 0.0:
        return 0.0, "goes_clear_or_off_peak"

    return round(adjustment, 2), "goes_cloud_adjustment"
