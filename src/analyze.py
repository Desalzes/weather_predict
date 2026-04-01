"""
Confidence scoring and weather event detection.
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger("weather.analyze")

DEFAULT_THRESHOLDS = {
    "heavy_rain_mm": 10.0,
    "heavy_rain_prob": 80.0,
    "temp_swing_delta": 5.0,
    "high_wind_kmh": 50.0,
    "snowfall_threshold": 0.0,
    "heat_extreme_c": 35.0,
    "cold_extreme_c": -10.0,
}


def compute_ensemble_spread(ensemble_data: dict) -> dict:
    """Compute mean, std, min, max, IQR per variable per timestep from ensemble members."""
    hourly_block = ensemble_data.get("hourly", {})
    spread = {}

    for var_name, members in hourly_block.items():
        if var_name == "time":
            continue
        if not members or not isinstance(members[0], (list, tuple)):
            arr = np.array(members, dtype=float)
            spread[var_name] = {
                "mean": arr.tolist(), "std": np.zeros(len(arr)).tolist(),
                "min": arr.tolist(), "max": arr.tolist(),
                "iqr": np.zeros(len(arr)).tolist(),
                "normalized_spread": np.zeros(len(arr)).tolist(),
            }
            continue

        arr = np.array(members, dtype=float)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
        mn = np.min(arr, axis=0)
        mx = np.max(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)

        spread[var_name] = {
            "mean": mean.tolist(), "std": std.tolist(),
            "min": mn.tolist(), "max": mx.tolist(),
            "iqr": (q75 - q25).tolist(),
            "normalized_spread": (std / (np.abs(mean) + 1e-6)).tolist(),
        }

    return spread


def detect_weather_events(hourly_data: dict, thresholds: Optional[dict] = None) -> list:
    """Scan hourly forecast for notable weather events."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    h = hourly_data.get("hourly", {})
    n = len(h.get("time", []))
    events = []

    def _collapse_runs(flags, values):
        runs = []
        in_run = False
        s = 0
        for i, f in enumerate(flags):
            if f and not in_run:
                s = i
                in_run = True
            elif not f and in_run:
                runs.append((s, i - 1, values[s:i]))
                in_run = False
        if in_run:
            runs.append((s, len(flags) - 1, values[s:]))
        return runs

    def _severity(ratio):
        if ratio >= 3.0: return "extreme"
        if ratio >= 2.0: return "high"
        if ratio >= 1.3: return "moderate"
        return "low"

    def _pad(series, length, fill=0.0):
        return list(series) + [fill] * (length - len(series)) if len(series) < length else series[:length]

    precip = _pad(h.get("precipitation") or [], n)
    precip_prob = _pad(h.get("precipitation_probability") or [], n)
    temp = _pad(h.get("temperature_2m") or [], n, fill=float("nan"))
    wind = _pad(h.get("wind_speed_10m") or h.get("windspeed_10m") or [], n)
    snowfall = _pad(h.get("snowfall") or [], n)

    flags = [(p > t["heavy_rain_mm"] or pp > t["heavy_rain_prob"]) for p, pp in zip(precip, precip_prob)]
    for s, e, vals in _collapse_runs(flags, precip):
        peak = float(max(vals))
        events.append({"event_type": "heavy_rain", "severity": _severity(peak / max(t["heavy_rain_mm"], 0.01)),
                        "start_hour": s, "end_hour": e, "peak_value": peak, "confidence": 0.8})

    for i in range(n - 6):
        valid = [temp[i + j] for j in range(7) if not math.isnan(temp[i + j])]
        if len(valid) >= 2:
            delta = max(valid) - min(valid)
            if delta > t["temp_swing_delta"]:
                events.append({"event_type": "temperature_swing", "severity": _severity(delta / t["temp_swing_delta"]),
                                "start_hour": i, "end_hour": i + 6, "peak_value": round(delta, 2), "confidence": 0.7})

    flags = [w > t["high_wind_kmh"] for w in wind]
    for s, e, vals in _collapse_runs(flags, wind):
        peak = float(max(vals))
        events.append({"event_type": "high_wind", "severity": _severity(peak / t["high_wind_kmh"]),
                        "start_hour": s, "end_hour": e, "peak_value": round(peak, 1), "confidence": 0.75})

    return events


def score_prediction_confidence(hourly_data: dict, ensemble_data: Optional[dict] = None) -> list:
    """Return per-hour confidence scores in [0.0, 1.0]."""
    h = hourly_data.get("hourly", {})
    n = len(h.get("time", []))
    if n == 0:
        return []

    precip_prob = list(h.get("precipitation_probability") or [50.0] * n)
    precip_prob += [50.0] * max(0, n - len(precip_prob))

    base = np.array([pp / 100.0 for pp in precip_prob[:n]], dtype=float)
    base = np.where(base < 0.05, 0.95, base)
    base = np.clip(base, 0.0, 1.0)

    horizon_penalty = np.minimum(np.arange(n, dtype=float) / 72.0, 1.0) * 0.20
    scores = base - horizon_penalty

    if ensemble_data is not None:
        spread = compute_ensemble_spread(ensemble_data)
        spread_penalties = np.zeros(n)
        count = 0
        for metrics in spread.values():
            ns = metrics.get("normalized_spread", [])
            if ns:
                ns_arr = np.array(ns[:n], dtype=float)
                if len(ns_arr) < n:
                    ns_arr = np.pad(ns_arr, (0, n - len(ns_arr)))
                spread_penalties += np.minimum(ns_arr * 0.3, 0.15)
                count += 1
        if count:
            scores -= spread_penalties / count

    return [round(float(s), 4) for s in np.clip(scores, 0.0, 1.0)]
