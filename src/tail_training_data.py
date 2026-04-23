"""Tail and bucket training-set builders for P2 tail calibration.

Produces `(raw_prob, actual_exceeded_0_1)` pairs by joining the archived
forecasts in `data/forecast_archive/*.csv` with station truth in
`data/station_actuals/*.csv`. The `raw_prob` column is computed via the same
Gaussian CDF the live matcher uses (`src/matcher.py::_normal_cdf`) so that
downstream calibrators see the same probability surface at training time as at
inference time.

Two builders are exposed:

- `build_tail_training_set(city, market_type, direction, threshold, ...)`:
  training rows for tail-threshold markets (e.g. "high above 80°F").
- `build_bucket_training_set(city, market_type, bucket_low, bucket_high, ...)`:
  training rows for bucket markets (e.g. "high between 70°F and 71°F").

Both builders mirror the lead-1 preference used by the existing temperature
training pipeline in `src/station_truth.py::build_training_set`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

from src.station_truth import (
    FORECAST_ARCHIVE_DIR,
    STATION_ACTUALS_DIR,
    _slugify_city,
)

_EMPTY_COLUMNS = [
    "date",
    "raw_prob",
    "actual_exceeded_0_1",
    "forecast_value_f",
    "sigma_f",
    "forecast_lead_days",
    "as_of_utc",
]

_BUCKET_EMPTY_COLUMNS = [
    "date",
    "raw_bucket_prob",
    "actual_in_bucket_0_1",
    "forecast_value_f",
    "sigma_f",
    "forecast_lead_days",
    "as_of_utc",
]


def _normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Standard normal CDF using math.erfc — matches `src/matcher.py`."""
    sigma = max(float(sigma), 1e-6)
    z = (x - mu) / sigma
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _threshold_exceeded(actual_value: float, threshold: float, direction: str) -> int:
    """Return 1 if the actual value exceeds the threshold in the given direction."""
    if direction == "above":
        return 1 if actual_value > threshold else 0
    if direction == "below":
        return 1 if actual_value < threshold else 0
    raise ValueError(f"Unknown direction {direction!r}; expected 'above' or 'below'.")


def _raw_prob_above(forecast: float, threshold: float, sigma: float) -> float:
    """P(X > threshold) under N(forecast, sigma)."""
    return 1.0 - _normal_cdf(threshold, forecast, sigma)


def _raw_prob_below(forecast: float, threshold: float, sigma: float) -> float:
    """P(X < threshold) under N(forecast, sigma)."""
    return _normal_cdf(threshold, forecast, sigma)


def _resolve_columns(market_type: str) -> tuple[str, str, str]:
    """Return (forecast_col, sigma_col, actual_col) for a market type."""
    if market_type == "high":
        return "forecast_high_f", "ensemble_high_std_f", "tmax_f"
    if market_type == "low":
        return "forecast_low_f", "ensemble_low_std_f", "tmin_f"
    raise ValueError(f"Unknown market_type {market_type!r}; expected 'high' or 'low'.")


def _load_and_join(
    city: str,
    market_type: str,
    actuals_dir: Optional[Path | str],
    archive_dir: Optional[Path | str],
) -> Optional[pd.DataFrame]:
    """Shared join logic. Returns None if no valid joined rows exist.

    On success the returned frame has columns:
      date (YYYY-MM-DD str), forecast_value_f, sigma_f, actual_value_f,
      forecast_lead_days, as_of_utc.
    """
    actuals_dir_path = Path(actuals_dir) if actuals_dir is not None else STATION_ACTUALS_DIR
    archive_dir_path = Path(archive_dir) if archive_dir is not None else FORECAST_ARCHIVE_DIR

    if not actuals_dir_path.exists() or not archive_dir_path.exists():
        return None

    slug = _slugify_city(city)
    actuals_path = actuals_dir_path / f"{slug}.csv"
    archive_path = archive_dir_path / f"{slug}.csv"

    if not actuals_path.exists() or not archive_path.exists():
        return None

    actuals = pd.read_csv(actuals_path)
    archive = pd.read_csv(archive_path)
    if actuals.empty or archive.empty:
        return None

    forecast_col, sigma_col, actual_col = _resolve_columns(market_type)
    required_archive_cols = {forecast_col, sigma_col, "date", "forecast_lead_days", "as_of_utc"}
    if not required_archive_cols.issubset(archive.columns):
        return None
    if actual_col not in actuals.columns or "date" not in actuals.columns:
        return None

    # Normalize date columns to YYYY-MM-DD strings (idempotent for already-normalized input).
    archive = archive.copy()
    actuals = actuals.copy()
    archive["date"] = pd.to_datetime(archive["date"]).dt.strftime("%Y-%m-%d")
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.strftime("%Y-%m-%d")

    # Prefer earliest lead-1 forecast per date (mirrors existing temp-calibration
    # training discipline in `build_training_set`).
    archive = archive.sort_values(["date", "forecast_lead_days", "as_of_utc"])
    archive = archive.drop_duplicates("date", keep="first")

    merged = archive.merge(
        actuals[["date", actual_col]],
        on="date",
        how="inner",
    )
    if merged.empty:
        return None

    merged = merged.rename(
        columns={
            forecast_col: "forecast_value_f",
            sigma_col: "sigma_f",
            actual_col: "actual_value_f",
        }
    )
    merged["forecast_value_f"] = pd.to_numeric(merged["forecast_value_f"], errors="coerce")
    merged["sigma_f"] = pd.to_numeric(merged["sigma_f"], errors="coerce")
    merged["actual_value_f"] = pd.to_numeric(merged["actual_value_f"], errors="coerce")

    merged = merged.dropna(subset=["forecast_value_f", "sigma_f", "actual_value_f"])
    if merged.empty:
        return None

    return merged.reset_index(drop=True)


def build_tail_training_set(
    city: str,
    market_type: str,
    direction: str,
    threshold: float,
    actuals_dir: Optional[Path | str] = None,
    archive_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Produce (raw_prob, actual_exceeded_0_1) training rows for a threshold market.

    Parameters
    ----------
    city : str
        City name (slugged via `_slugify_city`).
    market_type : str
        "high" or "low" — which forecast/actual columns to use.
    direction : str
        "above" or "below" — the tail side of the threshold.
    threshold : float
        The temperature threshold (°F).
    actuals_dir, archive_dir : Path or str, optional
        Override source directories (defaults to the module-level dirs in
        `src.station_truth`).

    Returns
    -------
    pd.DataFrame with columns `_EMPTY_COLUMNS` in that order. Empty if either
    source file is missing/empty or the directories do not exist at all.
    """
    merged = _load_and_join(city, market_type, actuals_dir, archive_dir)
    if merged is None:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    if direction == "above":
        merged["raw_prob"] = merged.apply(
            lambda r: _raw_prob_above(r["forecast_value_f"], threshold, r["sigma_f"]),
            axis=1,
        )
    elif direction == "below":
        merged["raw_prob"] = merged.apply(
            lambda r: _raw_prob_below(r["forecast_value_f"], threshold, r["sigma_f"]),
            axis=1,
        )
    else:
        raise ValueError(f"Unknown direction {direction!r}; expected 'above' or 'below'.")

    merged["actual_exceeded_0_1"] = merged["actual_value_f"].apply(
        lambda v: _threshold_exceeded(v, threshold, direction)
    ).astype("int64")

    result = merged[[
        "date",
        "raw_prob",
        "actual_exceeded_0_1",
        "forecast_value_f",
        "sigma_f",
        "forecast_lead_days",
        "as_of_utc",
    ]].copy()
    result["as_of_utc"] = result["as_of_utc"].astype(str)
    return result.reset_index(drop=True)


def build_bucket_training_set(
    city: str,
    market_type: str,
    bucket_low: float,
    bucket_high: float,
    actuals_dir: Optional[Path | str] = None,
    archive_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Produce (raw_bucket_prob, actual_in_bucket_0_1) training rows for a bucket market.

    The raw bucket probability is `F(bucket_high) - F(bucket_low)` under the
    forecast's normal distribution. The actual outcome is 1 if the station
    actual falls within `[bucket_low, bucket_high]` inclusive.

    Returns
    -------
    pd.DataFrame with columns `_BUCKET_EMPTY_COLUMNS` in that order. Empty if
    either source file is missing/empty or the directories do not exist at all.
    """
    merged = _load_and_join(city, market_type, actuals_dir, archive_dir)
    if merged is None:
        return pd.DataFrame(columns=_BUCKET_EMPTY_COLUMNS)

    merged["raw_bucket_prob"] = merged.apply(
        lambda r: _normal_cdf(bucket_high, r["forecast_value_f"], r["sigma_f"])
        - _normal_cdf(bucket_low, r["forecast_value_f"], r["sigma_f"]),
        axis=1,
    )
    merged["actual_in_bucket_0_1"] = merged["actual_value_f"].apply(
        lambda v: 1 if bucket_low <= v <= bucket_high else 0
    ).astype("int64")

    result = merged[[
        "date",
        "raw_bucket_prob",
        "actual_in_bucket_0_1",
        "forecast_value_f",
        "sigma_f",
        "forecast_lead_days",
        "as_of_utc",
    ]].copy()
    result["as_of_utc"] = result["as_of_utc"].astype(str)
    return result.reset_index(drop=True)
