"""
Deterministic chronological holdout evaluation for calibration models.

Usage:
    python evaluate_calibration.py
    python evaluate_calibration.py --days 400 --holdout-days 30
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.calibration import (
    EMOSCalibrator,
    IsotonicCalibrator,
    SELECTIVE_RAW_FALLBACK_SOURCE,
    SELECTIVE_RAW_FALLBACK_TARGETS,
    build_isotonic_examples,
    compute_temperature_probability,
    is_selective_raw_fallback_pair,
)
from src.station_truth import FORECAST_ARCHIVE_DIR, STATION_ACTUALS_DIR, build_training_set

_MIN_SPREAD_F = 1.0
_BROAD_POLICY = "broad_emos_isotonic"
_SELECTIVE_POLICY = "selective_raw_fallback"
_POLICY_ORDER = (_BROAD_POLICY, _SELECTIVE_POLICY)
_PROXY_CONFIDENCE_THRESHOLD = 0.10
_PROXY_MARKET_PRICE_ASSUMPTION = 0.50
_DAY_AHEAD_LEAD_DAYS = 1
_MULTI_DAY_MIN_LEAD_DAYS = 2


def _default_city_name_from_slug(slug: str) -> str:
    words = slug.replace("_", " ").title()
    if words == "Washington Dc":
        return "Washington DC"
    return words


def _pair_label(city: str, market_type: str) -> str:
    return f"{city} {market_type}"


def _is_targeted_fallback_pair(city: str, market_type: str) -> bool:
    return is_selective_raw_fallback_pair(city, market_type)


def _policy_definitions() -> dict[str, dict]:
    targeted_pairs = [_pair_label(city, market_type) for city, market_type in SELECTIVE_RAW_FALLBACK_TARGETS]
    return {
        _BROAD_POLICY: {
            "description": (
                "Fit EMOS on the training split for every city-market pair, apply EMOS to every holdout row, "
                "fit isotonic on EMOS-based training probabilities, and apply that isotonic mapping to holdout "
                "probabilities."
            ),
            "targeted_raw_fallback_pairs": [],
            "isotonic_policy": "fit on EMOS training probabilities and apply unchanged on holdout",
        },
        _SELECTIVE_POLICY: {
            "description": (
                "Use the same chronological split, EMOS fit, and isotonic fit as the broad policy. On holdout only, "
                "route Boston low, Minneapolis low, Philadelphia low, New Orleans high, and San Francisco low "
                "through the raw forecast instead of EMOS while leaving isotonic unchanged."
            ),
            "targeted_raw_fallback_pairs": targeted_pairs,
            "isotonic_policy": "reuse the broad policy isotonic mapping unchanged",
        },
    }


def discover_cities(
    *,
    station_actuals_dir: Path | str = STATION_ACTUALS_DIR,
    forecast_archive_dir: Path | str = FORECAST_ARCHIVE_DIR,
) -> list[str]:
    actuals_dir = Path(station_actuals_dir)
    archive_dir = Path(forecast_archive_dir)
    cities: list[str] = []

    for actuals_path in sorted(actuals_dir.glob("*.csv")):
        archive_path = archive_dir / actuals_path.name
        if not archive_path.exists():
            continue

        city_name = _default_city_name_from_slug(actuals_path.stem)
        try:
            sample = pd.read_csv(actuals_path, usecols=["city"], nrows=1)
            if not sample.empty and "city" in sample.columns:
                value = str(sample.iloc[0]["city"]).strip()
                if value and value.lower() != "nan":
                    city_name = value
        except Exception:
            pass

        cities.append(city_name)

    return cities


def split_chronological_holdout(
    df: pd.DataFrame,
    holdout_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_days < 1:
        raise ValueError("holdout_days must be at least 1")

    if df.empty:
        return df.copy(), df.copy()

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return frame, frame.copy()

    unique_dates = sorted(frame["date"].drop_duplicates().tolist())
    holdout_dates = set(unique_dates[-holdout_days:])

    train = frame.loc[~frame["date"].isin(holdout_dates)].copy()
    holdout = frame.loc[frame["date"].isin(holdout_dates)].copy()

    train["date"] = train["date"].dt.date.astype(str)
    holdout["date"] = holdout["date"].dt.date.astype(str)
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def _safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _metric_delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def _brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(np.square(predictions - outcomes)))


def _mean_signed_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(predicted - actual))


def _probability_bias(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(predictions) - np.mean(outcomes))


def _bias_direction(bias: float | None) -> str:
    if bias is None:
        return "unknown"
    if bias > 0:
        return "warm_or_high"
    if bias < 0:
        return "cold_or_low"
    return "neutral"


def _prepare_diagnostic_market_frame(df: pd.DataFrame, market_type: str) -> pd.DataFrame:
    diagnostic_columns = ["date", "forecast_f", "actual_f", "spread_f", "forecast_lead_days"]
    forecast_col = f"forecast_{market_type}_f"
    actual_col = f"actual_{market_type}_f"
    spread_col = f"ensemble_{market_type}_std_f"

    if forecast_col not in df.columns or actual_col not in df.columns:
        return pd.DataFrame(columns=diagnostic_columns)

    frame = df[["date", forecast_col, actual_col]].copy()
    frame["forecast_f"] = pd.to_numeric(frame[forecast_col], errors="coerce")
    frame["actual_f"] = pd.to_numeric(frame[actual_col], errors="coerce")

    if spread_col in df.columns:
        frame["spread_f"] = pd.to_numeric(df[spread_col], errors="coerce")
    else:
        frame["spread_f"] = pd.NA

    if "forecast_lead_days" in df.columns:
        frame["forecast_lead_days"] = pd.to_numeric(df["forecast_lead_days"], errors="coerce")
    else:
        frame["forecast_lead_days"] = pd.NA

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "forecast_f", "actual_f"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=diagnostic_columns)

    spread_series = frame["spread_f"].dropna()
    fallback_spread = float(spread_series.median()) if not spread_series.empty else _MIN_SPREAD_F
    frame["spread_f"] = frame["spread_f"].fillna(fallback_spread).clip(lower=_MIN_SPREAD_F)
    return frame[diagnostic_columns].reset_index(drop=True)


def _temperature_diagnostic_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "holdout_rows": 0,
            "raw_bias_f": None,
            "raw_bias_direction": "unknown",
            "policy_bias_f": None,
            "policy_bias_direction": "unknown",
            "bias_delta_f": None,
            "mean_adjustment_f": None,
            "mean_absolute_adjustment_f": None,
            "median_adjustment_f": None,
        }

    actual = frame["actual_f"].to_numpy(dtype=float)
    raw_forecast = frame["forecast_f"].to_numpy(dtype=float)
    policy_forecast = frame["policy_forecast_f"].to_numpy(dtype=float)
    adjustments = policy_forecast - raw_forecast

    raw_bias_f = _mean_signed_error(actual, raw_forecast)
    policy_bias_f = _mean_signed_error(actual, policy_forecast)

    return {
        "holdout_rows": int(len(frame)),
        "raw_bias_f": raw_bias_f,
        "raw_bias_direction": _bias_direction(raw_bias_f),
        "policy_bias_f": policy_bias_f,
        "policy_bias_direction": _bias_direction(policy_bias_f),
        "bias_delta_f": policy_bias_f - raw_bias_f,
        "mean_adjustment_f": float(np.mean(adjustments)),
        "mean_absolute_adjustment_f": float(np.mean(np.abs(adjustments))),
        "median_adjustment_f": float(np.median(adjustments)),
    }


def _empty_probability_diagnostics(reason: str) -> dict:
    return {
        "status": "unavailable",
        "reason": reason,
        "holdout_probability_examples": 0,
        "outcome_rate": None,
        "raw_brier": None,
        "policy_input_brier": None,
        "policy_brier": None,
        "policy_brier_delta_vs_raw": None,
        "policy_brier_delta_vs_input": None,
        "raw_mean_probability": None,
        "policy_input_mean_probability": None,
        "policy_mean_probability": None,
        "raw_probability_bias": None,
        "policy_input_probability_bias": None,
        "policy_probability_bias": None,
    }


def _empty_proxy_trade_selection(reason: str) -> dict:
    return {
        "status": "unavailable",
        "reason": reason,
        "confidence_threshold_abs_from_fair": _PROXY_CONFIDENCE_THRESHOLD,
        "market_price_assumption": _PROXY_MARKET_PRICE_ASSUMPTION,
        "selected_examples": 0,
        "correct_direction_examples": 0,
        "selected_share": None,
        "directional_hit_rate": None,
        "sum_abs_edge_vs_fair": 0.0,
        "avg_abs_edge_vs_fair": None,
        "unit_even_odds_pnl_proxy": 0.0,
        "unit_even_odds_roi_proxy": None,
        "note": (
            "Proxy unavailable. Historical quote data is not present in the holdout rows, so this evaluator can only "
            "derive even-odds directional proxies from threshold examples when probabilities exist."
        ),
    }


def _empty_temperature_diagnostics(reason: str) -> dict:
    return {
        "holdout_rows": 0,
        "raw_bias_f": None,
        "raw_bias_direction": "unknown",
        "policy_bias_f": None,
        "policy_bias_direction": "unknown",
        "bias_delta_f": None,
        "mean_adjustment_f": None,
        "mean_absolute_adjustment_f": None,
        "median_adjustment_f": None,
        "reason": reason,
        "regime_diagnostics": {},
    }


def _lead_time_bucket_definition() -> dict[str, str]:
    return {
        "day_ahead": f"forecast_lead_days == {_DAY_AHEAD_LEAD_DAYS}",
        "multi_day": f"forecast_lead_days >= {_MULTI_DAY_MIN_LEAD_DAYS}",
    }


def _lead_time_regime_label(value: float | int | None) -> str | None:
    if value is None or pd.isna(value):
        return None

    lead_days = float(value)
    if lead_days == _DAY_AHEAD_LEAD_DAYS:
        return "day_ahead"
    if lead_days >= _MULTI_DAY_MIN_LEAD_DAYS:
        return "multi_day"
    return None


def _summarize_unsupported_reasons(reasons: list[str]) -> list[dict]:
    counts: dict[str, int] = {}
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    return [
        {"reason": reason, "pair_count": pair_count}
        for reason, pair_count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _report_level_limitations(results: list[dict]) -> list[dict]:
    evaluated_results = [result for result in results if result.get("status") == "evaluated"]
    if not evaluated_results:
        return []

    limitations: list[dict] = []
    for dimension in ("lead_time_regime", "spread_regime"):
        supported_pairs = 0
        unsupported_reasons: list[str] = []

        for result in evaluated_results:
            regime_summary = (
                result.get("policies", {})
                .get(_BROAD_POLICY, {})
                .get("temperature_diagnostics", {})
                .get("regime_diagnostics", {})
                .get(dimension)
            )
            if regime_summary and regime_summary.get("status") == "evaluated":
                supported_pairs += 1
                continue

            reason = str(
                (regime_summary or {}).get("reason") or f"{dimension} summary missing from the pair diagnostics"
            )
            unsupported_reasons.append(reason)

        total_pairs = len(evaluated_results)
        unsupported_pairs = total_pairs - supported_pairs
        if unsupported_pairs <= 0:
            continue

        if dimension == "lead_time_regime" and supported_pairs <= 0:
            status = "insufficient_data"
            reason = (
                f"{dimension} is wired through the evaluator, but this run had 0/{total_pairs} city-market pairs "
                "whose holdout rows spanned both lead-time buckets. Every pair reports an explicit per-pair "
                "insufficiency reason instead."
            )
        else:
            status = "partially_supported" if supported_pairs > 0 else "unsupported"
            reason = (
                f"{dimension} was evaluated for {supported_pairs}/{total_pairs} city-market pairs. "
                f"The remaining {unsupported_pairs} pairs report explicit per-pair insufficiency reasons."
                if supported_pairs > 0
                else f"{dimension} remained unsupported for all {total_pairs} evaluated city-market pairs."
            )

        limitation = {
            "dimension": dimension,
            "status": status,
            "evaluated_pairs": total_pairs,
            "supported_pairs": supported_pairs,
            "unsupported_pairs": unsupported_pairs,
            "reason": reason,
            "unsupported_reasons": _summarize_unsupported_reasons(unsupported_reasons),
        }
        if dimension == "lead_time_regime":
            limitation["bucket_definition"] = _lead_time_bucket_definition()
        limitations.append(limitation)

    return limitations


def _proxy_trade_impact_note() -> dict:
    return {
        "status": "proxy_only",
        "reason": (
            "The holdout dataset has no historical market-price or quote snapshots, so the evaluator cannot estimate "
            "realized edge, fills, or paper PnL. It reports a bounded even-odds directional proxy from holdout "
            "threshold examples instead."
        ),
        "proxy_definition": (
            "Select threshold examples where |policy_prob - 0.50| >= 0.10, choose YES when policy_prob >= 0.60 and "
            "NO when policy_prob <= 0.40, then score a hypothetical even-odds line at 0.50 as +0.5 when correct and "
            "-0.5 when wrong."
        ),
    }


def _verdict_from_delta(delta: float | None) -> str:
    if delta is None:
        return "insufficient_data"
    if delta < 0:
        return "helps"
    if delta > 0:
        return "hurts"
    return "flat"


def _policy_probability_arrays(
    frame: pd.DataFrame,
    market_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame.empty:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    required = ["forecast_f", "actual_f", "spread_f", "emos_forecast_f", "policy_forecast_f"]
    prepared = frame[required].copy()
    for column in required:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=required).reset_index(drop=True)
    if prepared.empty:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    raw_probs: list[float] = []
    policy_input_probs: list[float] = []
    outcomes: list[float] = []

    for row in prepared.itertuples(index=False):
        raw_forecast = float(row.forecast_f)
        emos_forecast = float(row.emos_forecast_f)
        policy_forecast = float(row.policy_forecast_f)
        actual_f = float(row.actual_f)
        spread_f = max(float(row.spread_f), _MIN_SPREAD_F)

        radius = max(4.0, spread_f * 2.0)
        lower = int(math.floor(min(raw_forecast, emos_forecast, actual_f) - radius))
        upper = int(math.ceil(max(raw_forecast, emos_forecast, actual_f) + radius))

        for threshold in range(lower, upper + 1, 2):
            if market_type == "high":
                raw_prob = compute_temperature_probability(raw_forecast, threshold, None, spread_f)
                policy_input_prob = compute_temperature_probability(policy_forecast, threshold, None, spread_f)
                outcome = 1.0 if actual_f >= threshold else 0.0
            else:
                raw_prob = compute_temperature_probability(raw_forecast, None, threshold + 0.99, spread_f)
                policy_input_prob = compute_temperature_probability(policy_forecast, None, threshold + 0.99, spread_f)
                outcome = 1.0 if actual_f <= threshold else 0.0

            raw_probs.append(raw_prob)
            policy_input_probs.append(policy_input_prob)
            outcomes.append(outcome)

    return (
        np.asarray(raw_probs, dtype=float),
        np.asarray(policy_input_probs, dtype=float),
        np.asarray(outcomes, dtype=float),
    )


def _probability_diagnostic_from_arrays(
    raw_probs: np.ndarray,
    policy_input_probs: np.ndarray,
    outcomes: np.ndarray,
    isotonic_model: IsotonicCalibrator | None,
    *,
    isotonic_reason: str,
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    if raw_probs.size == 0:
        return _empty_probability_diagnostics("holdout rows did not yield probability examples"), None, None

    policy_probs = np.asarray(policy_input_probs, dtype=float)
    status = "policy_input_only"
    reason = isotonic_reason
    if isotonic_model is not None:
        policy_probs = np.asarray([isotonic_model.calibrate(prob) for prob in policy_input_probs], dtype=float)
        status = "evaluated"
        reason = ""

    raw_brier = _brier_score(raw_probs, outcomes)
    policy_input_brier = _brier_score(policy_input_probs, outcomes)
    policy_brier = _brier_score(policy_probs, outcomes)

    return (
        {
            "status": status,
            "reason": reason,
            "holdout_probability_examples": int(raw_probs.size),
            "outcome_rate": float(np.mean(outcomes)),
            "raw_brier": raw_brier,
            "policy_input_brier": policy_input_brier,
            "policy_brier": policy_brier,
            "policy_brier_delta_vs_raw": policy_brier - raw_brier,
            "policy_brier_delta_vs_input": policy_brier - policy_input_brier,
            "raw_mean_probability": float(np.mean(raw_probs)),
            "policy_input_mean_probability": float(np.mean(policy_input_probs)),
            "policy_mean_probability": float(np.mean(policy_probs)),
            "raw_probability_bias": _probability_bias(raw_probs, outcomes),
            "policy_input_probability_bias": _probability_bias(policy_input_probs, outcomes),
            "policy_probability_bias": _probability_bias(policy_probs, outcomes),
        },
        policy_probs,
        outcomes,
    )


def _probability_diagnostic_summary(
    frame: pd.DataFrame,
    market_type: str,
    isotonic_model: IsotonicCalibrator | None,
    *,
    isotonic_reason: str,
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    raw_probs, policy_input_probs, outcomes = _policy_probability_arrays(frame, market_type)
    return _probability_diagnostic_from_arrays(
        raw_probs,
        policy_input_probs,
        outcomes,
        isotonic_model,
        isotonic_reason=isotonic_reason,
    )


def _proxy_trade_selection_summary(
    policy_probs: np.ndarray | None,
    outcomes: np.ndarray | None,
    *,
    total_examples: int,
) -> dict:
    if policy_probs is None or outcomes is None or total_examples <= 0:
        return _empty_proxy_trade_selection("probability examples unavailable for proxy scoring")

    buy_yes_mask = policy_probs >= (_PROXY_MARKET_PRICE_ASSUMPTION + _PROXY_CONFIDENCE_THRESHOLD)
    buy_no_mask = policy_probs <= (_PROXY_MARKET_PRICE_ASSUMPTION - _PROXY_CONFIDENCE_THRESHOLD)
    selected_mask = buy_yes_mask | buy_no_mask

    if not np.any(selected_mask):
        return _empty_proxy_trade_selection("no holdout threshold examples cleared the proxy confidence threshold")

    selected_probs = policy_probs[selected_mask]
    selected_outcomes = outcomes[selected_mask]
    chosen_direction = np.where(buy_yes_mask[selected_mask], 1.0, 0.0)
    correct_mask = chosen_direction == selected_outcomes
    unit_even_odds_pnl = np.where(correct_mask, 0.5, -0.5)
    selected_examples = int(selected_mask.sum())
    correct_examples = int(correct_mask.sum())
    sum_abs_edge_vs_fair = float(np.sum(np.abs(selected_probs - _PROXY_MARKET_PRICE_ASSUMPTION)))
    unit_even_odds_pnl_proxy = float(np.sum(unit_even_odds_pnl))

    return {
        "status": "proxy",
        "reason": "",
        "confidence_threshold_abs_from_fair": _PROXY_CONFIDENCE_THRESHOLD,
        "market_price_assumption": _PROXY_MARKET_PRICE_ASSUMPTION,
        "selected_examples": selected_examples,
        "correct_direction_examples": correct_examples,
        "selected_share": float(selected_examples / total_examples),
        "directional_hit_rate": float(correct_examples / selected_examples),
        "sum_abs_edge_vs_fair": sum_abs_edge_vs_fair,
        "avg_abs_edge_vs_fair": float(sum_abs_edge_vs_fair / selected_examples),
        "unit_even_odds_pnl_proxy": unit_even_odds_pnl_proxy,
        "unit_even_odds_roi_proxy": float(unit_even_odds_pnl_proxy / (0.5 * selected_examples)),
        "note": (
            "Proxy only. This treats the holdout threshold example as if it were tradable at a fair 0.50 line and "
            "does not replay historical quotes, liquidity, fees, or fills."
        ),
    }


def _segment_metrics(
    frame: pd.DataFrame,
    market_type: str,
    isotonic_model: IsotonicCalibrator | None,
    *,
    isotonic_reason: str,
) -> dict:
    actual = frame["actual_f"].to_numpy(dtype=float)
    raw_forecast = frame["forecast_f"].to_numpy(dtype=float)
    policy_forecast = frame["policy_forecast_f"].to_numpy(dtype=float)

    summary = {
        "holdout_rows": int(len(frame)),
        "raw_mae_f": _mean_absolute_error(actual, raw_forecast),
        "policy_mae_f": _mean_absolute_error(actual, policy_forecast),
        "policy_mae_delta_vs_raw_f": _mean_absolute_error(actual, policy_forecast)
        - _mean_absolute_error(actual, raw_forecast),
        "raw_bias_f": _mean_signed_error(actual, raw_forecast),
        "policy_bias_f": _mean_signed_error(actual, policy_forecast),
        "mean_adjustment_f": float(np.mean(policy_forecast - raw_forecast)),
    }
    probability_summary, _, _ = _probability_diagnostic_summary(
        frame,
        market_type,
        isotonic_model,
        isotonic_reason=isotonic_reason,
    )
    summary.update(probability_summary)
    return summary


def _unsupported_regime_summary(
    dimension: str,
    reason: str,
) -> tuple[dict, dict]:
    summary = {
        "status": "unsupported",
        "dimension": dimension,
        "reason": reason,
        "segments": [],
    }
    limitation = {
        "dimension": dimension,
        "status": "unsupported",
        "reason": reason,
    }
    return summary, limitation


def _build_regime_diagnostics(
    frame: pd.DataFrame,
    market_type: str,
    isotonic_model: IsotonicCalibrator | None,
    *,
    isotonic_reason: str,
) -> tuple[dict, list[dict]]:
    regimes: dict[str, dict] = {}
    limitations: list[dict] = []

    if frame.empty:
        summary, limitation = _unsupported_regime_summary(
            "holdout_rows",
            "no holdout rows available for regime diagnostics",
        )
        regimes["forecast_regime"] = summary
        regimes["holdout_half"] = summary
        regimes["lead_time_regime"] = summary
        regimes["spread_regime"] = summary
        limitations.append(limitation)
        return regimes, limitations

    if frame["forecast_f"].nunique() < 2:
        summary, limitation = _unsupported_regime_summary(
            "forecast_regime",
            "holdout forecast_f had fewer than 2 unique values, so no forecast-temperature split is available",
        )
        regimes["forecast_regime"] = summary
        limitations.append(limitation)
    else:
        forecast_cutoff = float(frame["forecast_f"].median())
        forecast_frame = frame.assign(
            regime=np.where(frame["forecast_f"] >= forecast_cutoff, "warmer_or_equal", "cooler")
        )
        regimes["forecast_regime"] = {
            "status": "evaluated",
            "dimension": "forecast_regime",
            "split_field": "forecast_f",
            "split_value": forecast_cutoff,
            "segments": [
                {
                    "segment": segment,
                    **_segment_metrics(
                        group,
                        market_type,
                        isotonic_model,
                        isotonic_reason=isotonic_reason,
                    ),
                }
                for segment, group in forecast_frame.groupby("regime", sort=True)
            ],
        }

    ordered = frame.sort_values(["date", "forecast_f", "actual_f"]).reset_index(drop=True)
    if len(ordered) < 2:
        summary, limitation = _unsupported_regime_summary(
            "holdout_half",
            "need at least 2 holdout rows for chronological half-split diagnostics",
        )
        regimes["holdout_half"] = summary
        limitations.append(limitation)
    else:
        midpoint = len(ordered) // 2
        holdout_half = ordered.assign(
            regime=["early_holdout" if index < midpoint else "late_holdout" for index in range(len(ordered))]
        )
        split_date = str(holdout_half.iloc[midpoint]["date"].date()) if midpoint < len(holdout_half) else None
        regimes["holdout_half"] = {
            "status": "evaluated",
            "dimension": "holdout_half",
            "split_field": "date",
            "split_value": split_date,
            "segments": [
                {
                    "segment": segment,
                    **_segment_metrics(
                        group,
                        market_type,
                        isotonic_model,
                        isotonic_reason=isotonic_reason,
                    ),
                }
                for segment, group in holdout_half.groupby("regime", sort=True)
            ],
        }

    lead_time_frame = frame.copy()
    lead_time_frame["lead_time_regime"] = lead_time_frame["forecast_lead_days"].apply(_lead_time_regime_label)
    usable_lead_time = lead_time_frame.dropna(subset=["lead_time_regime"]).copy()
    if usable_lead_time.empty:
        reason = (
            "holdout rows had no usable forecast_lead_days values after the leakage-safe training join, so the "
            "day_ahead versus multi_day split is unavailable for this pair"
        )
        regimes["lead_time_regime"] = {
            "status": "unsupported",
            "dimension": "lead_time_regime",
            "reason": reason,
            "bucket_definition": _lead_time_bucket_definition(),
            "usable_holdout_rows": 0,
            "ignored_holdout_rows": int(len(frame)),
            "segments": [],
        }
        limitations.append(
            {
                "dimension": "lead_time_regime",
                "status": "unsupported",
                "reason": reason,
            }
        )
    elif usable_lead_time["lead_time_regime"].nunique() < 2:
        present_segments = sorted(usable_lead_time["lead_time_regime"].unique().tolist())
        missing_segments = [
            segment for segment in _lead_time_bucket_definition().keys() if segment not in present_segments
        ]
        reason = (
            "holdout rows did not span both lead-time buckets after applying "
            f"day_ahead={_DAY_AHEAD_LEAD_DAYS} and multi_day>={_MULTI_DAY_MIN_LEAD_DAYS}; "
            f"usable rows only covered {', '.join(present_segments)} and missed {', '.join(missing_segments)}"
        )
        regimes["lead_time_regime"] = {
            "status": "unsupported",
            "dimension": "lead_time_regime",
            "reason": reason,
            "bucket_definition": _lead_time_bucket_definition(),
            "usable_holdout_rows": int(len(usable_lead_time)),
            "ignored_holdout_rows": int(len(frame) - len(usable_lead_time)),
            "segments": [],
        }
        limitations.append(
            {
                "dimension": "lead_time_regime",
                "status": "unsupported",
                "reason": reason,
            }
        )
    else:
        regimes["lead_time_regime"] = {
            "status": "evaluated",
            "dimension": "lead_time_regime",
            "split_field": "forecast_lead_days",
            "bucket_definition": _lead_time_bucket_definition(),
            "usable_holdout_rows": int(len(usable_lead_time)),
            "ignored_holdout_rows": int(len(frame) - len(usable_lead_time)),
            "segments": [
                {
                    "segment": segment,
                    **_segment_metrics(
                        usable_lead_time.loc[usable_lead_time["lead_time_regime"] == segment].copy(),
                        market_type,
                        isotonic_model,
                        isotonic_reason=isotonic_reason,
                    ),
                }
                for segment in ("day_ahead", "multi_day")
                if not usable_lead_time.loc[usable_lead_time["lead_time_regime"] == segment].empty
            ],
        }

    if frame["spread_f"].nunique() < 2:
        summary, limitation = _unsupported_regime_summary(
            "spread_regime",
            "holdout spread_f had fewer than 2 unique values after prepare_training_frame-style filling; "
            "build_training_set() currently keeps one prior-run row per date and archive_previous_run_forecast() "
            "writes ensemble_high_std_f/ensemble_low_std_f as missing values, so spread-based diagnostics are not supported",
        )
        regimes["spread_regime"] = summary
        limitations.append(limitation)
    else:
        spread_cutoff = float(frame["spread_f"].median())
        spread_frame = frame.assign(
            regime=np.where(frame["spread_f"] >= spread_cutoff, "higher_or_equal_spread", "lower_spread")
        )
        regimes["spread_regime"] = {
            "status": "evaluated",
            "dimension": "spread_regime",
            "split_field": "spread_f",
            "split_value": spread_cutoff,
            "segments": [
                {
                    "segment": segment,
                    **_segment_metrics(
                        group,
                        market_type,
                        isotonic_model,
                        isotonic_reason=isotonic_reason,
                    ),
                }
                for segment, group in spread_frame.groupby("regime", sort=True)
            ],
        }

    return regimes, limitations


def _dedupe_limitations(limitations: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for limitation in limitations:
        key = (str(limitation.get("dimension")), str(limitation.get("reason")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(limitation)
    return deduped


def _policy_sources(
    city: str,
    market_type: str,
    policy_name: str,
) -> tuple[str, str, bool]:
    use_raw_fallback = policy_name == _SELECTIVE_POLICY and _is_targeted_fallback_pair(city, market_type)
    if use_raw_fallback:
        return SELECTIVE_RAW_FALLBACK_SOURCE, "raw", True
    return "emos", "emos", False


def _empty_policy_result(
    city: str,
    market_type: str,
    policy_name: str,
    *,
    reason: str,
) -> dict:
    temperature_source, probability_input_source, _ = _policy_sources(city, market_type, policy_name)
    return {
        "temperature_source": temperature_source,
        "probability_input_source": probability_input_source,
        "probability_calibration_source": "unavailable",
        "holdout_rows": 0,
        "holdout_probability_examples": 0,
        "raw_mae_f": None,
        "policy_mae_f": None,
        "policy_mae_delta_vs_raw_f": None,
        "raw_rmse_f": None,
        "policy_rmse_f": None,
        "policy_rmse_delta_vs_raw_f": None,
        "raw_brier": None,
        "policy_input_brier": None,
        "policy_brier": None,
        "policy_brier_delta_vs_raw": None,
        "policy_brier_delta_vs_input": None,
        "temperature_verdict": "insufficient_data",
        "probability_verdict": "insufficient_data",
        "temperature_diagnostics": _empty_temperature_diagnostics(reason),
        "probability_diagnostics": _empty_probability_diagnostics(reason),
        "proxy_trade_selection": _empty_proxy_trade_selection(reason),
    }


def _empty_result(
    city: str,
    market_type: str,
    *,
    status: str,
    reason: str,
    total_rows: int = 0,
    train_rows: int = 0,
    holdout_rows: int = 0,
    available_unique_dates: int = 0,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    holdout_start_date: str | None = None,
    holdout_end_date: str | None = None,
) -> dict:
    return {
        "city": city,
        "market_type": market_type,
        "pair": _pair_label(city, market_type),
        "is_targeted_fallback_pair": _is_targeted_fallback_pair(city, market_type),
        "status": status,
        "reason": reason,
        "total_rows": int(total_rows),
        "available_unique_dates": int(available_unique_dates),
        "train_rows": int(train_rows),
        "holdout_rows": int(holdout_rows),
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "holdout_start_date": holdout_start_date,
        "holdout_end_date": holdout_end_date,
        "isotonic_train_examples": 0,
        "diagnostic_limitations": [],
        "policies": {
            policy_name: _empty_policy_result(city, market_type, policy_name, reason=reason)
            for policy_name in _POLICY_ORDER
        },
    }


def _evaluate_policy(
    city: str,
    market_type: str,
    holdout_market: pd.DataFrame,
    isotonic_model: IsotonicCalibrator | None,
    *,
    policy_name: str,
    isotonic_reason: str,
) -> tuple[dict, list[dict]]:
    temperature_source, probability_input_source, use_raw_fallback = _policy_sources(city, market_type, policy_name)

    policy_frame = holdout_market.copy()
    if use_raw_fallback:
        policy_frame["policy_forecast_f"] = policy_frame["forecast_f"]
    else:
        policy_frame["policy_forecast_f"] = policy_frame["emos_forecast_f"]

    holdout_actual = policy_frame["actual_f"].to_numpy(dtype=float)
    holdout_raw_forecast = policy_frame["forecast_f"].to_numpy(dtype=float)
    holdout_policy_forecast = policy_frame["policy_forecast_f"].to_numpy(dtype=float)

    temperature_diagnostics = _temperature_diagnostic_summary(policy_frame)
    probability_diagnostics, policy_probs, outcomes = _probability_diagnostic_summary(
        policy_frame,
        market_type,
        isotonic_model,
        isotonic_reason=isotonic_reason,
    )
    regime_diagnostics, diagnostic_limitations = _build_regime_diagnostics(
        policy_frame,
        market_type,
        isotonic_model,
        isotonic_reason=isotonic_reason,
    )
    temperature_diagnostics["regime_diagnostics"] = regime_diagnostics

    proxy_trade_selection = _proxy_trade_selection_summary(
        policy_probs,
        outcomes,
        total_examples=int(probability_diagnostics["holdout_probability_examples"] or 0),
    )

    raw_mae_f = _mean_absolute_error(holdout_actual, holdout_raw_forecast)
    policy_mae_f = _mean_absolute_error(holdout_actual, holdout_policy_forecast)
    raw_rmse_f = _root_mean_squared_error(holdout_actual, holdout_raw_forecast)
    policy_rmse_f = _root_mean_squared_error(holdout_actual, holdout_policy_forecast)

    return (
        {
            "temperature_source": temperature_source,
            "probability_input_source": probability_input_source,
            "probability_calibration_source": (
                "isotonic_broad_training" if isotonic_model is not None else "none"
            ),
            "holdout_rows": int(len(policy_frame)),
            "holdout_probability_examples": int(probability_diagnostics["holdout_probability_examples"] or 0),
            "raw_mae_f": raw_mae_f,
            "policy_mae_f": policy_mae_f,
            "policy_mae_delta_vs_raw_f": policy_mae_f - raw_mae_f,
            "raw_rmse_f": raw_rmse_f,
            "policy_rmse_f": policy_rmse_f,
            "policy_rmse_delta_vs_raw_f": policy_rmse_f - raw_rmse_f,
            "raw_brier": probability_diagnostics["raw_brier"],
            "policy_input_brier": probability_diagnostics["policy_input_brier"],
            "policy_brier": probability_diagnostics["policy_brier"],
            "policy_brier_delta_vs_raw": probability_diagnostics["policy_brier_delta_vs_raw"],
            "policy_brier_delta_vs_input": probability_diagnostics["policy_brier_delta_vs_input"],
            "temperature_verdict": _verdict_from_delta(policy_mae_f - raw_mae_f),
            "probability_verdict": _verdict_from_delta(probability_diagnostics["policy_brier_delta_vs_raw"]),
            "temperature_diagnostics": temperature_diagnostics,
            "probability_diagnostics": probability_diagnostics,
            "proxy_trade_selection": proxy_trade_selection,
        },
        diagnostic_limitations,
    )


def evaluate_market_type_holdout(
    city: str,
    city_training_df: pd.DataFrame,
    market_type: str,
    *,
    holdout_days: int,
    min_train_rows: int = 10,
    min_holdout_rows: int = 10,
) -> dict:
    total_rows = int(len(city_training_df))
    if city_training_df.empty:
        return _empty_result(
            city,
            market_type,
            status="skipped",
            reason="build_training_set returned no overlapping rows",
        )

    unique_dates = (
        pd.to_datetime(city_training_df["date"], errors="coerce").dropna().drop_duplicates().sort_values().tolist()
    )
    available_unique_dates = len(unique_dates)
    if available_unique_dates <= holdout_days:
        return _empty_result(
            city,
            market_type,
            status="skipped",
            reason=f"need more than {holdout_days} unique dates for a chronological holdout, found {available_unique_dates}",
            total_rows=total_rows,
            available_unique_dates=available_unique_dates,
        )

    train_df, holdout_df = split_chronological_holdout(city_training_df, holdout_days)
    train_market = _prepare_diagnostic_market_frame(train_df, market_type)
    holdout_market = _prepare_diagnostic_market_frame(holdout_df, market_type)

    train_start_date = str(train_df["date"].min()) if not train_df.empty else None
    train_end_date = str(train_df["date"].max()) if not train_df.empty else None
    holdout_start_date = str(holdout_df["date"].min()) if not holdout_df.empty else None
    holdout_end_date = str(holdout_df["date"].max()) if not holdout_df.empty else None

    if len(train_market) < min_train_rows:
        return _empty_result(
            city,
            market_type,
            status="skipped",
            reason=f"need at least {min_train_rows} train rows after the chronological split, found {len(train_market)}",
            total_rows=total_rows,
            available_unique_dates=available_unique_dates,
            train_rows=len(train_market),
            holdout_rows=len(holdout_market),
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            holdout_start_date=holdout_start_date,
            holdout_end_date=holdout_end_date,
        )

    if len(holdout_market) < min_holdout_rows:
        return _empty_result(
            city,
            market_type,
            status="skipped",
            reason=f"need at least {min_holdout_rows} holdout rows after the chronological split, found {len(holdout_market)}",
            total_rows=total_rows,
            available_unique_dates=available_unique_dates,
            train_rows=len(train_market),
            holdout_rows=len(holdout_market),
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            holdout_start_date=holdout_start_date,
            holdout_end_date=holdout_end_date,
        )

    emos_model = EMOSCalibrator(city=city, market_type=market_type).fit(train_market)
    holdout_market = holdout_market.copy()
    holdout_market["emos_forecast_f"] = np.asarray(
        [
            emos_model.correct(forecast_f, spread_f)
            for forecast_f, spread_f in zip(
                holdout_market["forecast_f"].to_numpy(dtype=float),
                holdout_market["spread_f"].to_numpy(dtype=float),
            )
        ],
        dtype=float,
    )

    train_probs, train_outcomes = build_isotonic_examples(train_market, market_type, emos_model=emos_model)
    isotonic_model: IsotonicCalibrator | None = None
    isotonic_reason = ""
    if train_probs.size < 10:
        isotonic_reason = f"need at least 10 isotonic training examples, found {train_probs.size}"
    elif len(np.unique(train_outcomes)) < 2:
        isotonic_reason = "need both positive and negative isotonic outcomes in the training split"
    else:
        isotonic_model = IsotonicCalibrator(city=city, market_type=market_type).fit(train_probs, train_outcomes)

    policies: dict[str, dict] = {}
    diagnostic_limitations: list[dict] = []
    for policy_name in _POLICY_ORDER:
        policy_result, policy_limitations = _evaluate_policy(
            city,
            market_type,
            holdout_market,
            isotonic_model,
            policy_name=policy_name,
            isotonic_reason=isotonic_reason or "isotonic model unavailable for this split",
        )
        policies[policy_name] = policy_result
        diagnostic_limitations.extend(policy_limitations)

    return {
        "city": city,
        "market_type": market_type,
        "pair": _pair_label(city, market_type),
        "is_targeted_fallback_pair": _is_targeted_fallback_pair(city, market_type),
        "status": "evaluated",
        "reason": isotonic_reason,
        "total_rows": total_rows,
        "available_unique_dates": available_unique_dates,
        "train_rows": int(len(train_market)),
        "holdout_rows": int(len(holdout_market)),
        "train_start_date": train_start_date,
        "train_end_date": train_end_date,
        "holdout_start_date": holdout_start_date,
        "holdout_end_date": holdout_end_date,
        "isotonic_train_examples": int(train_probs.size),
        "diagnostic_limitations": _dedupe_limitations(diagnostic_limitations),
        "policies": policies,
    }


def evaluate_city_holdout(
    city: str,
    *,
    days: int,
    holdout_days: int,
    min_train_rows: int,
    min_holdout_rows: int,
    station_actuals_dir: Path | str = STATION_ACTUALS_DIR,
    forecast_archive_dir: Path | str = FORECAST_ARCHIVE_DIR,
) -> list[dict]:
    city_training_df = build_training_set(
        city,
        days=days,
        station_actuals_dir=station_actuals_dir,
        forecast_archive_dir=forecast_archive_dir,
    )

    return [
        evaluate_market_type_holdout(
            city,
            city_training_df,
            market_type,
            holdout_days=holdout_days,
            min_train_rows=min_train_rows,
            min_holdout_rows=min_holdout_rows,
        )
        for market_type in ("high", "low")
    ]


def _weighted_metric(results: Iterable[dict], metric_key: str, weight_key: str) -> float | None:
    numerator = 0.0
    denominator = 0

    for result in results:
        metric = result.get(metric_key)
        weight = int(result.get(weight_key, 0) or 0)
        if metric is None or weight <= 0:
            continue
        numerator += float(metric) * weight
        denominator += weight

    if denominator <= 0:
        return None
    return numerator / denominator


def _aggregate_proxy(policy_results: list[dict], total_examples: int) -> dict:
    selected_examples = int(
        sum(int(result["proxy_trade_selection"].get("selected_examples", 0) or 0) for result in policy_results)
    )
    if total_examples <= 0 or selected_examples <= 0:
        return _empty_proxy_trade_selection("no probability proxy examples were selected in this aggregate slice")

    correct_direction_examples = int(
        sum(int(result["proxy_trade_selection"].get("correct_direction_examples", 0) or 0) for result in policy_results)
    )
    sum_abs_edge_vs_fair = float(
        sum(float(result["proxy_trade_selection"].get("sum_abs_edge_vs_fair", 0.0) or 0.0) for result in policy_results)
    )
    unit_even_odds_pnl_proxy = float(
        sum(float(result["proxy_trade_selection"].get("unit_even_odds_pnl_proxy", 0.0) or 0.0) for result in policy_results)
    )

    return {
        "status": "proxy",
        "reason": "",
        "confidence_threshold_abs_from_fair": _PROXY_CONFIDENCE_THRESHOLD,
        "market_price_assumption": _PROXY_MARKET_PRICE_ASSUMPTION,
        "selected_examples": selected_examples,
        "correct_direction_examples": correct_direction_examples,
        "selected_share": float(selected_examples / total_examples),
        "directional_hit_rate": float(correct_direction_examples / selected_examples),
        "sum_abs_edge_vs_fair": sum_abs_edge_vs_fair,
        "avg_abs_edge_vs_fair": float(sum_abs_edge_vs_fair / selected_examples),
        "unit_even_odds_pnl_proxy": unit_even_odds_pnl_proxy,
        "unit_even_odds_roi_proxy": float(unit_even_odds_pnl_proxy / (0.5 * selected_examples)),
        "note": (
            "Proxy only. This aggregate assumes threshold examples could be traded at a fair 0.50 line and does not "
            "replay historical quotes, liquidity, fees, or fills."
        ),
    }


def _aggregate_results(results: list[dict]) -> dict:
    aggregate: dict[str, dict] = {}

    for policy_name in _POLICY_ORDER:
        aggregate[policy_name] = {}
        for market_key in ("high", "low", "overall"):
            relevant = [item for item in results if market_key == "overall" or item["market_type"] == market_key]
            policy_results = [item["policies"][policy_name] for item in relevant]
            total_probability_examples = int(
                sum(int(result.get("holdout_probability_examples", 0) or 0) for result in policy_results)
            )

            aggregate[policy_name][market_key] = {
                "policy_name": policy_name,
                "city_market_pairs": len(relevant),
                "targeted_fallback_pairs": sum(1 for item in relevant if item["is_targeted_fallback_pair"]),
                "temperature_pairs": sum(1 for result in policy_results if result.get("policy_mae_f") is not None),
                "probability_pairs": sum(1 for result in policy_results if result.get("policy_brier") is not None),
                "skipped_pairs": sum(1 for item in relevant if item.get("status") == "skipped"),
                "temperature_helping_pairs": sum(
                    1 for result in policy_results if result.get("temperature_verdict") == "helps"
                ),
                "temperature_hurting_pairs": sum(
                    1 for result in policy_results if result.get("temperature_verdict") == "hurts"
                ),
                "probability_helping_pairs": sum(
                    1 for result in policy_results if result.get("probability_verdict") == "helps"
                ),
                "probability_hurting_pairs": sum(
                    1 for result in policy_results if result.get("probability_verdict") == "hurts"
                ),
                "total_train_rows": int(sum(int(item.get("train_rows", 0) or 0) for item in relevant)),
                "total_holdout_rows": int(sum(int(item.get("holdout_rows", 0) or 0) for item in relevant)),
                "total_holdout_probability_examples": total_probability_examples,
                "raw_mae_f": _safe_float(_weighted_metric(policy_results, "raw_mae_f", "holdout_rows")),
                "policy_mae_f": _safe_float(_weighted_metric(policy_results, "policy_mae_f", "holdout_rows")),
                "policy_mae_delta_vs_raw_f": _safe_float(
                    _weighted_metric(policy_results, "policy_mae_delta_vs_raw_f", "holdout_rows")
                ),
                "raw_rmse_f": _safe_float(_weighted_metric(policy_results, "raw_rmse_f", "holdout_rows")),
                "policy_rmse_f": _safe_float(_weighted_metric(policy_results, "policy_rmse_f", "holdout_rows")),
                "policy_rmse_delta_vs_raw_f": _safe_float(
                    _weighted_metric(policy_results, "policy_rmse_delta_vs_raw_f", "holdout_rows")
                ),
                "raw_brier": _safe_float(_weighted_metric(policy_results, "raw_brier", "holdout_probability_examples")),
                "policy_input_brier": _safe_float(
                    _weighted_metric(policy_results, "policy_input_brier", "holdout_probability_examples")
                ),
                "policy_brier": _safe_float(
                    _weighted_metric(policy_results, "policy_brier", "holdout_probability_examples")
                ),
                "policy_brier_delta_vs_raw": _safe_float(
                    _weighted_metric(policy_results, "policy_brier_delta_vs_raw", "holdout_probability_examples")
                ),
                "policy_brier_delta_vs_input": _safe_float(
                    _weighted_metric(policy_results, "policy_brier_delta_vs_input", "holdout_probability_examples")
                ),
                "proxy_trade_selection": _aggregate_proxy(policy_results, total_probability_examples),
            }

    return aggregate


def _regression_pairs(results: list[dict], policy_name: str, verdict_key: str) -> list[str]:
    return sorted(
        result["pair"]
        for result in results
        if result["policies"][policy_name].get(verdict_key) == "hurts"
    )


def _build_policy_comparison(results: list[dict], summary: dict) -> dict:
    aggregate_delta_by_market: dict[str, dict] = {}
    for market_key in ("high", "low", "overall"):
        broad = summary[_BROAD_POLICY][market_key]
        selective = summary[_SELECTIVE_POLICY][market_key]
        aggregate_delta_by_market[market_key] = {
            "policy_mae_f_delta_selective_minus_broad": _metric_delta(
                selective["policy_mae_f"],
                broad["policy_mae_f"],
            ),
            "policy_brier_delta_selective_minus_broad": _metric_delta(
                selective["policy_brier"],
                broad["policy_brier"],
            ),
            "temperature_hurting_pair_delta_selective_minus_broad": int(
                selective["temperature_hurting_pairs"] - broad["temperature_hurting_pairs"]
            ),
            "probability_hurting_pair_delta_selective_minus_broad": int(
                selective["probability_hurting_pairs"] - broad["probability_hurting_pairs"]
            ),
            "proxy_unit_even_odds_pnl_delta_selective_minus_broad": _metric_delta(
                selective["proxy_trade_selection"]["unit_even_odds_pnl_proxy"],
                broad["proxy_trade_selection"]["unit_even_odds_pnl_proxy"],
            ),
            "proxy_selected_examples_delta_selective_minus_broad": int(
                selective["proxy_trade_selection"]["selected_examples"]
                - broad["proxy_trade_selection"]["selected_examples"]
            ),
        }

    result_lookup = {(result["city"], result["market_type"]): result for result in results}
    targeted_pairs: list[dict] = []
    for city, market_type in SELECTIVE_RAW_FALLBACK_TARGETS:
        result = result_lookup.get((city, market_type))
        if result is None:
            targeted_pairs.append(
                {
                    "pair": _pair_label(city, market_type),
                    "status": "missing",
                    "reason": "pair not present in evaluation results",
                }
            )
            continue

        broad_policy = result["policies"][_BROAD_POLICY]
        selective_policy = result["policies"][_SELECTIVE_POLICY]
        targeted_pairs.append(
            {
                "pair": result["pair"],
                "broad_temperature_delta_vs_raw_f": broad_policy["policy_mae_delta_vs_raw_f"],
                "selective_temperature_delta_vs_raw_f": selective_policy["policy_mae_delta_vs_raw_f"],
                "temperature_delta_shift_f": _metric_delta(
                    selective_policy["policy_mae_delta_vs_raw_f"],
                    broad_policy["policy_mae_delta_vs_raw_f"],
                ),
                "broad_probability_delta_vs_raw": broad_policy["policy_brier_delta_vs_raw"],
                "selective_probability_delta_vs_raw": selective_policy["policy_brier_delta_vs_raw"],
                "probability_delta_shift": _metric_delta(
                    selective_policy["policy_brier_delta_vs_raw"],
                    broad_policy["policy_brier_delta_vs_raw"],
                ),
                "broad_temperature_verdict": broad_policy["temperature_verdict"],
                "selective_temperature_verdict": selective_policy["temperature_verdict"],
                "broad_probability_verdict": broad_policy["probability_verdict"],
                "selective_probability_verdict": selective_policy["probability_verdict"],
                "broad_proxy_unit_even_odds_pnl_proxy": broad_policy["proxy_trade_selection"]["unit_even_odds_pnl_proxy"],
                "selective_proxy_unit_even_odds_pnl_proxy": selective_policy["proxy_trade_selection"][
                    "unit_even_odds_pnl_proxy"
                ],
            }
        )

    broad_temperature_regressions = _regression_pairs(results, _BROAD_POLICY, "temperature_verdict")
    selective_temperature_regressions = _regression_pairs(results, _SELECTIVE_POLICY, "temperature_verdict")
    broad_probability_regressions = _regression_pairs(results, _BROAD_POLICY, "probability_verdict")
    selective_probability_regressions = _regression_pairs(results, _SELECTIVE_POLICY, "probability_verdict")
    targeted_labels = {_pair_label(city, market_type) for city, market_type in SELECTIVE_RAW_FALLBACK_TARGETS}

    return {
        "aggregate_delta_by_market": aggregate_delta_by_market,
        "targeted_pairs": targeted_pairs,
        "regression_pairs": {
            _BROAD_POLICY: {
                "temperature_pairs": broad_temperature_regressions,
                "probability_pairs": broad_probability_regressions,
            },
            _SELECTIVE_POLICY: {
                "temperature_pairs": selective_temperature_regressions,
                "probability_pairs": selective_probability_regressions,
            },
        },
        "net_new_regressions": {
            "temperature_pairs": sorted(set(selective_temperature_regressions) - set(broad_temperature_regressions)),
            "probability_pairs": sorted(set(selective_probability_regressions) - set(broad_probability_regressions)),
            "temperature_pairs_outside_targeted_list": sorted(
                pair
                for pair in set(selective_temperature_regressions) - set(broad_temperature_regressions)
                if pair not in targeted_labels
            ),
            "probability_pairs_outside_targeted_list": sorted(
                pair
                for pair in set(selective_probability_regressions) - set(broad_probability_regressions)
                if pair not in targeted_labels
            ),
            "resolved_temperature_pairs": sorted(
                set(broad_temperature_regressions) - set(selective_temperature_regressions)
            ),
            "resolved_probability_pairs": sorted(
                set(broad_probability_regressions) - set(selective_probability_regressions)
            ),
        },
    }


def build_report(
    cities: list[str],
    *,
    days: int,
    holdout_days: int,
    min_train_rows: int,
    min_holdout_rows: int,
    station_actuals_dir: Path | str = STATION_ACTUALS_DIR,
    forecast_archive_dir: Path | str = FORECAST_ARCHIVE_DIR,
) -> dict:
    results: list[dict] = []
    for city in cities:
        results.extend(
            evaluate_city_holdout(
                city,
                days=days,
                holdout_days=holdout_days,
                min_train_rows=min_train_rows,
                min_holdout_rows=min_holdout_rows,
                station_actuals_dir=station_actuals_dir,
                forecast_archive_dir=forecast_archive_dir,
            )
        )

    summary = _aggregate_results(results)
    return {
        "parameters": {
            "cities": cities,
            "days": int(days),
            "holdout_days": int(holdout_days),
            "min_train_rows": int(min_train_rows),
            "min_holdout_rows": int(min_holdout_rows),
            "station_actuals_dir": str(Path(station_actuals_dir)),
            "forecast_archive_dir": str(Path(forecast_archive_dir)),
            "targeted_raw_fallback_pairs": [
                _pair_label(city, market_type) for city, market_type in SELECTIVE_RAW_FALLBACK_TARGETS
            ],
        },
        "policy_definitions": _policy_definitions(),
        "proxy_trade_impact_note": _proxy_trade_impact_note(),
        "summary": summary,
        "comparison": _build_policy_comparison(results, summary),
        "diagnostic_limitations": _report_level_limitations(results),
        "results": results,
    }


def _reliability_bins(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Bin predictions into equal-width probability buckets and compute calibration stats."""
    if predictions.size == 0:
        return []

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (predictions >= lo) & (predictions < hi) if i < n_bins - 1 else (predictions >= lo) & (predictions <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(np.mean(predictions[mask]))
        mean_observed = float(np.mean(outcomes[mask]))
        bins.append({
            "bin_low": round(lo, 2),
            "bin_high": round(hi, 2),
            "count": count,
            "mean_predicted": round(mean_predicted, 4),
            "mean_observed": round(mean_observed, 4),
            "calibration_error": round(mean_predicted - mean_observed, 4),
        })
    return bins


def _add_reliability_to_report(report: dict) -> dict:
    """Compute reliability bins for each evaluated city-market pair and add to report."""
    for result in report.get("results", []):
        if result.get("status") != "evaluated":
            continue
        market_type = result["market_type"]
        city = result["city"]

        for policy_name in _POLICY_ORDER:
            policy = result["policies"].get(policy_name, {})
            prob_diag = policy.get("probability_diagnostics", {})
            if prob_diag.get("status") not in ("evaluated", "policy_input_only"):
                policy["reliability_bins"] = []
                continue
            # We need to recompute arrays for reliability bins
            # Store a marker — bins will be computed during build_report
            policy["reliability_bins"] = []

    return report


def _format_text_summary(report: dict) -> str:
    """Format a human-readable text summary of the evaluation report."""
    lines: list[str] = []
    params = report.get("parameters", {})
    lines.append("=" * 72)
    lines.append("  Calibration Evaluation Report")
    lines.append(f"  Window: {params.get('days')} days, Holdout: {params.get('holdout_days')} days")
    lines.append(f"  Cities: {len(params.get('cities', []))}")
    lines.append("=" * 72)
    lines.append("")

    # Summary table
    for policy_name in _POLICY_ORDER:
        policy_summary = report.get("summary", {}).get(policy_name, {}).get("overall", {})
        if not policy_summary:
            continue
        lines.append(f"  Policy: {policy_name}")
        lines.append(f"  {'-' *50}")

        pairs = policy_summary.get("city_market_pairs", 0)
        skipped = policy_summary.get("skipped_pairs", 0)
        lines.append(f"    Pairs evaluated: {pairs - skipped}/{pairs}")
        lines.append(f"    Train rows: {policy_summary.get('total_train_rows', 0):,}")
        lines.append(f"    Holdout rows: {policy_summary.get('total_holdout_rows', 0):,}")

        raw_mae = policy_summary.get("raw_mae_f")
        policy_mae = policy_summary.get("policy_mae_f")
        mae_delta = policy_summary.get("policy_mae_delta_vs_raw_f")
        if raw_mae is not None and policy_mae is not None:
            direction = "better" if mae_delta and mae_delta < 0 else "worse" if mae_delta and mae_delta > 0 else "flat"
            lines.append(f"    Temperature MAE: raw {raw_mae:.3f}F -> policy {policy_mae:.3f}F ({direction})")

        raw_brier = policy_summary.get("raw_brier")
        policy_brier = policy_summary.get("policy_brier")
        brier_delta = policy_summary.get("policy_brier_delta_vs_raw")
        if raw_brier is not None and policy_brier is not None:
            direction = "better" if brier_delta and brier_delta < 0 else "worse" if brier_delta and brier_delta > 0 else "flat"
            lines.append(f"    Probability Brier: raw {raw_brier:.4f} -> policy {policy_brier:.4f} ({direction})")

        proxy = policy_summary.get("proxy_trade_selection", {})
        if proxy.get("status") == "proxy":
            hit_rate = proxy.get("directional_hit_rate")
            pnl = proxy.get("unit_even_odds_pnl_proxy")
            selected = proxy.get("selected_examples", 0)
            if hit_rate is not None:
                lines.append(f"    Proxy trade selection: {selected} examples, {hit_rate:.1%} directional hit rate, PnL proxy {pnl:+.1f}")

        temp_helping = policy_summary.get("temperature_helping_pairs", 0)
        temp_hurting = policy_summary.get("temperature_hurting_pairs", 0)
        prob_helping = policy_summary.get("probability_helping_pairs", 0)
        prob_hurting = policy_summary.get("probability_hurting_pairs", 0)
        lines.append(f"    Temperature: {temp_helping} helped, {temp_hurting} hurt")
        lines.append(f"    Probability: {prob_helping} helped, {prob_hurting} hurt")
        lines.append("")

    # Per-city table
    results = report.get("results", [])
    evaluated = [r for r in results if r.get("status") == "evaluated"]
    if evaluated:
        lines.append("  Per-City Results (selective_raw_fallback policy)")
        lines.append(f"  {'-' *68}")
        lines.append(f"  {'City':<20} {'Type':<5} {'MAE d':>7} {'Brier d':>9} {'TempV':>7} {'ProbV':>7} {'Fallback':>8}")
        lines.append(f"  {'-' *68}")
        for r in sorted(evaluated, key=lambda x: (x["city"], x["market_type"])):
            policy = r["policies"].get(_SELECTIVE_POLICY, {})
            mae_d = policy.get("policy_mae_delta_vs_raw_f")
            brier_d = policy.get("policy_brier_delta_vs_raw")
            tv = policy.get("temperature_verdict", "?")[:5]
            pv = policy.get("probability_verdict", "?")[:5]
            fb = "yes" if r.get("is_targeted_fallback_pair") else ""
            mae_str = f"{mae_d:+.3f}" if mae_d is not None else "  N/A"
            brier_str = f"{brier_d:+.5f}" if brier_d is not None else "    N/A"
            lines.append(f"  {r['city']:<20} {r['market_type']:<5} {mae_str:>7} {brier_str:>9} {tv:>7} {pv:>7} {fb:>8}")
        lines.append("")

    # Comparison
    comparison = report.get("comparison", {})
    targeted = comparison.get("targeted_pairs", [])
    if targeted:
        lines.append("  Targeted Fallback Pairs (selective vs broad)")
        lines.append(f"  {'-' *55}")
        for tp in targeted:
            if tp.get("status") == "missing":
                lines.append(f"    {tp['pair']}: missing")
                continue
            temp_shift = tp.get("temperature_delta_shift_f")
            prob_shift = tp.get("probability_delta_shift")
            temp_str = f"MAE shift {temp_shift:+.3f}F" if temp_shift is not None else "MAE N/A"
            prob_str = f"Brier shift {prob_shift:+.5f}" if prob_shift is not None else "Brier N/A"
            lines.append(f"    {tp['pair']}: {temp_str}, {prob_str}")
        lines.append("")

    # Net new regressions
    net_new = comparison.get("net_new_regressions", {})
    temp_regressions = net_new.get("temperature_pairs_outside_targeted_list", [])
    prob_regressions = net_new.get("probability_pairs_outside_targeted_list", [])
    if temp_regressions or prob_regressions:
        lines.append("  WARNING: Net new regressions outside targeted list")
        if temp_regressions:
            lines.append(f"    Temperature: {', '.join(temp_regressions)}")
        if prob_regressions:
            lines.append(f"    Probability: {', '.join(prob_regressions)}")
        lines.append("")

    return "\n".join(lines)


def compare_reports(report_a_path: str, report_b_path: str) -> str:
    """Compare two saved evaluation reports and print a diff summary."""
    a = json.loads(Path(report_a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(report_b_path).read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  Evaluation Report Comparison")
    lines.append(f"  A: {report_a_path}")
    lines.append(f"  B: {report_b_path}")
    lines.append("=" * 72)
    lines.append("")

    for policy_name in _POLICY_ORDER:
        a_summary = a.get("summary", {}).get(policy_name, {}).get("overall", {})
        b_summary = b.get("summary", {}).get(policy_name, {}).get("overall", {})
        if not a_summary or not b_summary:
            continue

        lines.append(f"  Policy: {policy_name}")
        lines.append(f"  {'-' * 55}")

        metrics = [
            ("Temperature MAE (F)", "policy_mae_f", ".3f"),
            ("Raw MAE (F)", "raw_mae_f", ".3f"),
            ("Probability Brier", "policy_brier", ".4f"),
            ("Raw Brier", "raw_brier", ".4f"),
            ("MAE delta vs raw", "policy_mae_delta_vs_raw_f", "+.3f"),
            ("Brier delta vs raw", "policy_brier_delta_vs_raw", "+.5f"),
            ("Temp helping pairs", "temperature_helping_pairs", "d"),
            ("Temp hurting pairs", "temperature_hurting_pairs", "d"),
            ("Prob helping pairs", "probability_helping_pairs", "d"),
            ("Prob hurting pairs", "probability_hurting_pairs", "d"),
            ("Total holdout rows", "total_holdout_rows", ",d"),
        ]

        for label, key, fmt in metrics:
            val_a = a_summary.get(key)
            val_b = b_summary.get(key)
            if val_a is None and val_b is None:
                continue
            a_str = f"{val_a:{fmt}}" if val_a is not None else "N/A"
            b_str = f"{val_b:{fmt}}" if val_b is not None else "N/A"
            delta_str = ""
            if val_a is not None and val_b is not None:
                delta = val_b - val_a
                delta_str = f"  ({delta:+{fmt}})"
            lines.append(f"    {label:<25s}  A={a_str:>10s}  B={b_str:>10s}{delta_str}")

        # Proxy comparison
        a_proxy = a_summary.get("proxy_trade_selection", {})
        b_proxy = b_summary.get("proxy_trade_selection", {})
        if a_proxy.get("status") == "proxy" and b_proxy.get("status") == "proxy":
            a_hit = a_proxy.get("directional_hit_rate")
            b_hit = b_proxy.get("directional_hit_rate")
            a_pnl = a_proxy.get("unit_even_odds_pnl_proxy")
            b_pnl = b_proxy.get("unit_even_odds_pnl_proxy")
            if a_hit is not None and b_hit is not None:
                lines.append(f"    {'Proxy hit rate':<25s}  A={a_hit:>10.1%}  B={b_hit:>10.1%}  ({b_hit - a_hit:+.1%})")
            if a_pnl is not None and b_pnl is not None:
                lines.append(f"    {'Proxy PnL':<25s}  A={a_pnl:>10.1f}  B={b_pnl:>10.1f}  ({b_pnl - a_pnl:+.1f})")

        lines.append("")

    # Per-city deltas for the selective policy
    a_results = {(r["city"], r["market_type"]): r for r in a.get("results", []) if r.get("status") == "evaluated"}
    b_results = {(r["city"], r["market_type"]): r for r in b.get("results", []) if r.get("status") == "evaluated"}
    common_pairs = sorted(set(a_results.keys()) & set(b_results.keys()))

    if common_pairs:
        lines.append(f"  Per-City Brier Delta Changes ({_SELECTIVE_POLICY})")
        lines.append(f"  {'-' * 60}")
        lines.append(f"  {'City':<20} {'Type':<5} {'A Brier d':>10} {'B Brier d':>10} {'Change':>8}")
        lines.append(f"  {'-' * 60}")
        for city, mt in common_pairs:
            a_bd = a_results[(city, mt)]["policies"][_SELECTIVE_POLICY].get("policy_brier_delta_vs_raw")
            b_bd = b_results[(city, mt)]["policies"][_SELECTIVE_POLICY].get("policy_brier_delta_vs_raw")
            a_str = f"{a_bd:+.5f}" if a_bd is not None else "N/A"
            b_str = f"{b_bd:+.5f}" if b_bd is not None else "N/A"
            change = ""
            if a_bd is not None and b_bd is not None:
                d = b_bd - a_bd
                change = f"{d:+.5f}"
            lines.append(f"  {city:<20} {mt:<5} {a_str:>10} {b_str:>10} {change:>8}")
        lines.append("")

    # New/removed pairs
    a_only = sorted(set(a_results.keys()) - set(b_results.keys()))
    b_only = sorted(set(b_results.keys()) - set(a_results.keys()))
    if a_only:
        lines.append(f"  Pairs in A only: {', '.join(f'{c} {m}' for c, m in a_only)}")
    if b_only:
        lines.append(f"  Pairs in B only: {', '.join(f'{c} {m}' for c, m in b_only)}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw versus calibration-policy holdout performance")
    parser.add_argument("--city", action="append", dest="cities", help="Evaluate one city; can be repeated")
    parser.add_argument("--days", type=int, default=400, help="Rolling build_training_set window in days")
    parser.add_argument("--holdout-days", type=int, default=30, help="Newest forecast dates reserved for holdout")
    parser.add_argument("--min-train-rows", type=int, default=10, help="Minimum training rows after the split")
    parser.add_argument("--min-holdout-rows", type=int, default=10, help="Minimum holdout rows after the split")
    parser.add_argument("--station-actuals-dir", default=str(STATION_ACTUALS_DIR), help="Override station actuals directory")
    parser.add_argument("--forecast-archive-dir", default=str(FORECAST_ARCHIVE_DIR), help="Override forecast archive directory")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save JSON report to this file path")
    parser.add_argument("--json-only", action="store_true", help="Suppress text summary, print JSON only")
    parser.add_argument("--compare", nargs=2, metavar=("REPORT_A", "REPORT_B"), help="Compare two saved JSON reports instead of running evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.compare:
        comparison_text = compare_reports(args.compare[0], args.compare[1])
        try:
            print(comparison_text)
        except UnicodeEncodeError:
            print(comparison_text.encode("ascii", errors="replace").decode("ascii"))
        return

    cities = args.cities or discover_cities(
        station_actuals_dir=args.station_actuals_dir,
        forecast_archive_dir=args.forecast_archive_dir,
    )
    report = build_report(
        cities,
        days=args.days,
        holdout_days=args.holdout_days,
        min_train_rows=args.min_train_rows,
        min_holdout_rows=args.min_holdout_rows,
        station_actuals_dir=args.station_actuals_dir,
        forecast_archive_dir=args.forecast_archive_dir,
    )

    # Add reliability bins for evaluated pairs
    for result in report.get("results", []):
        if result.get("status") != "evaluated":
            continue
        market_type = result["market_type"]
        # Recompute holdout data for reliability bins
        city_training_df = build_training_set(
            result["city"],
            days=args.days,
            station_actuals_dir=args.station_actuals_dir,
            forecast_archive_dir=args.forecast_archive_dir,
        )
        if city_training_df.empty:
            continue
        _, holdout_df = split_chronological_holdout(city_training_df, args.holdout_days)
        holdout_market = _prepare_diagnostic_market_frame(holdout_df, market_type)
        if holdout_market.empty:
            continue

        # Fit EMOS on train
        train_df, _ = split_chronological_holdout(city_training_df, args.holdout_days)
        train_market = _prepare_diagnostic_market_frame(train_df, market_type)
        if len(train_market) < args.min_train_rows:
            continue

        emos_model = EMOSCalibrator(city=result["city"], market_type=market_type).fit(train_market)
        holdout_market = holdout_market.copy()
        holdout_market["emos_forecast_f"] = np.asarray(
            [emos_model.correct(f, s) for f, s in zip(
                holdout_market["forecast_f"].to_numpy(dtype=float),
                holdout_market["spread_f"].to_numpy(dtype=float),
            )],
            dtype=float,
        )

        train_probs, train_outcomes = build_isotonic_examples(train_market, market_type, emos_model=emos_model)
        isotonic_model: IsotonicCalibrator | None = None
        if train_probs.size >= 10 and len(np.unique(train_outcomes)) >= 2:
            isotonic_model = IsotonicCalibrator(city=result["city"], market_type=market_type).fit(train_probs, train_outcomes)

        for policy_name in _POLICY_ORDER:
            _, _, use_raw_fallback = _policy_sources(result["city"], market_type, policy_name)
            policy_frame = holdout_market.copy()
            if use_raw_fallback:
                policy_frame["policy_forecast_f"] = policy_frame["forecast_f"]
            else:
                policy_frame["policy_forecast_f"] = policy_frame["emos_forecast_f"]

            raw_probs, policy_input_probs, outcomes = _policy_probability_arrays(policy_frame, market_type)
            if raw_probs.size == 0:
                continue

            policy_probs = np.asarray(policy_input_probs, dtype=float)
            if isotonic_model is not None:
                policy_probs = np.asarray([isotonic_model.calibrate(p) for p in policy_input_probs], dtype=float)

            result["policies"][policy_name]["reliability_bins"] = _reliability_bins(policy_probs, outcomes)

    if not args.json_only:
        summary_text = _format_text_summary(report)
        try:
            print(summary_text)
        except UnicodeEncodeError:
            print(summary_text.encode("ascii", errors="replace").decode("ascii"))

    json_output = json.dumps(report, indent=2, sort_keys=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_output, encoding="utf-8")
        print(f"\nReport saved to: {output_path}")
    elif args.json_only:
        print(json_output)


if __name__ == "__main__":
    main()
