"""Tests for `src.tail_training_data` — tail and bucket training-set builders."""

from pathlib import Path

import pandas as pd
import pytest

from src.tail_training_data import (
    build_bucket_training_set,
    build_tail_training_set,
)


def _write_actuals(path: Path, dates: list[str], tmax_f: list[float]) -> None:
    pd.DataFrame(
        {
            "date": dates,
            "tmax_f": tmax_f,
            "tmin_f": [None] * len(dates),
            "precip_in": [0.0] * len(dates),
            "precip_trace": [False] * len(dates),
            "cli_station": ["KNYC"] * len(dates),
            "source_url": [""] * len(dates),
            "city": ["New York"] * len(dates),
            "source": ["cli"] * len(dates),
            "archive_version": [1] * len(dates),
        }
    ).to_csv(path, index=False)


def _write_archive(
    path: Path,
    dates: list[str],
    forecast_high_f: list[float],
    ensemble_high_std_f: list[float],
) -> None:
    pd.DataFrame(
        {
            "as_of_utc": [f"{d}T12:00:00Z" for d in dates],
            "date": dates,
            "forecast_high_f": forecast_high_f,
            "forecast_low_f": [None] * len(dates),
            "ensemble_high_std_f": ensemble_high_std_f,
            "ensemble_low_std_f": [None] * len(dates),
            "forecast_model": ["open-meteo"] * len(dates),
            "forecast_lead_days": [1] * len(dates),
            "forecast_source": ["open-meteo"] * len(dates),
        }
    ).to_csv(path, index=False)


def test_build_tail_training_set_joins_forecasts_and_actuals(tmp_path):
    actuals_dir = tmp_path / "station_actuals"
    archive_dir = tmp_path / "forecast_archive"
    actuals_dir.mkdir()
    archive_dir.mkdir()

    dates = ["2026-04-01", "2026-04-02", "2026-04-03"]
    _write_actuals(actuals_dir / "new_york.csv", dates, [70.0, 82.0, 65.0])
    _write_archive(
        archive_dir / "new_york.csv",
        dates,
        forecast_high_f=[68.0, 75.0, 70.0],
        ensemble_high_std_f=[2.0, 3.0, 2.5],
    )

    df = build_tail_training_set(
        city="New York",
        market_type="high",
        direction="above",
        threshold=80.0,
        actuals_dir=actuals_dir,
        archive_dir=archive_dir,
    )

    assert list(df["date"]) == dates
    assert list(df["actual_exceeded_0_1"]) == [0, 1, 0]
    # raw_prob for 2026-04-02 with +0.99 offset: P(X > 80.99) with mean=75, std=3
    # → z=(80.99-75)/3 ≈ 1.997, P ≈ 0.023
    assert df["raw_prob"].iloc[1] == pytest.approx(0.023, abs=0.005)
    assert df["actual_exceeded_0_1"].dtype.kind in ("i", "u")


def test_build_bucket_training_set_joins_forecasts_and_actuals(tmp_path):
    actuals_dir = tmp_path / "station_actuals"
    archive_dir = tmp_path / "forecast_archive"
    actuals_dir.mkdir()
    archive_dir.mkdir()

    dates = ["2026-04-01", "2026-04-02"]
    _write_actuals(actuals_dir / "new_york.csv", dates, [70.5, 75.0])
    _write_archive(
        archive_dir / "new_york.csv",
        dates,
        forecast_high_f=[70.0, 76.0],
        ensemble_high_std_f=[2.0, 2.0],
    )

    df = build_bucket_training_set(
        city="New York",
        market_type="high",
        bucket_low=70.0,
        bucket_high=71.0,
        actuals_dir=actuals_dir,
        archive_dir=archive_dir,
    )

    assert list(df["date"]) == dates
    assert list(df["actual_in_bucket_0_1"]) == [1, 0]
    # Day-1: F(71) - F(70) with mean=70, std=2 → 0.6915 - 0.5 = 0.1915
    assert df["raw_bucket_prob"].iloc[0] == pytest.approx(0.1915, abs=0.002)
    assert df["actual_in_bucket_0_1"].dtype.kind in ("i", "u")


def test_raw_prob_above_matches_matcher_kalshi_threshold_convention():
    """Training-side raw_prob must use the same +0.99 offset as the live
    matcher's _kalshi_threshold_yes_probability for direction='above'."""
    from src.tail_training_data import _raw_prob_above
    from src.matcher import _kalshi_threshold_yes_probability

    for forecast, threshold, sigma in [
        (75.0, 80.0, 3.0),
        (65.0, 70.0, 2.5),
        (88.0, 85.0, 2.0),  # forecast above threshold
    ]:
        training = _raw_prob_above(forecast, threshold, sigma)
        serving = _kalshi_threshold_yes_probability("above", threshold, forecast, sigma)
        assert abs(training - serving) < 1e-9, (
            f"train/serve skew at forecast={forecast}, threshold={threshold}: "
            f"training={training}, serving={serving}"
        )


def test_raw_prob_below_matches_matcher_kalshi_threshold_convention():
    """Training-side raw_prob must use the same (no-offset) convention as the
    live matcher's _kalshi_threshold_yes_probability for direction='below'."""
    from src.tail_training_data import _raw_prob_below
    from src.matcher import _kalshi_threshold_yes_probability

    for forecast, threshold, sigma in [
        (75.0, 80.0, 3.0),
        (65.0, 70.0, 2.5),
        (60.0, 65.0, 2.0),
    ]:
        training = _raw_prob_below(forecast, threshold, sigma)
        serving = _kalshi_threshold_yes_probability("below", threshold, forecast, sigma)
        assert abs(training - serving) < 1e-9


def test_raw_prob_above_matches_matcher_after_sigma_clip(tmp_path):
    """The training builder must clip sigma to [1.0, 6.0] before the CDF,
    matching src/matcher.py._ensemble_sigma_for_date semantics. Otherwise
    the calibrator trains on a probability distribution that the live
    matcher never produces."""
    from src.tail_training_data import build_tail_training_set
    from src.matcher import _kalshi_threshold_yes_probability

    actuals_dir = tmp_path / "station_actuals"
    archive_dir = tmp_path / "forecast_archive"
    actuals_dir.mkdir()
    archive_dir.mkdir()

    dates = ["2030-01-01", "2030-01-02", "2030-01-03"]
    _write_actuals(actuals_dir / "new_york.csv", dates, [70.0, 75.0, 80.0])
    _write_archive(
        archive_dir / "new_york.csv",
        dates,
        forecast_high_f=[70.0, 70.0, 70.0],
        ensemble_high_std_f=[0.3, 3.0, 9.0],  # below, within, above clip
    )

    df = build_tail_training_set(
        city="New York",
        market_type="high",
        direction="above",
        threshold=75.0,
        actuals_dir=actuals_dir,
        archive_dir=archive_dir,
    )

    # Training raw_prob must equal serving probability with clipped sigma.
    for i, raw_sigma in enumerate([0.3, 3.0, 9.0]):
        clipped_sigma = max(1.0, min(6.0, raw_sigma))
        expected = _kalshi_threshold_yes_probability(
            "above", 75.0, 70.0, clipped_sigma,
        )
        assert df.iloc[i]["raw_prob"] == pytest.approx(expected, abs=1e-9), (
            f"sigma_raw={raw_sigma} clipped to {clipped_sigma}: "
            f"training={df.iloc[i]['raw_prob']}, serving_expected={expected}"
        )
    # Also assert the sigma_f column reflects the clip (not the raw archive value).
    assert list(df["sigma_f"]) == [1.0, 3.0, 6.0]


def test_build_bucket_training_set_rejects_inverted_bounds(tmp_path):
    from src.tail_training_data import build_bucket_training_set

    with pytest.raises(ValueError, match="bucket_low"):
        build_bucket_training_set(
            city="New York",
            market_type="high",
            bucket_low=71.0,
            bucket_high=70.0,
            actuals_dir=tmp_path / "a",
            archive_dir=tmp_path / "b",
        )
