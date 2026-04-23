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


def test_build_tail_training_set_fills_nan_sigma_with_fallback(tmp_path):
    """Training must mirror serving: NaN ensemble sigma → uncertainty_std_f
    (default 2.0), NOT drop the row. Otherwise training sees only ~23% of
    the archive (live-scan days with ensemble-sigma filled), which creates
    a recency bias the matcher never experiences at serve time.
    """
    import pandas as pd
    from src.tail_training_data import build_tail_training_set
    from src.matcher import _kalshi_threshold_yes_probability

    actuals_dir = tmp_path / "station_actuals"
    actuals_dir.mkdir()
    pd.DataFrame({
        "date": ["2026-04-01", "2026-04-02"],
        "tmax_f": [70.0, 82.0],
        "tmin_f": [55.0, 60.0],
        "precip_in": [0.0, 0.0],
        "precip_trace": [False, False],
        "cli_station": ["NYC", "NYC"],
        "source_url": ["", ""],
        "city": ["New York", "New York"],
        "source": ["cdo", "cdo"],
        "archive_version": ["", ""],
    }).to_csv(actuals_dir / "new_york.csv", index=False)

    archive_dir = tmp_path / "forecast_archive"
    archive_dir.mkdir()
    pd.DataFrame({
        "as_of_utc": ["2026-03-31T12:00:00+00:00", "2026-04-01T12:00:00+00:00"],
        "date": ["2026-04-01", "2026-04-02"],
        "forecast_high_f": [68.0, 75.0],
        "forecast_low_f": [52.0, 58.0],
        "ensemble_high_std_f": [None, 3.0],   # day 1 NaN, day 2 populated
        "ensemble_low_std_f": [None, 2.0],
        "forecast_model": ["best_match", "best_match"],
        "forecast_lead_days": [1, 1],
        "forecast_source": ["open_meteo_previous_runs"] * 2,
    }).to_csv(archive_dir / "new_york.csv", index=False)

    df = build_tail_training_set(
        city="New York", market_type="high", direction="above",
        threshold=80.0,
        actuals_dir=actuals_dir, archive_dir=archive_dir,
        uncertainty_std_f=2.0,
    )
    # BOTH rows must survive (not just the one with populated sigma)
    assert len(df) == 2
    assert list(df["date"]) == ["2026-04-01", "2026-04-02"]
    # Day 1 sigma was filled from 2.0 then clipped → 2.0
    assert df.iloc[0]["sigma_f"] == pytest.approx(2.0)
    # Day 1 raw_prob must equal serving probability with sigma=2.0
    expected_day1 = _kalshi_threshold_yes_probability("above", 80.0, 68.0, 2.0)
    assert df.iloc[0]["raw_prob"] == pytest.approx(expected_day1, abs=1e-9)


def test_build_bucket_training_set_fills_nan_sigma_with_fallback(tmp_path):
    """Bucket builder must apply the same NaN-sigma fallback as the tail
    builder and the live matcher.
    """
    import pandas as pd
    from src.tail_training_data import build_bucket_training_set

    actuals_dir = tmp_path / "station_actuals"
    actuals_dir.mkdir()
    pd.DataFrame({
        "date": ["2026-04-01", "2026-04-02"],
        "tmax_f": [70.5, 75.0],
        "tmin_f": [50.0, 55.0],
        "precip_in": [0.0, 0.0],
        "precip_trace": [False, False],
        "cli_station": ["NYC", "NYC"],
        "source_url": ["", ""],
        "city": ["New York", "New York"],
        "source": ["cdo", "cdo"],
        "archive_version": ["", ""],
    }).to_csv(actuals_dir / "new_york.csv", index=False)

    archive_dir = tmp_path / "forecast_archive"
    archive_dir.mkdir()
    pd.DataFrame({
        "as_of_utc": ["2026-03-31T12:00:00+00:00", "2026-04-01T12:00:00+00:00"],
        "date": ["2026-04-01", "2026-04-02"],
        "forecast_high_f": [70.0, 76.0],
        "forecast_low_f": [50.0, 55.0],
        "ensemble_high_std_f": [None, 2.0],
        "ensemble_low_std_f": [None, 1.5],
        "forecast_model": ["best_match", "best_match"],
        "forecast_lead_days": [1, 1],
        "forecast_source": ["open_meteo_previous_runs"] * 2,
    }).to_csv(archive_dir / "new_york.csv", index=False)

    df = build_bucket_training_set(
        city="New York", market_type="high",
        bucket_low=70.0, bucket_high=71.0,
        actuals_dir=actuals_dir, archive_dir=archive_dir,
        uncertainty_std_f=2.0,
    )
    assert len(df) == 2
    assert df.iloc[0]["sigma_f"] == pytest.approx(2.0)
