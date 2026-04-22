"""Tests for calibration module."""

import os
import time

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.calibration import (
    CalibrationManager,
    EMOSCalibrator,
    calibration_model_path,
    train_city_models,
)


def test_train_city_models_fits_ngr_when_enough_rows(tmp_path):
    """Test that NGR model is trained and saved when sufficient data exists."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "date": [f"2025-{((i%12)+1):02d}-{((i%28)+1):02d}" for i in range(n)],
        "forecast_high_f": rng.uniform(60, 90, n),
        "actual_high_f": rng.uniform(60, 90, n),
        "forecast_low_f": rng.uniform(40, 70, n),
        "actual_low_f": rng.uniform(40, 70, n),
        "ensemble_high_std_f": rng.uniform(0.5, 2.5, n),
        "ensemble_low_std_f": rng.uniform(0.5, 2.5, n),
        "ensemble_std_f": rng.uniform(0.5, 2.5, n),
        "forecast_lead_days": [1] * n,
        "as_of_utc": ["2025-01-01T00:00:00+00:00"] * n,
    })

    results = train_city_models("Austin", df, model_dir=tmp_path, min_training_rows=10)
    for market_type in ("high", "low"):
        outcome = results[market_type]
        assert outcome["trained_ngr"] is True
        assert outcome["ngr_path"] is not None
        ngr_path = Path(outcome["ngr_path"])
        assert ngr_path.exists()
        assert ngr_path.name == f"austin_{market_type}_ngr.pkl"


def test_calibration_manager_predict_distribution_uses_ngr_when_available(tmp_path):
    import numpy as np
    rng = np.random.default_rng(1)
    n = 80
    df = pd.DataFrame({
        "date": [f"2025-{((i%12)+1):02d}-{((i%28)+1):02d}" for i in range(n)],
        "forecast_high_f": rng.uniform(70, 85, n),
        "actual_high_f": rng.uniform(70, 85, n),
        "ensemble_high_std_f": rng.uniform(1.0, 2.0, n),
        "ensemble_std_f": rng.uniform(1.0, 2.0, n),
        "forecast_lead_days": [1] * n,
        "as_of_utc": ["2025-01-01T00:00:00+00:00"] * n,
    })
    train_city_models("Austin", df, model_dir=tmp_path, min_training_rows=10)

    manager = CalibrationManager(model_dir=tmp_path)
    mu, sigma, source = manager.predict_distribution(
        city="Austin", market_type="high",
        forecast_f=80.0, spread_f=1.5, lead_h=24.0, doy=150,
    )
    assert source == "ngr"
    assert 60.0 < mu < 100.0
    assert 0.25 < sigma < 12.0


def test_calibration_manager_predict_distribution_falls_back_to_raw_when_nothing_trained(tmp_path):
    manager = CalibrationManager(model_dir=tmp_path)
    mu, sigma, source = manager.predict_distribution(
        city="UnknownCity", market_type="high",
        forecast_f=75.0, spread_f=2.0, lead_h=24.0, doy=100,
    )
    assert source == "raw"
    assert mu == 75.0
    assert sigma == 2.0


def test_calibration_manager_predict_distribution_respects_selective_fallback(tmp_path):
    # Train a model for Boston low, then verify selective fallback pair returns raw.
    import numpy as np
    rng = np.random.default_rng(2)
    n = 80
    df = pd.DataFrame({
        "date": [f"2025-{((i%12)+1):02d}-{((i%28)+1):02d}" for i in range(n)],
        "forecast_low_f": rng.uniform(30, 55, n),
        "actual_low_f": rng.uniform(30, 55, n),
        "ensemble_low_std_f": rng.uniform(1.0, 2.5, n),
        "ensemble_std_f": rng.uniform(1.0, 2.5, n),
        "forecast_lead_days": [1] * n,
        "as_of_utc": ["2025-01-01T00:00:00+00:00"] * n,
    })
    train_city_models("Boston", df, model_dir=tmp_path, min_training_rows=10)

    manager = CalibrationManager(model_dir=tmp_path)
    mu, sigma, source = manager.predict_distribution(
        city="Boston", market_type="low",
        forecast_f=40.0, spread_f=2.0, lead_h=24.0, doy=100,
    )
    assert source == "raw_selective_fallback"
    assert mu == 40.0  # unchanged
    assert sigma >= 0.25  # sigma still produced (so matcher has something to work with)


def test_calibration_manager_picks_up_retrained_emos_model(tmp_path):
    """Manager must reload an EMOS model whose pkl file was rewritten after
    the cache entry was created. Simulates the weekly retrain scenario where
    a long-running run_loop would otherwise serve a stale model forever."""
    city = "Austin"
    market_type = "high"
    path = calibration_model_path(city, market_type, "emos", model_dir=tmp_path)

    # Version 1: adds +1°F bias.
    v1 = EMOSCalibrator(
        city=city, market_type=market_type,
        a=1.0, b=1.0, c=0.0,
        training_rows=20, is_fitted=True,
    )
    v1.save(path)

    manager = CalibrationManager(model_dir=tmp_path)
    first, _ = manager.correct_forecast(
        city=city, market_type=market_type,
        forecast_f=80.0, spread_f=1.5,
    )
    assert first == pytest.approx(81.0)

    # Version 2: adds +5°F bias; overwrite the same path and bump mtime.
    time.sleep(0.01)
    future_time = os.path.getmtime(path) + 10
    v2 = EMOSCalibrator(
        city=city, market_type=market_type,
        a=5.0, b=1.0, c=0.0,
        training_rows=20, is_fitted=True,
    )
    v2.save(path)
    os.utime(path, (future_time, future_time))

    second, _ = manager.correct_forecast(
        city=city, market_type=market_type,
        forecast_f=80.0, spread_f=1.5,
    )
    assert second == pytest.approx(85.0)


def test_calibration_manager_picks_up_newly_appeared_emos_model(tmp_path):
    """When no EMOS model exists at startup, the manager caches None. When a
    model later appears on disk, the manager must pick it up on the next
    call rather than continuing to serve the cached None."""
    city = "Austin"
    market_type = "high"

    manager = CalibrationManager(model_dir=tmp_path)
    # First call: no model -> falls back to raw.
    forecast_before, source_before = manager.correct_forecast(
        city=city, market_type=market_type,
        forecast_f=80.0, spread_f=1.5,
    )
    assert source_before == "raw"
    assert forecast_before == pytest.approx(80.0)

    # Now write a model and expect the next call to pick it up.
    path = calibration_model_path(city, market_type, "emos", model_dir=tmp_path)
    EMOSCalibrator(
        city=city, market_type=market_type,
        a=3.0, b=1.0, c=0.0,
        training_rows=20, is_fitted=True,
    ).save(path)

    forecast_after, source_after = manager.correct_forecast(
        city=city, market_type=market_type,
        forecast_f=80.0, spread_f=1.5,
    )
    assert source_after == "emos"
    assert forecast_after == pytest.approx(83.0)
