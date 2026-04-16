"""Tests for calibration module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.calibration import train_city_models, CalibrationManager


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
