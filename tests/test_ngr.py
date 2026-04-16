"""Tests for Non-homogeneous Gaussian Regression."""

import math
import numpy as np
import pandas as pd
import pytest

from src.ngr import gaussian_crps, NGRCalibrator, build_ngr_features


def test_gaussian_crps_closed_form_zero_at_perfect_forecast():
    assert gaussian_crps(mu=10.0, sigma=1e-6, y=10.0) == pytest.approx(0.0, abs=1e-3)


def test_gaussian_crps_matches_known_value():
    expected = 2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)
    assert gaussian_crps(mu=5.0, sigma=1.0, y=5.0) == pytest.approx(expected, rel=1e-6)


def test_gaussian_crps_is_positive_and_monotone_in_residual():
    base = gaussian_crps(mu=0.0, sigma=1.0, y=0.0)
    bigger = gaussian_crps(mu=0.0, sigma=1.0, y=3.0)
    biggest = gaussian_crps(mu=0.0, sigma=1.0, y=6.0)
    assert base > 0
    assert bigger > base
    assert biggest > bigger


def test_gaussian_crps_vectorized_matches_scalar():
    mus = np.array([0.0, 1.0, -2.0])
    sigmas = np.array([1.0, 2.0, 0.5])
    ys = np.array([0.5, 1.0, -3.0])
    scalar = np.array([
        gaussian_crps(float(m), float(s), float(y))
        for m, s, y in zip(mus, sigmas, ys)
    ])
    vectorized = gaussian_crps(mus, sigmas, ys)
    assert np.allclose(vectorized, scalar, rtol=1e-10)


def test_build_ngr_features_adds_lead_hours_and_doy():
    df = pd.DataFrame({
        "forecast_high_f": [80.0, 82.0],
        "actual_high_f": [81.0, 83.0],
        "ensemble_high_std_f": [1.2, 1.5],
        "ensemble_std_f": [1.2, 1.5],
        "ensemble_low_std_f": [1.1, 1.3],
        "forecast_low_f": [60.0, 62.0],
        "actual_low_f": [61.0, 63.0],
        "forecast_lead_days": [1, 2],
        "date": ["2025-04-01", "2025-07-01"],
        "as_of_utc": ["2025-03-31T12:00:00+00:00", "2025-06-29T12:00:00+00:00"],
    })

    feats = build_ngr_features(df, market_type="high")

    # Columns needed for NGR fit
    for col in ["forecast_f", "actual_f", "spread_f", "lead_h", "doy", "sin_doy", "cos_doy"]:
        assert col in feats.columns, f"missing column {col}"

    # lead_h = forecast_lead_days * 24
    assert feats["lead_h"].tolist() == [24.0, 48.0]
    # doy from date
    assert feats["doy"].tolist() == [91, 182]  # Apr 1 = day 91, Jul 1 = day 182 (non-leap)
    # sin_doy and cos_doy consistent
    assert abs(feats["sin_doy"].iloc[0] - math.sin(2 * math.pi * 91 / 365)) < 1e-9


def test_ngr_calibrator_initial_state():
    cal = NGRCalibrator(city="Austin", market_type="high")
    assert cal.city == "Austin"
    assert cal.market_type == "high"
    assert cal.is_fitted is False
    assert cal.training_rows == 0
