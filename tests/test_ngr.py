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


def test_ngr_fit_recovers_known_relationship():
    """Synthetic data where actual = forecast + seasonal bias, residual std grows with spread."""
    rng = np.random.default_rng(42)
    n = 500
    forecast = rng.uniform(60, 90, size=n)
    spread = rng.uniform(0.5, 3.0, size=n)
    doy = rng.integers(1, 366, size=n)
    lead_h = rng.choice([24, 48, 72], size=n).astype(float)

    # True generative process:
    # mu_true = forecast + 0.5 + 1.5*sin(2pi*doy/365)
    # sigma_true = 1.0 + 0.8*spread
    sin_doy = np.sin(2 * math.pi * doy / 365)
    cos_doy = np.cos(2 * math.pi * doy / 365)
    mu_true = forecast + 0.5 + 1.5 * sin_doy
    sigma_true = 1.0 + 0.8 * spread
    actual = mu_true + rng.normal(0, sigma_true)

    # Generate dates from DOY (more reliable than formatting)
    base_date = pd.Timestamp("2025-01-01")
    dates = [(base_date + pd.Timedelta(days=int(d-1))).strftime("%Y-%m-%d") for d in doy]

    df = pd.DataFrame({
        "forecast_high_f": forecast,
        "actual_high_f": actual,
        "ensemble_high_std_f": spread,
        "forecast_lead_days": lead_h / 24,
        "date": dates,
        "as_of_utc": ["2025-01-01T00:00:00+00:00"] * n,
    })

    cal = NGRCalibrator(city="Test", market_type="high").fit(df)

    assert cal.is_fitted
    assert cal.training_rows == n
    # Point prediction on the mean of the training range should be within 1F
    mu_pred, sigma_pred = cal.predict(forecast_f=75.0, spread_f=2.0, lead_h=48.0, doy=180)
    assert 74.0 < mu_pred < 78.0
    # Sigma should respond to spread
    _, sigma_low = cal.predict(forecast_f=75.0, spread_f=0.5, lead_h=48.0, doy=180)
    _, sigma_high = cal.predict(forecast_f=75.0, spread_f=3.0, lead_h=48.0, doy=180)
    assert sigma_high > sigma_low


def test_ngr_fit_raises_on_too_few_rows():
    df = pd.DataFrame({
        "forecast_high_f": [70.0, 72.0],
        "actual_high_f": [71.0, 73.0],
        "ensemble_high_std_f": [1.0, 1.0],
        "forecast_lead_days": [1, 1],
        "date": ["2025-04-01", "2025-04-02"],
        "as_of_utc": ["2025-03-31T12:00:00+00:00", "2025-04-01T12:00:00+00:00"],
    })
    with pytest.raises(ValueError, match="at least"):
        NGRCalibrator(city="Test", market_type="high").fit(df, min_rows=20)


def test_ngr_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    doy_vals = rng.integers(1, 366, size=n)
    base_date = pd.Timestamp("2025-01-01")
    dates = [(base_date + pd.Timedelta(days=int(d-1))).strftime("%Y-%m-%d") for d in doy_vals]

    df = pd.DataFrame({
        "forecast_high_f": rng.uniform(60, 90, n),
        "actual_high_f": rng.uniform(60, 90, n),
        "ensemble_high_std_f": rng.uniform(0.5, 2.5, n),
        "forecast_lead_days": [1] * n,
        "date": dates,
        "as_of_utc": ["2025-01-01T00:00:00+00:00"] * n,
    })

    original = NGRCalibrator(city="Test", market_type="high").fit(df)
    path = tmp_path / "ngr.pkl"
    original.save(path)

    loaded = NGRCalibrator.load(path)
    assert loaded.is_fitted
    assert loaded.training_rows == original.training_rows
    np.testing.assert_allclose(loaded.alpha, original.alpha)
    np.testing.assert_allclose(loaded.beta, original.beta)
    assert loaded.sigma2_floor == original.sigma2_floor

    # Predictions identical
    for _ in range(5):
        f = float(rng.uniform(60, 90))
        s = float(rng.uniform(0.5, 3.0))
        l = float(rng.choice([24, 48]))
        d = int(rng.integers(1, 366))
        assert original.predict(f, s, l, d) == loaded.predict(f, s, l, d)
