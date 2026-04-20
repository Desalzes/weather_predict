import numpy as np
import pytest


def test_logistic_rain_calibrator_fit_and_predict():
    from src.rain_calibration import LogisticRainCalibrator

    rng = np.random.default_rng(7)
    n = 300
    raw_probs = rng.uniform(0.05, 0.95, n)
    # True wet-day rate is biased vs raw_probs: wet_rate = sigmoid(2 * raw_prob - 1)
    true = 1 / (1 + np.exp(-(2.0 * raw_probs - 1.0)))
    outcomes = (rng.uniform(0, 1, n) < true).astype(int)

    cal = LogisticRainCalibrator(city="New York")
    cal.fit(raw_probs, outcomes)
    preds = np.array([cal.predict(p) for p in [0.1, 0.5, 0.9]])
    assert preds[0] < preds[1] < preds[2]  # monotone
    assert all(0.001 <= p <= 0.999 for p in preds)  # clipped


def test_logistic_rain_calibrator_save_load(tmp_path):
    from src.rain_calibration import LogisticRainCalibrator

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.95, 100)
    out = (rng.uniform(0, 1, 100) < raw).astype(int)

    cal = LogisticRainCalibrator(city="New York")
    cal.fit(raw, out)
    path = tmp_path / "ny_rain_logistic.pkl"
    cal.save(path)

    loaded = LogisticRainCalibrator.load(path)
    assert loaded.city == "New York"
    assert abs(loaded.predict(0.4) - cal.predict(0.4)) < 1e-9


def test_logistic_rain_calibrator_degenerate_outcomes_falls_through():
    """If outcomes are all 0 (or all 1) the model cannot fit - predict must
    fall through to raw (clipped) rather than raise or silently store junk."""
    from src.rain_calibration import LogisticRainCalibrator
    import numpy as np

    cal = LogisticRainCalibrator(city="New York")
    cal.fit(np.array([0.2, 0.5, 0.8]), np.array([0, 0, 0]))
    # Must not raise; predictions fall through to clipped raw probability
    assert cal.predict(0.5) == pytest.approx(0.5)
    assert cal.predict(0.0) == 0.001
    assert cal.predict(1.0) == 0.999


def test_isotonic_rain_calibrator_preserves_monotonicity():
    import numpy as np
    from src.rain_calibration import IsotonicRainCalibrator

    rng = np.random.default_rng(3)
    preds = np.linspace(0.02, 0.98, 200)
    outcomes = (rng.uniform(0, 1, 200) < preds).astype(int)
    cal = IsotonicRainCalibrator(city="New York")
    cal.fit(preds, outcomes)
    out = [cal.predict(p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
    assert all(b >= a - 1e-9 for a, b in zip(out, out[1:]))


def test_rain_calibration_manager_round_trip(tmp_path):
    import numpy as np
    from src.rain_calibration import (
        LogisticRainCalibrator, IsotonicRainCalibrator, RainCalibrationManager,
    )

    rng = np.random.default_rng(11)
    probs = rng.uniform(0.05, 0.95, 300)
    outcomes = (rng.uniform(0, 1, 300) < probs).astype(int)

    logistic = LogisticRainCalibrator(city="New York")
    logistic.fit(probs, outcomes)
    isotonic = IsotonicRainCalibrator(city="New York")
    isotonic.fit(probs, outcomes)
    logistic.save(tmp_path / "new_york_rain_binary_logistic.pkl")
    isotonic.save(tmp_path / "new_york_rain_binary_isotonic.pkl")

    mgr = RainCalibrationManager(model_dir=tmp_path)
    result = mgr.calibrate_rain_probability(city="New York", raw_prob=0.42)
    assert result is not None
    assert 0.001 <= result["calibrated_prob"] <= 0.999
    assert result["forecast_calibration_source"] == "logistic"
    assert result["probability_calibration_source"] == "isotonic"


def test_rain_calibration_manager_returns_none_when_no_models(tmp_path):
    from src.rain_calibration import RainCalibrationManager

    mgr = RainCalibrationManager(model_dir=tmp_path)
    assert mgr.calibrate_rain_probability(city="New York", raw_prob=0.5) is None
