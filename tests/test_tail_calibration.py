import numpy as np
import pytest


def test_tail_binary_calibrator_fit_and_predict():
    from src.tail_calibration import TailBinaryCalibrator

    rng = np.random.default_rng(7)
    n = 300
    raw_probs = rng.uniform(0.02, 0.98, n)
    # True rate is biased: tails are 2x the raw
    true_rate = np.clip(raw_probs * 1.3 + 0.02, 0.001, 0.999)
    outcomes = (rng.uniform(0, 1, n) < true_rate).astype(int)

    cal = TailBinaryCalibrator(city="New York", market_type="high", direction="above")
    cal.fit(raw_probs, outcomes)

    preds = np.array([cal.predict(p) for p in [0.1, 0.5, 0.9]])
    assert preds[0] < preds[1] < preds[2]  # monotone
    assert all(0.001 <= p <= 0.999 for p in preds)


def test_tail_binary_calibrator_save_load(tmp_path):
    from src.tail_calibration import TailBinaryCalibrator

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.95, 100)
    out = (rng.uniform(0, 1, 100) < raw).astype(int)

    cal = TailBinaryCalibrator(city="New York", market_type="high", direction="above")
    cal.fit(raw, out)
    path = tmp_path / "new_york_high_above_tail"
    cal.save(path)

    loaded = TailBinaryCalibrator.load(path)
    assert loaded.city == "New York"
    assert loaded.market_type == "high"
    assert loaded.direction == "above"
    assert abs(loaded.predict(0.4) - cal.predict(0.4)) < 1e-9
