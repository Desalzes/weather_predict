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

    # Directional check: fixture biases toward higher true rates than raw
    # (true = 1.3 * raw + 0.02). A fitted chain should nudge predict(0.5) up.
    assert cal.predict(0.5) > 0.55, (
        f"expected bias correction to push predict(0.5) above 0.55, got {cal.predict(0.5)}"
    )


def test_tail_binary_calibrator_degenerate_outcomes_falls_through():
    """All-zero outcomes must fall through to clipped raw probability."""
    from src.tail_calibration import TailBinaryCalibrator

    cal = TailBinaryCalibrator(city="New York", market_type="high", direction="above")
    cal.fit(np.array([0.2, 0.5, 0.8]), np.array([0, 0, 0]))
    # Both stages set _model = None; predict returns clipped raw probability
    assert cal.predict(0.5) == pytest.approx(0.5)
    assert cal.predict(0.0) == 0.001
    assert cal.predict(1.0) == 0.999


def test_tail_binary_calibrator_all_ones_outcomes_falls_through():
    """All-one outcomes must also fall through cleanly (no crash)."""
    from src.tail_calibration import TailBinaryCalibrator

    cal = TailBinaryCalibrator(city="New York", market_type="low", direction="below")
    cal.fit(np.array([0.2, 0.5, 0.8]), np.array([1, 1, 1]))
    assert cal.predict(0.5) == pytest.approx(0.5)
    assert cal.predict(0.0) == 0.001
    assert cal.predict(1.0) == 0.999


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


def test_bucket_distributional_calibrator_fit_and_predict():
    from src.tail_calibration import BucketDistributionalCalibrator

    rng = np.random.default_rng(11)
    n = 200
    raw_probs = rng.uniform(0.05, 0.45, n)
    # True bucket rate is 1.2x raw (systematic underconfidence)
    true_rate = np.clip(raw_probs * 1.2, 0.001, 0.999)
    outcomes = (rng.uniform(0, 1, n) < true_rate).astype(int)

    cal = BucketDistributionalCalibrator(city="New York", market_type="high")
    cal.fit(raw_probs, outcomes)

    preds = [cal.predict(p) for p in [0.05, 0.2, 0.4]]
    assert preds[0] < preds[1] < preds[2]
    assert all(0.001 <= p <= 0.999 for p in preds)

    # Fixture biases true_rate to 1.2x raw; fitted chain should nudge predict up.
    assert cal.predict(0.3) > 0.3, (
        f"expected bias correction to push predict(0.3) above raw, got {cal.predict(0.3)}"
    )


def test_bucket_calibrator_save_load(tmp_path):
    from src.tail_calibration import BucketDistributionalCalibrator

    rng = np.random.default_rng(13)
    raw = rng.uniform(0.05, 0.40, 100)
    out = (rng.uniform(0, 1, 100) < raw).astype(int)

    cal = BucketDistributionalCalibrator(city="New York", market_type="high")
    cal.fit(raw, out)
    path = tmp_path / "new_york_high_bucket"
    cal.save(path)

    loaded = BucketDistributionalCalibrator.load(path)
    assert loaded.city == "New York"
    assert loaded.market_type == "high"
    assert abs(loaded.predict(0.2) - cal.predict(0.2)) < 1e-9


def test_bucket_distributional_calibrator_degenerate_outcomes_falls_through():
    from src.tail_calibration import BucketDistributionalCalibrator

    cal = BucketDistributionalCalibrator(city="New York", market_type="high")
    cal.fit(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))
    assert cal.predict(0.15) == pytest.approx(0.15)
    assert cal.predict(0.0) == 0.001
    assert cal.predict(1.0) == 0.999


def test_tail_calibration_manager_threshold_path(tmp_path):
    from src.tail_calibration import TailBinaryCalibrator, TailCalibrationManager

    rng = np.random.default_rng(17)
    probs = rng.uniform(0.05, 0.95, 200)
    outcomes = (rng.uniform(0, 1, 200) < probs).astype(int)
    cal = TailBinaryCalibrator(city="New York", market_type="high", direction="above")
    cal.fit(probs, outcomes)
    save_prefix = tmp_path / "new_york_high_above_tail"
    cal.save(save_prefix)

    mgr = TailCalibrationManager(model_dir=tmp_path)
    result = mgr.calibrate_tail_probability(
        city="New York",
        market_type="high",
        direction="above",
        is_bucket=False,
        raw_prob=0.25,
    )
    assert result is not None
    assert 0.001 <= result["calibrated_prob"] <= 0.999
    assert result["source"] == "logistic+isotonic"


def test_tail_calibration_manager_returns_none_when_no_model(tmp_path):
    from src.tail_calibration import TailCalibrationManager
    mgr = TailCalibrationManager(model_dir=tmp_path)
    assert mgr.calibrate_tail_probability(
        city="New York", market_type="high", direction="above",
        is_bucket=False, raw_prob=0.25,
    ) is None
    assert mgr.calibrate_tail_probability(
        city="New York", market_type="high", direction="above",
        is_bucket=True, raw_prob=0.25,
    ) is None


def test_tail_calibration_manager_bucket_path(tmp_path):
    from src.tail_calibration import BucketDistributionalCalibrator, TailCalibrationManager
    rng = np.random.default_rng(19)
    probs = rng.uniform(0.05, 0.40, 150)
    outcomes = (rng.uniform(0, 1, 150) < probs).astype(int)
    cal = BucketDistributionalCalibrator(city="New York", market_type="high")
    cal.fit(probs, outcomes)
    cal.save(tmp_path / "new_york_high_bucket")

    mgr = TailCalibrationManager(model_dir=tmp_path)
    result = mgr.calibrate_tail_probability(
        city="New York", market_type="high",
        direction="above",  # ignored for is_bucket=True
        is_bucket=True, raw_prob=0.20,
    )
    assert result is not None
    assert result["source"] == "logistic+isotonic"


def test_tail_calibration_manager_picks_up_retrained_model(tmp_path):
    """mtime invalidation — retrained pkl is picked up without restart."""
    import os
    import time
    from src.tail_calibration import TailBinaryCalibrator, TailCalibrationManager

    save_prefix = tmp_path / "new_york_high_above_tail"

    rng = np.random.default_rng(23)
    biased_probs = np.concatenate([rng.uniform(0.05, 0.15, 200), rng.uniform(0.75, 0.95, 20)])
    biased_outcomes = np.concatenate([np.zeros(200, dtype=int), np.ones(20, dtype=int)])
    v1 = TailBinaryCalibrator(city="New York", market_type="high", direction="above")
    v1.fit(biased_probs, biased_outcomes)
    v1.save(save_prefix)

    mgr = TailCalibrationManager(model_dir=tmp_path)
    first = mgr.calibrate_tail_probability(
        city="New York", market_type="high", direction="above",
        is_bucket=False, raw_prob=0.5,
    )
    assert first is not None

    time.sleep(0.01)
    future_time = os.path.getmtime(f"{save_prefix}_logistic.pkl") + 10

    opposite_probs = np.concatenate([rng.uniform(0.05, 0.15, 20), rng.uniform(0.75, 0.95, 200)])
    opposite_outcomes = np.concatenate([np.zeros(20, dtype=int), np.ones(200, dtype=int)])
    v2 = TailBinaryCalibrator(city="New York", market_type="high", direction="above")
    v2.fit(opposite_probs, opposite_outcomes)
    v2.save(save_prefix)
    for suffix in ("_logistic.pkl", "_isotonic.pkl", "_meta.pkl"):
        os.utime(f"{save_prefix}{suffix}", (future_time, future_time))

    second = mgr.calibrate_tail_probability(
        city="New York", market_type="high", direction="above",
        is_bucket=False, raw_prob=0.5,
    )
    assert second is not None
    assert abs(first["calibrated_prob"] - second["calibrated_prob"]) > 0.05


def test_tail_calibration_manager_picks_up_newly_appeared_model(tmp_path):
    """A model that didn't exist at first query appears later — manager reloads."""
    import numpy as np
    from src.tail_calibration import TailBinaryCalibrator, TailCalibrationManager

    mgr = TailCalibrationManager(model_dir=tmp_path)
    assert mgr.calibrate_tail_probability(
        city="New York", market_type="high", direction="above",
        is_bucket=False, raw_prob=0.5,
    ) is None

    rng = np.random.default_rng(29)
    probs = rng.uniform(0.05, 0.95, 150)
    outcomes = (rng.uniform(0, 1, 150) < probs).astype(int)
    cal = TailBinaryCalibrator(city="New York", market_type="high", direction="above")
    cal.fit(probs, outcomes)
    cal.save(tmp_path / "new_york_high_above_tail")

    result = mgr.calibrate_tail_probability(
        city="New York", market_type="high", direction="above",
        is_bucket=False, raw_prob=0.5,
    )
    assert result is not None
