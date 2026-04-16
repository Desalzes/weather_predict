"""Tests for calibration module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.calibration import train_city_models


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
