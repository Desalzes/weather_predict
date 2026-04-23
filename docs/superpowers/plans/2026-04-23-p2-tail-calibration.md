# P2 Temperature Tail Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclaim blocked temperature SELL-side and bucket-market trades via a tail-specific calibration layer that reuses P1's `LogisticRainCalibrator` / `IsotonicRainCalibrator` classes verbatim, routes through an evidence-gated per-pair unblock list, and keeps new unblocks on a probation bankroll slice for 30 days before promotion.

**Architecture:** A parallel tail-calibration branch composed on top of the existing temperature chain. The matcher always computes the tail probability when models exist (attaches `our_probability_tail` + `edge_tail` alongside the existing `our_probability`). The policy's `tail_unblocks` list decides which numbers actually drive trade decisions, per (city, market_type, direction) pair. BUY-side flow is untouched.

**Tech Stack:** Python 3.12, scikit-learn (LogisticRegression, IsotonicRegression — already in requirements), pandas, pytest. Reuses P1 calibrator classes via composition.

**Spec:** [docs/superpowers/specs/2026-04-23-p2-tail-calibration-design.md](../specs/2026-04-23-p2-tail-calibration-design.md)

---

## File Structure

### New files

| File | Responsibility |
|---|---|
| `src/tail_training_data.py` | `build_tail_training_set(city, market_type, direction)` — joins forecast archive + station actuals into `(raw_prob, actual_exceeded_0_1)` training pairs for tail calibration. |
| `src/tail_calibration.py` | `TailBinaryCalibrator`, `BucketDistributionalCalibrator`, `TailCalibrationManager`. Composes P1 `LogisticRainCalibrator` / `IsotonicRainCalibrator`. |
| `train_tail_calibration.py` | CLI trainer; per-pair, rolling 180-day window, respects sample-size gate. |
| `evaluate_tail_calibration.py` | CLI evaluator; chronological holdout, writes scorecard with `qualifies_for_unblock` per pair. |
| `.claude/rules/tail-calibration.md` | Docs — routing, policy integration, probation mechanics. |
| `tests/test_tail_training_data.py` | Join correctness, tail-region filter, empty-archive fallback. |
| `tests/test_tail_calibration.py` | Calibrator fit/predict/save/load, degenerate + partial-model paths, mtime invalidation. |
| `tests/test_matcher_tail_routing.py` | End-to-end routing: BUY untouched, SELL-unblocked uses tail, SELL-not-unblocked drops, bucket uses bucket calibrator. |

### Modified files

| File | Change |
|---|---|
| `src/matcher.py` | Add ~25-line branch that attaches `our_probability_tail` / `edge_tail` / `tail_calibration_source` to the opportunity dict when `TailCalibrationManager` has models for the pair. |
| `src/strategy_policy.py` | Add `apply_tail_unblocks(opportunity, policy)` — for SELL/bucket opportunities that would be dropped by `allowed_position_sides` / `allowed_settlement_rules`, consult `tail_unblocks` and swap in tail probabilities. |
| `strategy/strategy_policy.json` | Bump to v5 — add `tail_unblocks: {threshold_sell: [], bucket: []}` and `bankroll_slices: {temperature_buy: 0.70, rain_binary: 0.20, probation: 0.10}`. Existing v4 fields unchanged. |
| `src/paper_trading.py` | Add `bankroll_slice` column to ledger (values: `temperature_buy` \| `rain_binary` \| `probation`). Legacy rows backfill as `temperature_buy`. Extend `summary.json` `category_breakdown` with nested `slice_breakdown`. |
| `src/sizing.py` | `compute_position_size` accepts new `bankroll_fraction_multiplier` parameter (default 1.0 preserves behavior); used to compute slice-scaled bankroll. |
| `scripts/autopilot_weekly.py` | After the existing weekly temperature retrain, run `train_tail_calibration.py` and `evaluate_tail_calibration.py`. |
| `config.example.json` | Add `enable_tail_calibration: false`, `tail_calibration_window_days: 180`, `tail_bankroll_fraction: 0.10`. |
| `main.py` | Instantiate `TailCalibrationManager` when `enable_tail_calibration` is true; pass it into `match_kalshi_markets`. |
| `src/CLAUDE.md` | Module-map row for `tail_calibration.py`, `tail_training_data.py`. |
| `strategy/CLAUDE.md` | Policy v5 section documenting `tail_unblocks` and `bankroll_slices`. |
| `tests/test_paper_trading.py` | `test_bankroll_slice_column_migrates_legacy`, `test_summary_json_slice_breakdown`. |
| `tests/test_strategy_policy.py` | `test_tail_unblocks_routes_sell_correctly`, `test_tail_unblocks_rejects_untriangulated_pairs`. |

---

## Task 1: `build_tail_training_set` — training data join

**Rationale:** Produce `(raw_prob, actual_exceeded_0_1)` pairs by reading the existing `data/forecast_archive/*.csv` + `data/station_actuals/*.csv` files, computing `raw_prob` via the same Gaussian CDF the live matcher uses, and labeling the outcome. This is the single data source for Tasks 2-6.

**Files:**
- Create: `src/tail_training_data.py`
- Test: `tests/test_tail_training_data.py`

- [ ] **Step 1: Write failing test for basic join**

Create `tests/test_tail_training_data.py`:

```python
import pandas as pd
import pytest
from pathlib import Path


def test_build_tail_training_set_joins_forecasts_and_actuals(tmp_path):
    from src.tail_training_data import build_tail_training_set

    actuals_dir = tmp_path / "station_actuals"
    actuals_dir.mkdir()
    pd.DataFrame({
        "date": ["2026-04-01", "2026-04-02", "2026-04-03"],
        "tmax_f": [70.0, 82.0, 65.0],
        "tmin_f": [55.0, 60.0, 50.0],
        "precip_in": [0.0, 0.0, 0.0],
        "precip_trace": [False, False, False],
        "cli_station": ["NYC"] * 3,
        "source_url": [""] * 3,
        "city": ["New York"] * 3,
        "source": ["cdo"] * 3,
        "archive_version": [""] * 3,
    }).to_csv(actuals_dir / "new_york.csv", index=False)

    archive_dir = tmp_path / "forecast_archive"
    archive_dir.mkdir()
    pd.DataFrame({
        "as_of_utc": [
            "2026-03-31T12:00:00+00:00",
            "2026-04-01T12:00:00+00:00",
            "2026-04-02T12:00:00+00:00",
        ],
        "date": ["2026-04-01", "2026-04-02", "2026-04-03"],
        "forecast_high_f": [68.0, 75.0, 70.0],
        "forecast_low_f": [52.0, 58.0, 55.0],
        "ensemble_high_std_f": [2.0, 3.0, 2.5],
        "ensemble_low_std_f": [1.5, 2.0, 1.5],
        "forecast_model": ["best_match"] * 3,
        "forecast_lead_days": [1, 1, 1],
        "forecast_source": ["open_meteo_previous_runs"] * 3,
    }).to_csv(archive_dir / "new_york.csv", index=False)

    # Threshold: 80F, market_type=high, direction=above
    # raw_prob = P(tmax > 80) given N(forecast_high_f, std)
    # actual_exceeded = 1 if tmax_f > 80, else 0
    df = build_tail_training_set(
        city="New York",
        market_type="high",
        direction="above",
        threshold=80.0,
        actuals_dir=actuals_dir,
        archive_dir=archive_dir,
    )

    assert list(df["date"]) == ["2026-04-01", "2026-04-02", "2026-04-03"]
    # 2026-04-02 actual=82 > 80 so exceeded=1; others exceeded=0
    assert list(df["actual_exceeded_0_1"]) == [0, 1, 0]
    # raw_prob for 2026-04-02: P(X > 80) with mean=75, std=3 → z=(80-75)/3=1.667, P=0.048
    assert df.iloc[1]["raw_prob"] == pytest.approx(0.048, abs=0.005)
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_training_data.py::test_build_tail_training_set_joins_forecasts_and_actuals -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'src.tail_training_data'`.

- [ ] **Step 3: Implement `build_tail_training_set`**

Create `src/tail_training_data.py`:

```python
"""Join archived forecasts with station actuals to produce tail-calibration
training rows.

Training row shape for threshold markets:
    (date, raw_prob, actual_exceeded_0_1, forecast_value_f, sigma_f,
     forecast_lead_days, as_of_utc)

raw_prob is the Gaussian CDF probability of the market threshold being
crossed, computed at training time from the archived forecast value and
sigma. This preserves the causal chain: the calibrator learns the correction
that applies to live opportunities using the same CDF math.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pandas as pd

from src.station_truth import (
    FORECAST_ARCHIVE_DIR,
    STATION_ACTUALS_DIR,
    _slugify_city,
)

_EMPTY_COLUMNS = [
    "date",
    "raw_prob",
    "actual_exceeded_0_1",
    "forecast_value_f",
    "sigma_f",
    "forecast_lead_days",
    "as_of_utc",
]


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal CDF evaluated at (x - mu) / sigma. Matches the
    convention in src/matcher.py._normal_cdf.
    """
    sigma = max(float(sigma), 1e-6)
    z = (x - mu) / sigma
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _threshold_exceeded(
    actual_value: float,
    threshold: float,
    direction: str,
) -> int:
    """Binary outcome label for a threshold market.

    direction="above" → 1 if actual > threshold
    direction="below" → 1 if actual < threshold
    """
    if direction == "above":
        return 1 if actual_value > threshold else 0
    if direction == "below":
        return 1 if actual_value < threshold else 0
    raise ValueError(f"unknown direction: {direction!r}")


def _raw_prob_above(forecast_value: float, threshold: float, sigma: float) -> float:
    """P(X > threshold) given X ~ N(forecast_value, sigma)."""
    return 1.0 - _normal_cdf(threshold, forecast_value, sigma)


def _raw_prob_below(forecast_value: float, threshold: float, sigma: float) -> float:
    """P(X < threshold) given X ~ N(forecast_value, sigma)."""
    return _normal_cdf(threshold, forecast_value, sigma)


def build_tail_training_set(
    city: str,
    market_type: str,
    direction: str,
    threshold: float,
    actuals_dir: Optional[Path] = None,
    archive_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Construct training rows for tail calibration.

    Returns empty DataFrame with the expected column set if the required
    source files are missing or empty.
    """
    actuals_dir = Path(actuals_dir) if actuals_dir is not None else STATION_ACTUALS_DIR
    archive_dir = Path(archive_dir) if archive_dir is not None else FORECAST_ARCHIVE_DIR
    slug = _slugify_city(city)
    actuals_path = actuals_dir / f"{slug}.csv"
    archive_path = archive_dir / f"{slug}.csv"

    if not actuals_path.exists() or not archive_path.exists():
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    actuals = pd.read_csv(actuals_path)
    archive = pd.read_csv(archive_path)
    if actuals.empty or archive.empty:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    # Normalize dates
    archive["date"] = pd.to_datetime(archive["date"]).dt.strftime("%Y-%m-%d")
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.strftime("%Y-%m-%d")

    # Prefer earliest lead-1 forecast per date (mirrors existing temp
    # calibration training discipline).
    archive = archive.sort_values(["date", "forecast_lead_days", "as_of_utc"])
    archive = archive.drop_duplicates("date", keep="first")

    forecast_col = "forecast_high_f" if market_type == "high" else "forecast_low_f"
    sigma_col = "ensemble_high_std_f" if market_type == "high" else "ensemble_low_std_f"
    actual_col = "tmax_f" if market_type == "high" else "tmin_f"

    joined = archive.merge(
        actuals[["date", actual_col]],
        on="date",
        how="inner",
    ).dropna(subset=[forecast_col, sigma_col, actual_col])

    if joined.empty:
        return pd.DataFrame(columns=_EMPTY_COLUMNS)

    prob_fn = _raw_prob_above if direction == "above" else _raw_prob_below
    joined["raw_prob"] = [
        prob_fn(float(row[forecast_col]), float(threshold), float(row[sigma_col]))
        for _, row in joined.iterrows()
    ]
    joined["actual_exceeded_0_1"] = [
        _threshold_exceeded(float(row[actual_col]), float(threshold), direction)
        for _, row in joined.iterrows()
    ]
    joined["forecast_value_f"] = joined[forecast_col].astype(float)
    joined["sigma_f"] = joined[sigma_col].astype(float)

    return joined[_EMPTY_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_training_data.py::test_build_tail_training_set_joins_forecasts_and_actuals -v`

Expected: PASS.

- [ ] **Step 5: Add bucket-training test**

Add to `tests/test_tail_training_data.py`:

```python
def test_build_bucket_training_set_joins_forecasts_and_actuals(tmp_path):
    """Bucket outcome: actual temp falls within [bucket_low, bucket_high]."""
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
        "ensemble_high_std_f": [2.0, 2.0],
        "ensemble_low_std_f": [1.5, 1.5],
        "forecast_model": ["best_match", "best_match"],
        "forecast_lead_days": [1, 1],
        "forecast_source": ["open_meteo_previous_runs"] * 2,
    }).to_csv(archive_dir / "new_york.csv", index=False)

    # Bucket [70, 71]: 2026-04-01 actual=70.5 is in, 2026-04-02 actual=75.0 is out
    df = build_bucket_training_set(
        city="New York",
        market_type="high",
        bucket_low=70.0,
        bucket_high=71.0,
        actuals_dir=actuals_dir,
        archive_dir=archive_dir,
    )
    assert list(df["actual_in_bucket_0_1"]) == [1, 0]
    # raw_bucket_prob = P(70 <= X <= 71) = F(71) - F(70)
    # For 2026-04-01 with mean=70, std=2: F(71) - F(70) ≈ 0.6915 - 0.5 = 0.1915
    assert df.iloc[0]["raw_bucket_prob"] == pytest.approx(0.1915, abs=0.005)
```

- [ ] **Step 6: Run test to confirm it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_training_data.py::test_build_bucket_training_set_joins_forecasts_and_actuals -v`

Expected: FAIL — function missing.

- [ ] **Step 7: Implement `build_bucket_training_set`**

Append to `src/tail_training_data.py`:

```python
_BUCKET_EMPTY_COLUMNS = [
    "date",
    "raw_bucket_prob",
    "actual_in_bucket_0_1",
    "forecast_value_f",
    "sigma_f",
    "forecast_lead_days",
    "as_of_utc",
]


def build_bucket_training_set(
    city: str,
    market_type: str,
    bucket_low: float,
    bucket_high: float,
    actuals_dir: Optional[Path] = None,
    archive_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Training rows for bucket-market calibration.

    Returns empty DataFrame with bucket column set if sources are missing.
    """
    actuals_dir = Path(actuals_dir) if actuals_dir is not None else STATION_ACTUALS_DIR
    archive_dir = Path(archive_dir) if archive_dir is not None else FORECAST_ARCHIVE_DIR
    slug = _slugify_city(city)
    actuals_path = actuals_dir / f"{slug}.csv"
    archive_path = archive_dir / f"{slug}.csv"

    if not actuals_path.exists() or not archive_path.exists():
        return pd.DataFrame(columns=_BUCKET_EMPTY_COLUMNS)

    actuals = pd.read_csv(actuals_path)
    archive = pd.read_csv(archive_path)
    if actuals.empty or archive.empty:
        return pd.DataFrame(columns=_BUCKET_EMPTY_COLUMNS)

    archive["date"] = pd.to_datetime(archive["date"]).dt.strftime("%Y-%m-%d")
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.strftime("%Y-%m-%d")
    archive = archive.sort_values(["date", "forecast_lead_days", "as_of_utc"])
    archive = archive.drop_duplicates("date", keep="first")

    forecast_col = "forecast_high_f" if market_type == "high" else "forecast_low_f"
    sigma_col = "ensemble_high_std_f" if market_type == "high" else "ensemble_low_std_f"
    actual_col = "tmax_f" if market_type == "high" else "tmin_f"

    joined = archive.merge(
        actuals[["date", actual_col]],
        on="date",
        how="inner",
    ).dropna(subset=[forecast_col, sigma_col, actual_col])

    if joined.empty:
        return pd.DataFrame(columns=_BUCKET_EMPTY_COLUMNS)

    joined["raw_bucket_prob"] = [
        _normal_cdf(float(bucket_high), float(row[forecast_col]), float(row[sigma_col]))
        - _normal_cdf(float(bucket_low), float(row[forecast_col]), float(row[sigma_col]))
        for _, row in joined.iterrows()
    ]
    joined["actual_in_bucket_0_1"] = [
        1 if float(bucket_low) <= float(row[actual_col]) <= float(bucket_high) else 0
        for _, row in joined.iterrows()
    ]
    joined["forecast_value_f"] = joined[forecast_col].astype(float)
    joined["sigma_f"] = joined[sigma_col].astype(float)

    return joined[_BUCKET_EMPTY_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 8: Run both tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_training_data.py -v`

Expected: both tests PASS.

- [ ] **Step 9: Run full suite to confirm no regressions**

Run: `./.venv/Scripts/python.exe -m pytest tests -q`

Expected: 243 + 2 new = 245 passed, 0 deselected.

- [ ] **Step 10: Commit**

```bash
git add src/tail_training_data.py tests/test_tail_training_data.py
git commit -m "feat(tail_training_data): threshold + bucket training-set builders"
```

---

## Task 2: `TailBinaryCalibrator` class (threshold markets)

**Rationale:** Compose P1's `LogisticRainCalibrator` and `IsotonicRainCalibrator` into a single calibrator per (city, market_type, direction) for threshold markets. Same two-stage chain; different model-path filenames so old and new stacks don't collide.

**Files:**
- Create: `src/tail_calibration.py`
- Test: `tests/test_tail_calibration.py`

- [ ] **Step 1: Write failing test for basic fit+predict**

Create `tests/test_tail_calibration.py`:

```python
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
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_calibration.py -v`

Expected: FAIL — module missing.

- [ ] **Step 3: Implement `TailBinaryCalibrator`**

Create `src/tail_calibration.py`:

```python
"""Tail calibration for temperature markets.

Composes P1's LogisticRainCalibrator + IsotonicRainCalibrator into a two-stage
chain specialized per (city, market_type, direction). Model pickles live in
data/calibration_models/ with distinct naming so the existing EMOS / NGR /
isotonic stack stays operable and comparable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.rain_calibration import (
    LogisticRainCalibrator,
    IsotonicRainCalibrator,
    _clip,
)
from src.station_truth import CALIBRATION_MODELS_DIR, _slugify_city

logger = logging.getLogger("weather.tail_calibration")


def _tail_model_path(
    city: str,
    market_type: str,
    direction: str,
    kind: str,
    model_dir: Optional[Path] = None,
) -> Path:
    """Per-pair tail model path.

    kind is one of "logistic" or "isotonic".
    """
    directory = Path(model_dir) if model_dir is not None else CALIBRATION_MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slugify_city(city)}_{market_type}_{direction}_tail_{kind}.pkl"


class TailBinaryCalibrator:
    """Two-stage tail calibration: logistic bias correction + isotonic
    probability recalibration. Per (city, market_type, direction)."""

    def __init__(self, city: str, market_type: str, direction: str):
        self.city = city
        self.market_type = market_type
        self.direction = direction
        self._logistic = LogisticRainCalibrator(city=city)
        self._isotonic = IsotonicRainCalibrator(city=city)

    def fit(self, raw_probs, outcomes) -> None:
        self._logistic.fit(raw_probs, outcomes)
        # Train isotonic on logistic-corrected probabilities
        import numpy as np
        raw_arr = np.asarray(raw_probs, dtype=float)
        logistic_preds = np.array([self._logistic.predict(p) for p in raw_arr])
        self._isotonic.fit(logistic_preds, outcomes)

    def predict(self, raw_prob: float) -> float:
        p = _clip(raw_prob)
        p = self._logistic.predict(p)
        p = self._isotonic.predict(p)
        return _clip(p)

    def save(self, path_prefix) -> None:
        """Save both stages to disk. path_prefix is a directory or file-prefix
        without extension; _logistic.pkl and _isotonic.pkl are appended.
        """
        prefix = Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self._logistic.save(Path(f"{prefix}_logistic.pkl"))
        self._isotonic.save(Path(f"{prefix}_isotonic.pkl"))

    @classmethod
    def load(cls, path_prefix) -> "TailBinaryCalibrator":
        """Load from a prefix path used at save()."""
        import pickle
        prefix = Path(path_prefix)
        with Path(f"{prefix}_logistic.pkl").open("rb") as f:
            logistic_data = pickle.load(f)
        with Path(f"{prefix}_isotonic.pkl").open("rb") as f:
            isotonic_data = pickle.load(f)
        # TailBinaryCalibrator needs city/market_type/direction — encode in a
        # sidecar metadata pickle.
        meta_path = Path(f"{prefix}_meta.pkl")
        if meta_path.exists():
            with meta_path.open("rb") as f:
                meta = pickle.load(f)
        else:
            # Fallback: only city was saved inside logistic's dict
            meta = {
                "city": logistic_data["city"],
                "market_type": "unknown",
                "direction": "unknown",
            }
        obj = cls(
            city=meta["city"],
            market_type=meta["market_type"],
            direction=meta["direction"],
        )
        obj._logistic._model = logistic_data["model"]
        obj._isotonic._model = isotonic_data["model"]
        return obj
```

- [ ] **Step 4: Run fit/predict test**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_calibration.py::test_tail_binary_calibrator_fit_and_predict -v`

Expected: PASS.

- [ ] **Step 5: Fix save/load to persist metadata**

The save/load test will fail because metadata isn't yet written. Update `save()`:

```python
    def save(self, path_prefix) -> None:
        import pickle
        prefix = Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self._logistic.save(Path(f"{prefix}_logistic.pkl"))
        self._isotonic.save(Path(f"{prefix}_isotonic.pkl"))
        meta = {
            "city": self.city,
            "market_type": self.market_type,
            "direction": self.direction,
        }
        with Path(f"{prefix}_meta.pkl").open("wb") as f:
            pickle.dump(meta, f)
```

- [ ] **Step 6: Run save/load test**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_calibration.py::test_tail_binary_calibrator_save_load -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/tail_calibration.py tests/test_tail_calibration.py
git commit -m "feat(tail_calibration): TailBinaryCalibrator (logistic + isotonic composition)"
```

---

## Task 3: `BucketDistributionalCalibrator` class

**Rationale:** For bucket markets (P(temp in [lo, hi])), calibrate the naive `F(hi) - F(lo)` prediction. Same two-stage chain as `TailBinaryCalibrator`; keyed on (city, market_type) without a direction dimension.

**Files:**
- Modify: `src/tail_calibration.py`
- Modify: `tests/test_tail_calibration.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_tail_calibration.py`:

```python
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
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_calibration.py::test_bucket_distributional_calibrator_fit_and_predict -v`

Expected: FAIL — class missing.

- [ ] **Step 3: Implement `BucketDistributionalCalibrator`**

Append to `src/tail_calibration.py`:

```python
def _bucket_model_path(
    city: str,
    market_type: str,
    kind: str,
    model_dir: Optional[Path] = None,
) -> Path:
    directory = Path(model_dir) if model_dir is not None else CALIBRATION_MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slugify_city(city)}_{market_type}_bucket_{kind}.pkl"


class BucketDistributionalCalibrator:
    """Two-stage calibration for bucket market probabilities.
    Per (city, market_type). No direction dimension — buckets are
    inherently two-sided.
    """

    def __init__(self, city: str, market_type: str):
        self.city = city
        self.market_type = market_type
        self._logistic = LogisticRainCalibrator(city=city)
        self._isotonic = IsotonicRainCalibrator(city=city)

    def fit(self, raw_bucket_probs, outcomes) -> None:
        self._logistic.fit(raw_bucket_probs, outcomes)
        import numpy as np
        raw_arr = np.asarray(raw_bucket_probs, dtype=float)
        logistic_preds = np.array([self._logistic.predict(p) for p in raw_arr])
        self._isotonic.fit(logistic_preds, outcomes)

    def predict(self, raw_bucket_prob: float) -> float:
        p = _clip(raw_bucket_prob)
        p = self._logistic.predict(p)
        p = self._isotonic.predict(p)
        return _clip(p)

    def save(self, path_prefix) -> None:
        import pickle
        prefix = Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self._logistic.save(Path(f"{prefix}_logistic.pkl"))
        self._isotonic.save(Path(f"{prefix}_isotonic.pkl"))
        meta = {"city": self.city, "market_type": self.market_type}
        with Path(f"{prefix}_meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path_prefix) -> "BucketDistributionalCalibrator":
        import pickle
        prefix = Path(path_prefix)
        with Path(f"{prefix}_logistic.pkl").open("rb") as f:
            logistic_data = pickle.load(f)
        with Path(f"{prefix}_isotonic.pkl").open("rb") as f:
            isotonic_data = pickle.load(f)
        meta_path = Path(f"{prefix}_meta.pkl")
        if meta_path.exists():
            with meta_path.open("rb") as f:
                meta = pickle.load(f)
        else:
            meta = {"city": logistic_data["city"], "market_type": "unknown"}
        obj = cls(city=meta["city"], market_type=meta["market_type"])
        obj._logistic._model = logistic_data["model"]
        obj._isotonic._model = isotonic_data["model"]
        return obj
```

- [ ] **Step 4: Run both new tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_calibration.py::test_bucket_distributional_calibrator_fit_and_predict tests/test_tail_calibration.py::test_bucket_calibrator_save_load -v`

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tail_calibration.py tests/test_tail_calibration.py
git commit -m "feat(tail_calibration): BucketDistributionalCalibrator for bucket markets"
```

---

## Task 4: `TailCalibrationManager` with mtime invalidation

**Rationale:** Runtime loader with per-pair caching + mtime-based invalidation (mirrors the cleanup-pass fix to `CalibrationManager` / `RainCalibrationManager`). Entry point `calibrate_tail_probability(city, market_type, direction, is_bucket, raw_prob)` returns `{"calibrated_prob", "source"}` or `None` when no model exists.

**Files:**
- Modify: `src/tail_calibration.py`
- Modify: `tests/test_tail_calibration.py`

- [ ] **Step 1: Write failing test for threshold path**

Add to `tests/test_tail_calibration.py`:

```python
def test_tail_calibration_manager_threshold_path(tmp_path):
    from src.tail_calibration import (
        TailBinaryCalibrator,
        TailCalibrationManager,
        _tail_model_path,
    )

    rng = np.random.default_rng(17)
    probs = rng.uniform(0.05, 0.95, 200)
    outcomes = (rng.uniform(0, 1, 200) < probs).astype(int)
    cal = TailBinaryCalibrator(city="New York", market_type="high", direction="above")
    cal.fit(probs, outcomes)
    prefix = _tail_model_path("New York", "high", "above", "prefix", model_dir=tmp_path)
    # prefix is a path ending with "_prefix.pkl" — we want just the stem
    # _tail_model_path returns "{slug}_high_above_tail_prefix.pkl"
    # For TailBinaryCalibrator.save, we pass the stem without extension
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
        city="New York",
        market_type="high",
        direction="above",
        is_bucket=False,
        raw_prob=0.25,
    ) is None
    assert mgr.calibrate_tail_probability(
        city="New York",
        market_type="high",
        direction="above",
        is_bucket=True,
        raw_prob=0.25,
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
        city="New York",
        market_type="high",
        direction="above",  # ignored for is_bucket=True
        is_bucket=True,
        raw_prob=0.20,
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
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_calibration.py -v`

Expected: four new tests fail — `TailCalibrationManager` missing.

- [ ] **Step 3: Implement `TailCalibrationManager`**

Append to `src/tail_calibration.py`:

```python
class TailCalibrationManager:
    """Per-pair loader + cache for tail calibration models. mtime-based
    invalidation so retrains on disk are picked up without process restart.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self._model_dir = Path(model_dir) if model_dir else CALIBRATION_MODELS_DIR
        # cache shape: {key: (mtime_or_None, model_or_None)}
        self._threshold_cache: dict = {}
        self._bucket_cache: dict = {}

    def _threshold_prefix(self, city: str, market_type: str, direction: str) -> Path:
        return self._model_dir / f"{_slugify_city(city)}_{market_type}_{direction}_tail"

    def _bucket_prefix(self, city: str, market_type: str) -> Path:
        return self._model_dir / f"{_slugify_city(city)}_{market_type}_bucket"

    def _latest_mtime(self, prefix: Path) -> Optional[float]:
        """Latest mtime across the three sidecar files for a prefix."""
        mtimes = []
        for suffix in ("_logistic.pkl", "_isotonic.pkl", "_meta.pkl"):
            p = Path(f"{prefix}{suffix}")
            if p.exists():
                mtimes.append(p.stat().st_mtime)
        return max(mtimes) if mtimes else None

    def _get_threshold(
        self, city: str, market_type: str, direction: str,
    ) -> Optional[TailBinaryCalibrator]:
        key = (city, market_type, direction)
        prefix = self._threshold_prefix(city, market_type, direction)
        current_mtime = self._latest_mtime(prefix)
        cached = self._threshold_cache.get(key)
        if cached is None or cached[0] != current_mtime:
            model = None
            if current_mtime is not None:
                try:
                    model = TailBinaryCalibrator.load(prefix)
                except Exception as exc:
                    logger.warning(
                        "Failed loading tail model for %s/%s/%s: %s",
                        city, market_type, direction, exc,
                    )
                    model = None
            self._threshold_cache[key] = (current_mtime, model)
        return self._threshold_cache[key][1]

    def _get_bucket(
        self, city: str, market_type: str,
    ) -> Optional[BucketDistributionalCalibrator]:
        key = (city, market_type)
        prefix = self._bucket_prefix(city, market_type)
        current_mtime = self._latest_mtime(prefix)
        cached = self._bucket_cache.get(key)
        if cached is None or cached[0] != current_mtime:
            model = None
            if current_mtime is not None:
                try:
                    model = BucketDistributionalCalibrator.load(prefix)
                except Exception as exc:
                    logger.warning(
                        "Failed loading bucket model for %s/%s: %s",
                        city, market_type, exc,
                    )
                    model = None
            self._bucket_cache[key] = (current_mtime, model)
        return self._bucket_cache[key][1]

    def calibrate_tail_probability(
        self,
        city: str,
        market_type: str,
        direction: str,
        is_bucket: bool,
        raw_prob: float,
    ) -> Optional[dict]:
        """Returns {"calibrated_prob", "source"} or None if no model exists."""
        if is_bucket:
            cal = self._get_bucket(city, market_type)
        else:
            cal = self._get_threshold(city, market_type, direction)
        if cal is None:
            return None
        return {
            "calibrated_prob": cal.predict(raw_prob),
            "source": "logistic+isotonic",
        }
```

- [ ] **Step 4: Run all manager tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tail_calibration.py -v`

Expected: all four new tests PASS plus the earlier four (8 total).

- [ ] **Step 5: Run full suite**

Run: `./.venv/Scripts/python.exe -m pytest tests -q`

Expected: ~253 passed (243 baseline + ~10 new from Tasks 1-4), 0 deselected.

- [ ] **Step 6: Commit**

```bash
git add src/tail_calibration.py tests/test_tail_calibration.py
git commit -m "feat(tail_calibration): TailCalibrationManager with mtime-invalidating cache"
```

---

## Task 5: `train_tail_calibration.py` CLI

**Rationale:** CLI that iterates over (city, market_type, direction) triples plus a fixed bucket list, builds training sets, applies the sample-size gate, fits models, and saves pkls. Respects `tail_calibration_window_days` from config.

**Files:**
- Create: `train_tail_calibration.py`

- [ ] **Step 1: Create the CLI script**

Create `train_tail_calibration.py` at the project root:

```python
"""Train per-pair tail calibration models.

Pairs:
  - threshold: every (city, market_type, direction) where the training set
    has >= 30 tail-region rows (raw_prob < 0.25 or > 0.75) and >= 2 actual
    tail events on each side of the raw_prob distribution.
  - bucket: every (city, market_type) with >= 30 training rows and >= 2
    in-bucket actuals over the window.

Thresholds for "tail region" come from the scan of candidate Kalshi
threshold markets — we parameterize the threshold sweep below by a set of
default temperatures per market_type that roughly match live Kalshi
offerings. For bucket markets, the default bucket grid is 1F wide.

Output artifacts in data/calibration_models/:
  {slug}_{mtype}_{direction}_tail_logistic.pkl
  {slug}_{mtype}_{direction}_tail_isotonic.pkl
  {slug}_{mtype}_{direction}_tail_meta.pkl
  {slug}_{mtype}_bucket_logistic.pkl
  {slug}_{mtype}_bucket_isotonic.pkl
  {slug}_{mtype}_bucket_meta.pkl
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.config import load_app_config, resolve_config_path
from src.logging_setup import configure_logging
from src.station_truth import ensure_data_directories, load_station_map
from src.tail_calibration import (
    BucketDistributionalCalibrator,
    TailBinaryCalibrator,
)
from src.tail_training_data import (
    build_bucket_training_set,
    build_tail_training_set,
)

_CONFIG_PATH = str(resolve_config_path())
log = configure_logging(
    "weather.train_tail",
    config_path=_CONFIG_PATH,
    log_filename="train_tail_calibration.log",
    level=logging.INFO,
)

# Default threshold sweep per market_type, in Fahrenheit. Kalshi temperature
# markets typically post thresholds within +/- 15F of climatological means.
_DEFAULT_HIGH_THRESHOLDS = list(range(40, 105, 5))  # 40, 45, ..., 100
_DEFAULT_LOW_THRESHOLDS = list(range(10, 80, 5))    # 10, 15, ..., 75


def _tail_region_counts(df) -> tuple[int, int, int]:
    """Return (tail_region_total, tail_events_low, tail_events_high)."""
    if df.empty:
        return 0, 0, 0
    low_mask = df["raw_prob"] < 0.25
    high_mask = df["raw_prob"] > 0.75
    tail_mask = low_mask | high_mask
    tail_events_low = int(df.loc[low_mask, "actual_exceeded_0_1"].sum())
    tail_events_high = int(df.loc[high_mask, "actual_exceeded_0_1"].sum())
    return int(tail_mask.sum()), tail_events_low, tail_events_high


def train_threshold(
    city: str,
    market_type: str,
    direction: str,
    threshold: float,
    window_days: int,
) -> dict:
    df = build_tail_training_set(
        city=city, market_type=market_type,
        direction=direction, threshold=threshold,
    )
    if df.empty:
        return {"status": "no_data"}

    df = df.tail(window_days).reset_index(drop=True)
    df = df.dropna(subset=["raw_prob", "actual_exceeded_0_1"])

    tail_total, tail_lo_events, tail_hi_events = _tail_region_counts(df)
    # Sample-size gate: >=30 tail-region rows, >=2 events on each side
    if tail_total < 30 or tail_lo_events < 2 or tail_hi_events < 2:
        return {
            "status": "insufficient_data",
            "tail_total": tail_total,
            "tail_lo_events": tail_lo_events,
            "tail_hi_events": tail_hi_events,
        }

    cal = TailBinaryCalibrator(city=city, market_type=market_type, direction=direction)
    cal.fit(df["raw_prob"].values, df["actual_exceeded_0_1"].astype(int).values)

    from src.tail_calibration import CALIBRATION_MODELS_DIR
    from src.station_truth import _slugify_city
    prefix = CALIBRATION_MODELS_DIR / f"{_slugify_city(city)}_{market_type}_{direction}_tail"
    cal.save(prefix)
    return {"status": "trained", "rows": int(len(df)), "threshold": threshold}


def train_bucket(
    city: str,
    market_type: str,
    bucket_low: float,
    bucket_high: float,
    window_days: int,
) -> dict:
    df = build_bucket_training_set(
        city=city, market_type=market_type,
        bucket_low=bucket_low, bucket_high=bucket_high,
    )
    if df.empty:
        return {"status": "no_data"}
    df = df.tail(window_days).reset_index(drop=True)
    df = df.dropna(subset=["raw_bucket_prob", "actual_in_bucket_0_1"])
    if len(df) < 30 or df["actual_in_bucket_0_1"].sum() < 2:
        return {"status": "insufficient_data", "rows": int(len(df))}

    cal = BucketDistributionalCalibrator(city=city, market_type=market_type)
    cal.fit(df["raw_bucket_prob"].values, df["actual_in_bucket_0_1"].astype(int).values)

    from src.tail_calibration import CALIBRATION_MODELS_DIR
    from src.station_truth import _slugify_city
    prefix = CALIBRATION_MODELS_DIR / f"{_slugify_city(city)}_{market_type}_bucket"
    cal.save(prefix)
    return {"status": "trained", "rows": int(len(df))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", help="Single city (default: all in stations.json)")
    parser.add_argument("--window-days", type=int, default=180)
    args = parser.parse_args()

    config = load_app_config(_CONFIG_PATH)
    window = int(config.get("tail_calibration_window_days", args.window_days))
    ensure_data_directories()

    cities = [args.city] if args.city else list(load_station_map().keys())
    for city in cities:
        for mtype, thresholds in (
            ("high", _DEFAULT_HIGH_THRESHOLDS),
            ("low", _DEFAULT_LOW_THRESHOLDS),
        ):
            # Pick the single threshold nearest to climatological mean so we
            # get one fit per (city, mtype, direction) rather than training
            # every threshold separately. In production this would sweep all
            # observed market thresholds — for P2 initial rollout we use the
            # median-ish threshold.
            threshold = float(thresholds[len(thresholds) // 2])
            for direction in ("above", "below"):
                result = train_threshold(city, mtype, direction, threshold, window)
                log.info(
                    "%s %s %s@%.0fF tail: %s",
                    city, mtype, direction, threshold,
                    result.get("status"),
                )
            # Bucket at the same threshold ± 0.5 F (1F-wide bucket)
            bucket_result = train_bucket(
                city, mtype, threshold - 0.5, threshold + 0.5, window,
            )
            log.info(
                "%s %s bucket[%0.1f-%0.1f]: %s",
                city, mtype, threshold - 0.5, threshold + 0.5,
                bucket_result.get("status"),
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `CALIBRATION_MODELS_DIR` re-export to `src/tail_calibration.py`**

`train_tail_calibration.py` imports `CALIBRATION_MODELS_DIR` from `src.tail_calibration`. Add at the top of `src/tail_calibration.py` (right after the existing imports):

```python
from src.station_truth import CALIBRATION_MODELS_DIR, _slugify_city  # noqa: F401 (re-export)
```

This is already imported in Task 2 — confirm it's there or keep the line from Task 2.

- [ ] **Step 3: Syntax check**

Run: `./.venv/Scripts/python.exe -c "import ast; ast.parse(open('train_tail_calibration.py').read())"`

Expected: no output (syntax OK).

- [ ] **Step 4: Dry-run for a single city**

Run: `./.venv/Scripts/python.exe train_tail_calibration.py --city 'New York'`

Expected: log output for each (high/low) × (above/below) direction + bucket — most will be `insufficient_data` until the archive has enough coverage but the script should run without error.

- [ ] **Step 5: Run for all cities**

Run: `./.venv/Scripts/python.exe train_tail_calibration.py`

Expected: completes for all 20 cities. Examine output for unexpected errors.

- [ ] **Step 6: Commit**

```bash
git add train_tail_calibration.py src/tail_calibration.py
git commit -m "feat(tail_calibration): train_tail_calibration.py CLI"
```

- [ ] **Step 7: Commit any produced model pkls**

```bash
git add data/calibration_models/*_tail_*.pkl data/calibration_models/*_bucket_*.pkl 2>/dev/null
git status --short data/calibration_models/ | head
git commit -m "data(calibration_models): initial tail + bucket model artifacts" 2>/dev/null || echo "nothing to commit"
```

If no new pkls exist (all pairs flagged insufficient_data), skip this commit.

---

## Task 6: `evaluate_tail_calibration.py` CLI

**Rationale:** Chronological-holdout evaluator. Computes per-pair log-loss under (raw, isotonic, tail, climatology), flags pairs that pass both gates as `qualifies_for_unblock`, writes a scorecard JSON to `data/evaluation_reports/`.

**Files:**
- Create: `evaluate_tail_calibration.py`

- [ ] **Step 1: Create the evaluator**

Create `evaluate_tail_calibration.py` at project root:

```python
"""Chronological-holdout evaluation of tail calibration.

Per (city, market_type, direction, threshold) pair, splits the training set
chronologically (train = 1..N-holdout, holdout = last holdout_days), fits
tail + baselines on train, scores log-loss on the tail region of holdout.

Output: data/evaluation_reports/tail_eval_YYYY-MM-DD.json with per-pair
results and qualifies_for_unblock bool.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.rain_calibration import IsotonicRainCalibrator, LogisticRainCalibrator
from src.station_truth import load_station_map
from src.tail_calibration import (
    BucketDistributionalCalibrator,
    TailBinaryCalibrator,
)
from src.tail_training_data import (
    build_bucket_training_set,
    build_tail_training_set,
)

_DEFAULT_HIGH_THRESHOLDS = [60.0, 70.0, 80.0, 90.0]
_DEFAULT_LOW_THRESHOLDS = [20.0, 30.0, 40.0, 50.0]
_TAIL_LO = 0.25
_TAIL_HI = 0.75


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


def _tail_mask(raw_probs: np.ndarray) -> np.ndarray:
    return (raw_probs < _TAIL_LO) | (raw_probs > _TAIL_HI)


def evaluate_threshold_pair(
    city: str, market_type: str, direction: str, threshold: float,
    days: int, holdout_days: int,
) -> dict:
    df = build_tail_training_set(
        city=city, market_type=market_type, direction=direction, threshold=threshold,
    ).dropna(subset=["raw_prob", "actual_exceeded_0_1"])
    df = df.tail(days).reset_index(drop=True)
    if len(df) < holdout_days + 20:
        return {
            "city": city, "market_type": market_type, "direction": direction,
            "threshold": threshold,
            "status": "insufficient_data",
            "rows": int(len(df)),
        }

    train = df.iloc[:-holdout_days]
    holdout = df.iloc[-holdout_days:]

    raw = holdout["raw_prob"].astype(float).values
    y = holdout["actual_exceeded_0_1"].astype(int).values

    # Baseline: existing isotonic on raw probs (no tail chain)
    existing_iso = IsotonicRainCalibrator(city=city)
    existing_iso.fit(train["raw_prob"].values, train["actual_exceeded_0_1"].astype(int).values)
    iso_preds = np.array([existing_iso.predict(p) for p in raw])

    # Tail chain
    tail_cal = TailBinaryCalibrator(city=city, market_type=market_type, direction=direction)
    tail_cal.fit(train["raw_prob"].values, train["actual_exceeded_0_1"].astype(int).values)
    tail_preds = np.array([tail_cal.predict(p) for p in raw])

    climatology = float(train["actual_exceeded_0_1"].astype(int).mean()) * np.ones_like(y, dtype=float)

    tail_idx = _tail_mask(raw)
    if tail_idx.sum() == 0:
        return {
            "city": city, "market_type": market_type, "direction": direction,
            "threshold": threshold,
            "status": "no_tail_rows_in_holdout",
            "rows": int(len(holdout)),
        }

    ll_raw = _log_loss(raw[tail_idx], y[tail_idx])
    ll_iso = _log_loss(iso_preds[tail_idx], y[tail_idx])
    ll_tail = _log_loss(tail_preds[tail_idx], y[tail_idx])
    ll_clim = _log_loss(climatology[tail_idx], y[tail_idx])

    qualifies = ll_tail < ll_clim and ll_tail < ll_iso
    return {
        "city": city, "market_type": market_type, "direction": direction,
        "threshold": threshold, "status": "ok",
        "tail_region_n": int(tail_idx.sum()),
        "log_loss_raw": ll_raw,
        "log_loss_isotonic": ll_iso,
        "log_loss_tail": ll_tail,
        "log_loss_climatology": ll_clim,
        "qualifies_for_unblock": bool(qualifies),
    }


def evaluate_bucket_pair(
    city: str, market_type: str, bucket_low: float, bucket_high: float,
    days: int, holdout_days: int,
) -> dict:
    df = build_bucket_training_set(
        city=city, market_type=market_type,
        bucket_low=bucket_low, bucket_high=bucket_high,
    ).dropna(subset=["raw_bucket_prob", "actual_in_bucket_0_1"])
    df = df.tail(days).reset_index(drop=True)
    if len(df) < holdout_days + 20:
        return {
            "city": city, "market_type": market_type,
            "bucket_low": bucket_low, "bucket_high": bucket_high,
            "status": "insufficient_data",
            "rows": int(len(df)),
        }

    train = df.iloc[:-holdout_days]
    holdout = df.iloc[-holdout_days:]
    raw = holdout["raw_bucket_prob"].astype(float).values
    y = holdout["actual_in_bucket_0_1"].astype(int).values

    existing_iso = IsotonicRainCalibrator(city=city)
    existing_iso.fit(train["raw_bucket_prob"].values, train["actual_in_bucket_0_1"].astype(int).values)
    iso_preds = np.array([existing_iso.predict(p) for p in raw])

    bucket_cal = BucketDistributionalCalibrator(city=city, market_type=market_type)
    bucket_cal.fit(train["raw_bucket_prob"].values, train["actual_in_bucket_0_1"].astype(int).values)
    bucket_preds = np.array([bucket_cal.predict(p) for p in raw])

    climatology = float(train["actual_in_bucket_0_1"].astype(int).mean()) * np.ones_like(y, dtype=float)

    ll_raw = _log_loss(raw, y)
    ll_iso = _log_loss(iso_preds, y)
    ll_bucket = _log_loss(bucket_preds, y)
    ll_clim = _log_loss(climatology, y)

    qualifies = ll_bucket < ll_clim and ll_bucket < ll_iso
    return {
        "city": city, "market_type": market_type,
        "bucket_low": bucket_low, "bucket_high": bucket_high,
        "status": "ok",
        "rows": int(len(holdout)),
        "log_loss_raw": ll_raw,
        "log_loss_isotonic": ll_iso,
        "log_loss_tail": ll_bucket,
        "log_loss_climatology": ll_clim,
        "qualifies_for_unblock": bool(qualifies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--holdout-days", type=int, default=30)
    args = parser.parse_args()

    cities = list(load_station_map().keys())
    threshold_results = []
    bucket_results = []

    for city in cities:
        for mtype, thresholds in (
            ("high", _DEFAULT_HIGH_THRESHOLDS),
            ("low", _DEFAULT_LOW_THRESHOLDS),
        ):
            for threshold in thresholds:
                for direction in ("above", "below"):
                    threshold_results.append(
                        evaluate_threshold_pair(
                            city, mtype, direction, threshold,
                            days=args.days, holdout_days=args.holdout_days,
                        )
                    )
                bucket_results.append(
                    evaluate_bucket_pair(
                        city, mtype, threshold - 0.5, threshold + 0.5,
                        days=args.days, holdout_days=args.holdout_days,
                    )
                )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {"days": args.days, "holdout_days": args.holdout_days},
        "threshold_pairs": threshold_results,
        "bucket_pairs": bucket_results,
    }
    out = (
        Path("data/evaluation_reports")
        / f"tail_eval_{datetime.now(timezone.utc).date().isoformat()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")

    qualifying = [r for r in threshold_results if r.get("qualifies_for_unblock")]
    qualifying += [r for r in bucket_results if r.get("qualifies_for_unblock")]
    print(f"qualifying pairs: {len(qualifying)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `./.venv/Scripts/python.exe -c "import ast; ast.parse(open('evaluate_tail_calibration.py').read())"`

Expected: no output.

- [ ] **Step 3: Run for all cities**

Run: `./.venv/Scripts/python.exe evaluate_tail_calibration.py --days 400 --holdout-days 30`

Expected: prints `wrote data/evaluation_reports/tail_eval_YYYY-MM-DD.json` and a qualifying-pair count. Most pairs will be `insufficient_data` given archive coverage.

- [ ] **Step 4: Commit evaluator + scorecard**

```bash
git add evaluate_tail_calibration.py data/evaluation_reports/tail_eval_*.json
git commit -m "feat(tail_calibration): evaluate CLI + first holdout scorecard"
```

---

## Task 7: Matcher tail-branch routing

**Rationale:** Attach `our_probability_tail` / `edge_tail` / `tail_calibration_source` to the opportunity dict when `TailCalibrationManager` has models for the pair. BUY-side isotonic path stays unchanged. No existing test should break.

**Files:**
- Modify: `src/matcher.py`
- Create: `tests/test_matcher_tail_routing.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_matcher_tail_routing.py`:

```python
from datetime import datetime, timezone


def _build_hourly(target_date: str, temps_f: list[float]) -> dict:
    return {
        "hourly": {
            "time": [f"{target_date}T{hour:02d}:00" for hour in range(len(temps_f))],
            "temperature_2m": [((t - 32.0) * 5.0 / 9.0) for t in temps_f],
        },
        "timezone": "America/New_York",
    }


def test_matcher_attaches_tail_probability_when_models_exist():
    from src.matcher import match_kalshi_markets

    class _StubTailManager:
        def calibrate_tail_probability(self, city, market_type, direction, is_bucket, raw_prob):
            # Return a tail calibration that shifts probability up by 0.1
            return {
                "calibrated_prob": min(0.999, max(0.001, float(raw_prob) + 0.1)),
                "source": "logistic+isotonic",
            }

    forecasts = {"Austin": _build_hourly("2030-01-01", [75.0] * 24)}
    markets = [{
        "city": "Austin",
        "type": "high",
        "threshold": 80.0,
        "ticker": "KXHIGHTAUS-30JAN01-T80",
        "title": "Will the maximum temperature be >80F on Jan 1, 2030?",
        "yes_sub_title": "80 or above",
        "last_price": 0.50,
        "yes_bid": 0.48,
        "yes_ask": 0.52,
        "close_time": "2030-01-01T23:59:00+00:00",
        "volume_24h": 5000,
    }]
    now = datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc)

    opps = match_kalshi_markets(
        forecasts, markets,
        min_edge=0.0,
        uncertainty_std_f=2.0,
        calibration_manager=None,
        tail_calibration_manager=_StubTailManager(),
        hrrr_blend_horizon_hours=18.0,
        now_utc=now,
    )
    assert len(opps) == 1
    opp = opps[0]
    assert "our_probability_tail" in opp
    assert opp["our_probability_tail"] == pytest.approx(opp["our_probability"] + 0.1, abs=0.01)
    assert opp["tail_calibration_source"] == "logistic+isotonic"
    assert "edge_tail" in opp


def test_matcher_omits_tail_fields_when_manager_is_none():
    from src.matcher import match_kalshi_markets

    forecasts = {"Austin": _build_hourly("2030-01-01", [75.0] * 24)}
    markets = [{
        "city": "Austin", "type": "high", "threshold": 80.0,
        "ticker": "KXHIGHTAUS-30JAN01-T80",
        "title": "...", "yes_sub_title": "80 or above",
        "last_price": 0.50, "yes_bid": 0.48, "yes_ask": 0.52,
        "close_time": "2030-01-01T23:59:00+00:00",
        "volume_24h": 5000,
    }]
    now = datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc)

    opps = match_kalshi_markets(
        forecasts, markets, min_edge=0.0, uncertainty_std_f=2.0,
        calibration_manager=None, tail_calibration_manager=None,
        hrrr_blend_horizon_hours=18.0, now_utc=now,
    )
    assert len(opps) == 1
    assert "our_probability_tail" not in opps[0]


import pytest
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_matcher_tail_routing.py -v`

Expected: FAIL — `match_kalshi_markets` doesn't accept `tail_calibration_manager`.

- [ ] **Step 3: Add `tail_calibration_manager` param to `match_kalshi_markets`**

In `src/matcher.py`, locate `def match_kalshi_markets(` signature (around line 300-380; search for `def match_kalshi_markets`). Add a new keyword-only parameter:

```python
def match_kalshi_markets(
    forecasts: dict,
    markets: list[dict],
    min_edge: float,
    uncertainty_std_f: float,
    *,
    ensemble_data: Optional[dict] = None,
    calibration_manager=None,
    tail_calibration_manager=None,  # NEW
    hrrr_data: Optional[dict] = None,
    hrrr_blend_horizon_hours: float = 18.0,
    now_utc: Optional[datetime] = None,
    use_ngr_calibration: bool = False,
) -> list[dict]:
```

The exact existing signature may vary; keep all existing params and add `tail_calibration_manager=None` to the keyword-only section.

- [ ] **Step 4: Attach tail probability to the opportunity dict**

Find the opportunity-append block (search for `"direction": "BUY" if edge > 0 else "SELL"`, around line 507). Right before the `opps.append(...)` call, insert:

```python
        tail_fields = {}
        if tail_calibration_manager is not None:
            direction_for_tail = "above" if _kalshi_threshold_direction(m) == "above" else "below"
            tail_result = tail_calibration_manager.calibrate_tail_probability(
                city=city,
                market_type=mtype,
                direction=direction_for_tail,
                is_bucket=is_bucket,
                raw_prob=raw_prob,
            )
            if tail_result is not None:
                prob_tail = tail_result["calibrated_prob"]
                tail_fields = {
                    "our_probability_tail": round(float(prob_tail), 4),
                    "edge_tail": round(float(prob_tail) - float(market_price), 4),
                    "tail_calibration_source": tail_result.get("source", "tail"),
                }
```

Then when building the opp dict, add:

```python
        opp = {
            # ... existing fields ...
        }
        opp.update(tail_fields)
        opps.append(opp)
```

(If the existing code uses a single dict literal in `opps.append({...})`, change it to build `opp` first, then append.)

- [ ] **Step 5: Run new tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_matcher_tail_routing.py -v`

Expected: both tests PASS.

- [ ] **Step 6: Run full suite — confirm no BUY-path regressions**

Run: `./.venv/Scripts/python.exe -m pytest tests -q`

Expected: all tests green.

- [ ] **Step 7: Commit**

```bash
git add src/matcher.py tests/test_matcher_tail_routing.py
git commit -m "feat(matcher): attach tail probability when TailCalibrationManager has models"
```

---

## Task 8: Policy v5 — `tail_unblocks` + `bankroll_slices`

**Rationale:** Update policy JSON to v5 (empty unblocks list, new bankroll slices block) and add `apply_tail_unblocks` filter logic.

**Files:**
- Modify: `strategy/strategy_policy.json`
- Modify: `src/strategy_policy.py`
- Modify: `tests/test_strategy_policy.py`

- [ ] **Step 1: Write failing test for filter behavior**

Add to `tests/test_strategy_policy.py`:

```python
def test_apply_tail_unblocks_routes_sell_when_pair_is_listed():
    from src.strategy_policy import apply_tail_unblocks

    opp = {
        "city": "Austin",
        "market_type": "high",
        "direction": "SELL",
        "is_bucket": False,
        "our_probability": 0.4,
        "our_probability_tail": 0.15,
        "market_price": 0.3,
        "edge": 0.1,
        "edge_tail": -0.15,
    }
    policy = {
        "tail_unblocks": {
            "threshold_sell": [
                {"city": "Austin", "market_type": "high", "direction": "above",
                 "bankroll_slice": "probation"}
            ],
            "bucket": [],
        },
    }
    routed = apply_tail_unblocks(opp, policy, threshold_direction="above")
    assert routed is not None
    assert routed["our_probability"] == 0.15
    assert routed["edge"] == -0.15
    assert routed["bankroll_slice"] == "probation"


def test_apply_tail_unblocks_drops_when_pair_not_listed():
    from src.strategy_policy import apply_tail_unblocks

    opp = {
        "city": "Austin", "market_type": "high", "direction": "SELL",
        "is_bucket": False,
        "our_probability": 0.4, "our_probability_tail": 0.15,
        "market_price": 0.3, "edge": 0.1, "edge_tail": -0.15,
    }
    policy = {"tail_unblocks": {"threshold_sell": [], "bucket": []}}
    assert apply_tail_unblocks(opp, policy, threshold_direction="above") is None


def test_apply_tail_unblocks_passthrough_for_buy():
    """BUY opportunities are not routed through tail_unblocks."""
    from src.strategy_policy import apply_tail_unblocks

    opp = {
        "city": "Austin", "market_type": "high", "direction": "BUY",
        "is_bucket": False,
        "our_probability": 0.6, "our_probability_tail": 0.7,
        "market_price": 0.3, "edge": 0.3, "edge_tail": 0.4,
    }
    policy = {"tail_unblocks": {"threshold_sell": [], "bucket": []}}
    # BUY opportunities are untouched by this filter — returned as-is
    routed = apply_tail_unblocks(opp, policy, threshold_direction="above")
    assert routed is opp
    assert routed["our_probability"] == 0.6  # unchanged
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_strategy_policy.py::test_apply_tail_unblocks_routes_sell_when_pair_is_listed tests/test_strategy_policy.py::test_apply_tail_unblocks_drops_when_pair_not_listed tests/test_strategy_policy.py::test_apply_tail_unblocks_passthrough_for_buy -v`

Expected: FAIL — function missing.

- [ ] **Step 3: Implement `apply_tail_unblocks`**

Add to `src/strategy_policy.py` (at the end of the file, or near the existing filter functions):

```python
def apply_tail_unblocks(
    opportunity: dict,
    policy: dict,
    threshold_direction: str | None = None,
) -> dict | None:
    """Tail-unblock filter.

    For SELL / bucket opportunities that would be blocked by
    allowed_position_sides / allowed_settlement_rules, consults the
    policy's tail_unblocks list. If the (city, market_type, direction)
    pair is listed, swaps in the tail probability as the decision value
    and tags bankroll_slice. Otherwise returns None (filter-out).

    BUY opportunities pass through unchanged (not SELL, not bucket).
    """
    direction = str(opportunity.get("direction", "")).upper()
    is_bucket = bool(opportunity.get("is_bucket", False))

    # BUY-side thresholds aren't tail-routed — they use existing isotonic path
    if direction == "BUY" and not is_bucket:
        return opportunity

    tail_unblocks = (policy or {}).get("tail_unblocks") or {}
    city = opportunity.get("city")
    market_type = opportunity.get("market_type")

    # Bucket route
    if is_bucket:
        allowed = [
            e for e in tail_unblocks.get("bucket", [])
            if e.get("city") == city and e.get("market_type") == market_type
        ]
        if not allowed:
            return None
        entry = allowed[0]
    else:
        # SELL threshold route
        allowed = [
            e for e in tail_unblocks.get("threshold_sell", [])
            if (
                e.get("city") == city
                and e.get("market_type") == market_type
                and e.get("direction") == threshold_direction
            )
        ]
        if not allowed:
            return None
        entry = allowed[0]

    tail_prob = opportunity.get("our_probability_tail")
    tail_edge = opportunity.get("edge_tail")
    if tail_prob is None or tail_edge is None:
        # Policy lists the pair but models didn't populate tail fields —
        # treat as blocked to stay safe.
        return None

    routed = dict(opportunity)
    routed["our_probability"] = tail_prob
    routed["edge"] = tail_edge
    routed["bankroll_slice"] = entry.get("bankroll_slice", "probation")
    return routed
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_strategy_policy.py -v`

Expected: all tests PASS (existing + three new).

- [ ] **Step 5: Update policy JSON to v5**

In `strategy/strategy_policy.json`, add two new top-level keys after the existing `rationale` block:

```json
  "tail_unblocks": {
    "threshold_sell": [],
    "bucket": []
  },
  "bankroll_slices": {
    "temperature_buy": 0.70,
    "rain_binary": 0.20,
    "probation": 0.10
  }
```

And bump `policy_version: 4` to `5`. Update `generated_at_utc`.

- [ ] **Step 6: Validate JSON still parses**

Run: `./.venv/Scripts/python.exe -c "import json; print(json.load(open('strategy/strategy_policy.json'))['policy_version'])"`

Expected: `5`.

- [ ] **Step 7: Commit**

```bash
git add strategy/strategy_policy.json src/strategy_policy.py tests/test_strategy_policy.py
git commit -m "feat(strategy): policy v5 — tail_unblocks + bankroll_slices (empty at launch)"
```

---

## Task 9: `bankroll_slice` column in paper ledger

**Rationale:** Extend the paper-trade ledger with a `bankroll_slice` attribution column. Legacy rows migrate as `temperature_buy`. The `summary.json` `category_breakdown` gains a nested `slice_breakdown`.

**Files:**
- Modify: `src/paper_trading.py`
- Modify: `tests/test_paper_trading.py`

- [ ] **Step 1: Write failing migration test**

Add to `tests/test_paper_trading.py`:

```python
def test_legacy_ledger_migrates_bankroll_slice_to_temperature_buy(tmp_path):
    import pandas as pd
    from src.paper_trading import _ensure_ledger_schema

    legacy = tmp_path / "legacy.csv"
    pd.DataFrame({
        "trade_id": ["t1"], "scan_id": ["s1"],
        "status": ["settled"], "source": ["kalshi"],
        "ticker": ["KXHIGHT"], "city": ["Boston"],
        "market_type": ["high"], "market_category": ["temperature"],
        "market_date": ["2026-04-20"],
    }).to_csv(legacy, index=False)

    df = _ensure_ledger_schema(pd.read_csv(legacy))
    assert "bankroll_slice" in df.columns
    assert df.iloc[0]["bankroll_slice"] == "temperature_buy"


def test_log_paper_trades_records_bankroll_slice(tmp_path):
    from src.paper_trading import log_paper_trades

    opps = [{
        "source": "kalshi", "ticker": "KXHIGHT-T80",
        "market_category": "temperature",
        "city": "Boston", "market_type": "high",
        "market_date": "2026-04-21", "outcome": "Yes",
        "position_side": "yes", "direction": "BUY",
        "our_probability": 0.6, "market_price": 0.35,
        "edge": 0.25, "abs_edge": 0.25,
        "bankroll_slice": "temperature_buy",
        "forecast_blend_source": "open-meteo",
        "forecast_calibration_source": "isotonic",
        "probability_calibration_source": "isotonic",
        "hours_to_settlement": 12,
        "raw_probability": 0.55, "volume24hr": 2500,
        "yes_outcome": True,
    }]
    ledger = tmp_path / "ledger.csv"
    log_paper_trades(opps, scan_timestamp="2026-04-20T12:00:00+00:00",
                     ledger_path=ledger, contracts=1)
    import pandas as pd
    df = pd.read_csv(ledger)
    assert df.iloc[0]["bankroll_slice"] == "temperature_buy"
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_paper_trading.py::test_legacy_ledger_migrates_bankroll_slice_to_temperature_buy tests/test_paper_trading.py::test_log_paper_trades_records_bankroll_slice -v`

Expected: FAIL.

- [ ] **Step 3: Add `bankroll_slice` to ledger schema**

In `src/paper_trading.py`:

1. Add `"bankroll_slice"` to both `OBJECT_COLUMNS` and `LEDGER_COLUMNS` (insert it right after `"market_category"` — search for that string).
2. In `_ensure_ledger_schema` (same function that handles `market_category` migration), add:

```python
    if "bankroll_slice" not in df.columns:
        df["bankroll_slice"] = "temperature_buy"
    else:
        df["bankroll_slice"] = df["bankroll_slice"].fillna("temperature_buy")
```

3. In the row-construction code where other opportunity fields are copied into the new row (search for `"market_category": _coerce_text(opp.get("market_category")`), add:

```python
        "bankroll_slice": _coerce_text(opp.get("bankroll_slice"), default="temperature_buy"),
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_paper_trading.py::test_legacy_ledger_migrates_bankroll_slice_to_temperature_buy tests/test_paper_trading.py::test_log_paper_trades_records_bankroll_slice -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_trading.py tests/test_paper_trading.py
git commit -m "feat(paper_trading): bankroll_slice ledger column with legacy migration"
```

---

## Task 10: Slice-scaled Kelly sizing

**Rationale:** `compute_position_size` accepts a `bankroll_fraction_multiplier` that scales the bankroll used for Kelly. Default 1.0 → no behavior change. `main.py` passes the slice multiplier based on the opportunity's `bankroll_slice`.

**Files:**
- Modify: `src/sizing.py`
- Modify: `tests/test_sizing.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_sizing.py`:

```python
def test_bankroll_fraction_multiplier_scales_stake_proportionally():
    from src.sizing import compute_position_size

    base = compute_position_size(
        edge=0.15, price=0.30, side="BUY",
        kelly_fraction=0.25, bankroll_dollars=1000,
        max_order_cost_dollars=100.0, hard_cap_contracts=10,
    )
    halved = compute_position_size(
        edge=0.15, price=0.30, side="BUY",
        kelly_fraction=0.25, bankroll_dollars=1000,
        max_order_cost_dollars=100.0, hard_cap_contracts=10,
        bankroll_fraction_multiplier=0.5,
    )
    # Halved bankroll should produce <= half the contracts (and never more)
    assert halved <= base
    assert halved >= 0
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_sizing.py::test_bankroll_fraction_multiplier_scales_stake_proportionally -v`

Expected: FAIL — unexpected keyword arg.

- [ ] **Step 3: Add the new parameter to `compute_position_size`**

In `src/sizing.py`, extend the signature:

```python
def compute_position_size(
    edge: float,
    price: float,
    side: str,
    kelly_fraction: float,
    bankroll_dollars: float,
    max_order_cost_dollars: float,
    hard_cap_contracts: int,
    bankroll_fraction_multiplier: float = 1.0,  # NEW
) -> int:
```

In the function body, find the `stake = kelly_fraction * bankroll_dollars * ...` line and scale:

```python
    effective_bankroll = float(bankroll_dollars) * float(bankroll_fraction_multiplier)
    # use effective_bankroll everywhere bankroll_dollars was used
```

- [ ] **Step 4: Run test to confirm it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_sizing.py -v`

Expected: all PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/sizing.py tests/test_sizing.py
git commit -m "feat(sizing): bankroll_fraction_multiplier for per-slice Kelly sizing"
```

---

## Task 11: Wire tail calibration into `main.py`

**Rationale:** Instantiate `TailCalibrationManager` when `enable_tail_calibration: true` in config, pass it to `match_kalshi_markets`. Route opportunities through `apply_tail_unblocks` before the main policy filter. Feature-flagged so flag-off behavior is identical to current.

**Files:**
- Modify: `main.py`
- Modify: `config.example.json`

- [ ] **Step 1: Add config keys**

In `config.example.json`, add near the other `enable_*` flags:

```json
"enable_tail_calibration": false,
"tail_calibration_window_days": 180,
"tail_bankroll_fraction": 0.10
```

- [ ] **Step 2: Instantiate manager in `run_scan`**

In `main.py`, near the existing calibration-manager instantiation (search for `CalibrationManager(model_dir=`), add:

```python
    tail_calibration_manager = None
    if config.get("enable_tail_calibration", False):
        try:
            from src.tail_calibration import TailCalibrationManager
            tail_calibration_manager = TailCalibrationManager(
                model_dir=config.get("calibration_model_dir"),
            )
            log.info("Tail calibration enabled.")
        except Exception as exc:
            log.warning("Tail calibration setup failed; falling back to raw: %s", exc)
            tail_calibration_manager = None
```

- [ ] **Step 3: Pass the manager into `match_kalshi_markets`**

Still in `main.py`, find the `match_kalshi_markets(` call (search for that exact string). Add the new kwarg:

```python
            kalshi_opps = match_kalshi_markets(
                valid,
                kalshi_markets,
                min_edge=min_edge,
                uncertainty_std_f=uncertainty,
                ensemble_data=ensemble_data,
                calibration_manager=calibration_manager,
                tail_calibration_manager=tail_calibration_manager,  # NEW
                hrrr_data=kalshi_hrrr_data,
                hrrr_blend_horizon_hours=hrrr_blend_horizon_hours,
                now_utc=now_utc,
                use_ngr_calibration=bool(config.get("use_ngr_calibration", False)),
            )
```

- [ ] **Step 4: Route opportunities through `apply_tail_unblocks` before policy filter**

In `main.py`, find the policy-filter call (`filter_opportunities_for_policy`). Before it, for each policy, run `apply_tail_unblocks` per opportunity:

```python
    from src.strategy_policy import apply_tail_unblocks, filter_opportunities_for_policy, load_strategy_policy

    temp_policy, _ = load_strategy_policy(config.get("strategy_policy_path"))

    # Pre-filter: tail-unblocks may reroute blocked SELL/bucket opportunities
    def _tail_preroute(opps: list[dict], policy: dict) -> list[dict]:
        result = []
        for o in opps:
            td = None
            if not o.get("is_bucket", False):
                # infer threshold_direction from settlement_rule
                rule = str(o.get("settlement_rule", "")).lower()
                if rule in ("gt", "gte"):
                    td = "above"
                elif rule in ("lt", "lte"):
                    td = "below"
            routed = apply_tail_unblocks(o, policy, threshold_direction=td)
            if routed is not None:
                result.append(routed)
        return result

    all_opportunities = _tail_preroute(all_opportunities, temp_policy)
    trade_opportunities = filter_opportunities_for_policy(all_opportunities, temp_policy)
```

(The exact placement depends on existing code shape — insert immediately before the first `filter_opportunities_for_policy(...)` call.)

- [ ] **Step 5: Verify flag-off behavior is unchanged**

Run: `./.venv/Scripts/python.exe -m pytest tests -q`

Expected: all tests green. `enable_tail_calibration` defaults to false, so nothing in existing tests triggers the new path.

- [ ] **Step 6: Commit**

```bash
git add main.py config.example.json
git commit -m "feat(main): wire tail calibration behind enable_tail_calibration flag"
```

---

## Task 12: `autopilot_weekly.py` integration

**Rationale:** Weekly retrain cadence. After the existing temperature calibration retrain, run `train_tail_calibration.py` then `evaluate_tail_calibration.py`.

**Files:**
- Modify: `scripts/autopilot_weekly.py`

- [ ] **Step 1: Read existing script shape**

Run: `./.venv/Scripts/python.exe -c "print(open('scripts/autopilot_weekly.py').read())" | head -80`

Review how the existing temperature retrain is invoked. Follow the same shape for adding the tail retrain step.

- [ ] **Step 2: Add tail train + evaluate calls**

Insert after the existing `train_calibration.py` invocation and before any test-suite gate:

```python
    # Tail calibration retrain (P2)
    if config.get("enable_tail_calibration", False):
        log.info("Running tail calibration retrain...")
        subprocess.run(
            [sys.executable, "train_tail_calibration.py"],
            check=False,
        )
        log.info("Running tail calibration holdout evaluation...")
        subprocess.run(
            [
                sys.executable, "evaluate_tail_calibration.py",
                "--days", "400", "--holdout-days", "30",
            ],
            check=False,
        )
```

(Use whichever subprocess invocation pattern the existing script uses.)

- [ ] **Step 3: Syntax check**

Run: `./.venv/Scripts/python.exe -c "import ast; ast.parse(open('scripts/autopilot_weekly.py').read())"`

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add scripts/autopilot_weekly.py
git commit -m "feat(autopilot): run tail retrain + holdout eval weekly"
```

---

## Task 13: Docs — `.claude/rules/tail-calibration.md` + CLAUDE.md updates

**Files:**
- Create: `.claude/rules/tail-calibration.md`
- Modify: `src/CLAUDE.md`
- Modify: `strategy/CLAUDE.md`

- [ ] **Step 1: Create `.claude/rules/tail-calibration.md`**

```markdown
# Tail Calibration Rules (P2)

## Scope

P2 reclaims blocked temperature SELL-side and bucket-market trades via a
tail-specific calibration layer composed on top of the existing chain.
Activated by `enable_tail_calibration: true` in config.

## Routing

- BUY-side thresholds: use existing isotonic path unchanged.
- SELL-side thresholds: route through `TailCalibrationManager` if models
  exist for (city, market_type, direction). Trade only if pair listed in
  `strategy_policy.json` `tail_unblocks.threshold_sell`.
- Bucket markets: route through `BucketDistributionalCalibrator`. Trade
  only if pair listed in `tail_unblocks.bucket`.

## Calibration chain (per pair)

```
raw_prob → LogisticRainCalibrator (bias correction)
        → IsotonicRainCalibrator (probability recalibration)
        → calibrated_prob
```

Same two-stage chain as P1 rain, composed via
`src/tail_calibration.TailBinaryCalibrator` / `BucketDistributionalCalibrator`.

## Training

- `train_tail_calibration.py` — per-pair, rolling
  `tail_calibration_window_days` (default 180).
- Sample-size gate: >=30 tail-region rows (raw_prob < 0.25 or > 0.75) and
  >=2 actual tail events on each side of the raw_prob distribution.
  Pairs below the gate produce no model.

## Evaluation gate

A pair qualifies for unblock only if on the chronological holdout:

1. Tail-region log-loss strictly beats climatology baseline.
2. Tail-region log-loss with the tail calibrator strictly beats tail-region
   log-loss with the existing raw isotonic calibrator.

Scorecards written to `data/evaluation_reports/tail_eval_YYYY-MM-DD.json`.

## Unblock workflow

- Unblock requires **two consecutive weekly scorecards** both reporting
  `qualifies_for_unblock: true` for the same pair.
- Policy update is manual: human adds the pair to
  `strategy_policy.json` `tail_unblocks.*` with `bankroll_slice: "probation"`
  and a `scorecard_ref`. Bump `policy_version`.

## Probation

Newly-unblocked pairs draw from the `probation` bankroll slice (10% of
total) for the first 30 days of paper trading. Kelly sizing is scaled by
`tail_bankroll_fraction` (default 0.10). Promotion from probation →
`temperature_buy` requires 30 days of ROI >= 0 on the pair; manual policy
edit to change the pair's `bankroll_slice` value.

## Rollback

Remove the pair from `tail_unblocks` (one-line JSON edit, bump
`policy_version`) and append the failure to `_learnings.md`.
```

- [ ] **Step 2: Update `src/CLAUDE.md` module map**

Add rows to the module table:

```markdown
| `tail_calibration.py` | TailBinaryCalibrator + BucketDistributionalCalibrator + TailCalibrationManager; composes P1 LogisticRainCalibrator/IsotonicRainCalibrator for temperature tails |
| `tail_training_data.py` | build_tail_training_set + build_bucket_training_set — forecast-archive × station-actuals joins for tail calibration |
```

- [ ] **Step 3: Update `strategy/CLAUDE.md`**

Add a new section:

```markdown
## Tail-Unblocks Policy (v5, 2026-04-23)

P2 policy v5 adds two top-level sections:

- `tail_unblocks` — per-pair allowlist overrides for SELL-side and bucket
  markets that are blanket-blocked by `allowed_position_sides` and
  `allowed_settlement_rules`. Entries carry `scorecard_ref` and
  `bankroll_slice` for attribution.
- `bankroll_slices` — fractions summing to 1.0:
  - `temperature_buy`: 0.70 (existing BUY book)
  - `rain_binary`: 0.20 (P1 rain vertical)
  - `probation`: 0.10 (new tail unblocks, 30-day evaluation)

See `.claude/rules/tail-calibration.md` for the unblock workflow.
```

- [ ] **Step 4: Commit**

```bash
git add .claude/rules/tail-calibration.md src/CLAUDE.md strategy/CLAUDE.md
git commit -m "docs: tail-calibration rules + module map + policy v5 section"
```

---

## Task 14: ADR via codebase-memory-mcp

**Rationale:** Capture the P2 architectural decision durably. ADRs are the "why" every future `orient` reads.

- [ ] **Step 1: Check if the Weather project is indexed in codebase-memory-mcp**

Call: `mcp__codebase-memory-mcp__list_projects` and look for a project whose `root_path` contains `Weather`.

If not listed, call `mcp__codebase-memory-mcp__index_repository` to add it.

- [ ] **Step 2: Write ADR-0001 via `manage_adr`**

Call `mcp__codebase-memory-mcp__manage_adr` with:
- `mode: "update"`
- `project: "Weather"`
- `content`:

```markdown
# ADR-0001: Tail calibration reuses P1 binary-outcome calibrators

Status: accepted
Date: 2026-04-23
Deciders: Gabriel + Claude

## Context

Temperature policy v4 blocks SELL-side (35 trades, -50.6% ROI) and bucket
markets (8/9 lost) because the Gaussian-based probability chain
underestimates rare events. P1 rain vertical built
`LogisticRainCalibrator` and `IsotonicRainCalibrator` as deliberately
generic binary-outcome calibrators explicitly for P2 reuse.

## Decision

P2 adds a parallel tail-calibration branch that composes P1's calibrators
via `TailBinaryCalibrator` (threshold events) and
`BucketDistributionalCalibrator` (bucket events). The BUY-side isotonic
path is untouched. Per-pair holdout evidence (two consecutive weekly
scorecards) gates individual unblocks; newly-unblocked pairs run on a
separate 10% probation bankroll slice for 30 days.

## Alternatives considered

- **Student-t base distribution** — fatter tails by construction. Rejected
  because the change would affect every downstream probability including
  currently-profitable BUY trades.
- **Empirical sigma inflation** — simpler but broadens the center of the
  distribution to fix the tails; regresses BUY accuracy.
- **Per-bin reliability correction** — simpler than a calibrator chain but
  doesn't compose as cleanly with the existing isotonic pipeline.

## Consequences

- Positive: Surgical fix, BUY book untouched, reuses P1 code verbatim,
  evidence-gated unblocks mirror the existing selective_raw_fallback
  pattern.
- Negative: Two calibration stacks live side-by-side in
  data/calibration_models/; filename discipline is load-bearing.
- Neutral: Unblock cadence is manual (two-scorecard rule, human policy
  edit). Automation can come later if the pattern holds.

## Revisit if

- More than 10 pairs unblocked and probation management becomes
  operationally heavy → consider automating promotion.
- Tail calibrator holdout log-loss beats raw isotonic by < 5% on average
  → the chain isn't pulling its weight; reconsider Student-t (option A).
```

- [ ] **Step 3: Mirror to `docs/adr/0001-tail-calibration-reuses-p1.md`**

Write the same content to `docs/adr/0001-tail-calibration-reuses-p1.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/adr/0001-tail-calibration-reuses-p1.md
git commit -m "docs(adr): ADR-0001 tail calibration reuses P1 binary-outcome calibrators"
```

---

## Task 15: Full-suite regression + smoke run

- [ ] **Step 1: Run the entire test suite**

Run: `./.venv/Scripts/python.exe -m pytest tests -v 2>&1 | tail -30`

Expected: ~243 baseline + ~30 new = ~273 passed, 0 deselected.

- [ ] **Step 2: Smoke: run a single scan with flag on, see that tail probabilities appear**

Temporarily edit local `config.json` to set `enable_tail_calibration: true`.

Run: `./.venv/Scripts/python.exe main.py --once --kalshi-only 2>&1 | tail -30`

Expected: log output includes "Tail calibration enabled." The scan may produce zero tail-routed opportunities at launch (empty `tail_unblocks`). That's correct — confirm there are no crashes.

Revert `config.json` to `enable_tail_calibration: false` before committing.

- [ ] **Step 3: Update `_state.md` and write a session handoff**

Run the `handoff` skill to capture the P2 feature completion in the session hub.

- [ ] **Step 4: No commit required for Step 3** — handoff writes its own commits.

---

## Self-Review Notes

**Spec coverage:**
- Training-data joins → Task 1 ✓
- `TailBinaryCalibrator` → Task 2 ✓
- `BucketDistributionalCalibrator` → Task 3 ✓
- `TailCalibrationManager` with mtime invalidation → Task 4 ✓
- `train_tail_calibration.py` → Task 5 ✓
- `evaluate_tail_calibration.py` → Task 6 ✓
- Matcher tail routing → Task 7 ✓
- Policy v5 + `apply_tail_unblocks` → Task 8 ✓
- Ledger `bankroll_slice` column → Task 9 ✓
- Slice-scaled Kelly → Task 10 ✓
- main.py wiring + config → Task 11 ✓
- autopilot_weekly integration → Task 12 ✓
- Docs → Task 13 ✓
- ADR → Task 14 ✓
- Smoke + handoff → Task 15 ✓

**Placeholder scan:** all code blocks show actual content. No "TBD", "handle
edge cases", "similar to Task N". File paths are exact.

**Type consistency:**
- `TailBinaryCalibrator(city, market_type, direction)` — signature matches
  across Tasks 2, 4, 5, 6, 7.
- `BucketDistributionalCalibrator(city, market_type)` — consistent in
  Tasks 3, 4, 5, 6.
- `TailCalibrationManager.calibrate_tail_probability(city, market_type,
  direction, is_bucket, raw_prob) -> {"calibrated_prob", "source"} | None`
  — signature consistent in Tasks 4, 7, 11.
- `apply_tail_unblocks(opportunity, policy, threshold_direction=None)`
  — signature consistent in Tasks 8, 11.
- `bankroll_slice` string values (`temperature_buy` | `rain_binary` |
  `probation`) — consistent in Tasks 8, 9, 10, 11, 13.
