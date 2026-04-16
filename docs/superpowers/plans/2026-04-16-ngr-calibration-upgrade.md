# NGR Calibration Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace linear EMOS with joint (μ, σ) Non-homogeneous Gaussian Regression, add opportunity-level archive logging, and introduce quarter-Kelly sizing with Policy v4 — all gated behind config flags so rollout is phased and reversible.

**Architecture:** New `NGRCalibrator` class in `src/calibration.py` fit by CRPS minimization. `CalibrationManager.predict_distribution()` returns (μ, σ, source) for the matcher to use in probability computation. A new `src/opportunity_log.py` writes every scored opportunity to `data/opportunity_archive/YYYY-MM-DD.csv` with a settlement pass. A new `compute_position_size` in `src/paper_trading.py` sizes trades by exact binary Kelly. Policy v4 lowers thresholds and increases candidate count, deferred behind a policy-file swap.

**Tech Stack:** Python 3.12, numpy, pandas, scipy.optimize (L-BFGS-B for CRPS minimization), scikit-learn (existing IsotonicRegression retained), pytest.

**Spec reference:** [`docs/superpowers/specs/2026-04-16-ngr-calibration-upgrade-design.md`](../specs/2026-04-16-ngr-calibration-upgrade-design.md)

---

## File Structure

**New files:**
- `src/ngr.py` — NGR model class, CRPS helper, fit + predict (isolated from `calibration.py` which is already 438 lines)
- `src/opportunity_log.py` — archive writer + settler
- `tests/test_ngr.py` — unit tests for CRPS, fit, predict, persistence
- `tests/test_opportunity_log.py` — unit tests for log + settle
- `tests/test_kelly_sizing.py` — unit tests for sizing

**Modified:**
- `src/calibration.py` — `CalibrationManager.predict_distribution()` dispatches to NGR → EMOS → raw fallback chain
- `src/matcher.py` — compute `lead_h` and `doy`, call `predict_distribution()`, remove σ clamp on the active path
- `src/station_truth.py` — no schema change; `build_training_set` output already has `date`, `forecast_lead_days`, `as_of_utc` — we derive `lead_h` and `doy` in NGR training
- `src/paper_trading.py` — add `compute_position_size()`, wire into `log_paper_trades`
- `train_calibration.py` — fit NGR alongside EMOS + isotonic
- `evaluate_calibration.py` — add NGR as a named policy in holdout comparison
- `main.py` — write opportunity archive each scan; settle archive on `--settle-paper-trades`
- `config.example.json` — new config keys
- `strategy/strategy_policy.json` — v4 payload, previous saved as `strategy_policy_v3.json`

---

## Task 1 — CRPS closed-form helper

**Files:**
- Create: `src/ngr.py`
- Test: `tests/test_ngr.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ngr.py
"""Tests for Non-homogeneous Gaussian Regression."""

import math
import numpy as np
import pytest

from src.ngr import gaussian_crps


def test_gaussian_crps_closed_form_zero_at_perfect_forecast():
    # Perfect forecast (mu == y) with sigma -> 0 should give CRPS -> 0.
    assert gaussian_crps(mu=10.0, sigma=1e-6, y=10.0) == pytest.approx(0.0, abs=1e-3)


def test_gaussian_crps_matches_known_value():
    # sigma=1, y=mu: CRPS = sigma * (2*phi(0) - 1/sqrt(pi)) = 1 * (2/sqrt(2*pi) - 1/sqrt(pi))
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ngr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ngr'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ngr.py
"""Non-homogeneous Gaussian Regression calibration.

Fits a predictive normal distribution whose mean and variance are both
functions of forecast features. Trained by minimizing the closed-form
Gaussian CRPS (Gneiting et al. 2005).
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm


def gaussian_crps(mu, sigma, y):
    """Closed-form continuous ranked probability score for N(mu, sigma^2).

    CRPS(N(mu,sigma^2), y) = sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (y - mu) / sigma.

    Scalar or numpy array inputs supported.
    """
    mu_a = np.asarray(mu, dtype=float)
    sigma_a = np.asarray(sigma, dtype=float)
    y_a = np.asarray(y, dtype=float)
    sigma_a = np.maximum(sigma_a, 1e-9)
    z = (y_a - mu_a) / sigma_a
    term = z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi)
    result = sigma_a * term
    if np.isscalar(mu) and np.isscalar(sigma) and np.isscalar(y):
        return float(result)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ngr.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ngr.py tests/test_ngr.py
git commit -m "feat(ngr): add Gaussian CRPS closed-form helper"
```

---

## Task 2 — NGRCalibrator dataclass + feature design

**Files:**
- Modify: `src/ngr.py`
- Modify: `tests/test_ngr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ngr.py`:

```python
import pandas as pd
from src.ngr import NGRCalibrator, build_ngr_features


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ngr.py::test_build_ngr_features_adds_lead_hours_and_doy tests/test_ngr.py::test_ngr_calibrator_initial_state -v`
Expected: FAIL with `ImportError: cannot import name 'NGRCalibrator'`

- [ ] **Step 3: Write implementation**

Append to `src/ngr.py`:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


def build_ngr_features(df: pd.DataFrame, market_type: str) -> pd.DataFrame:
    """Produce the feature frame used to fit and predict with NGR.

    Expects build_training_set output columns plus `date`, `forecast_lead_days`,
    `as_of_utc`. Returns a frame with columns:
    forecast_f, actual_f, spread_f, lead_h, doy, sin_doy, cos_doy.
    """
    if market_type not in {"high", "low"}:
        raise ValueError(f"Unsupported market_type: {market_type}")

    forecast_col = f"forecast_{market_type}_f"
    actual_col = f"actual_{market_type}_f"
    spread_col = f"ensemble_{market_type}_std_f"

    work = df.copy()
    if spread_col not in work.columns or work[spread_col].isna().all():
        spread_col = "ensemble_std_f"

    out = pd.DataFrame({
        "forecast_f": pd.to_numeric(work[forecast_col], errors="coerce"),
        "actual_f": pd.to_numeric(work[actual_col], errors="coerce"),
        "spread_f": pd.to_numeric(work.get(spread_col, 1.0), errors="coerce"),
    })

    out["lead_h"] = pd.to_numeric(work["forecast_lead_days"], errors="coerce") * 24.0
    parsed_date = pd.to_datetime(work["date"], errors="coerce")
    out["doy"] = parsed_date.dt.dayofyear.astype("Int64")
    out["sin_doy"] = np.sin(2.0 * math.pi * out["doy"].astype(float) / 365.0)
    out["cos_doy"] = np.cos(2.0 * math.pi * out["doy"].astype(float) / 365.0)

    out = out.dropna(subset=["forecast_f", "actual_f", "lead_h", "doy"]).copy()
    out["spread_f"] = out["spread_f"].fillna(out["spread_f"].median() if not out["spread_f"].dropna().empty else 1.0)
    out["spread_f"] = out["spread_f"].clip(lower=0.1)
    return out.reset_index(drop=True)


@dataclass
class NGRCalibrator:
    """Joint (mu, sigma^2) regression calibrated by CRPS minimization."""

    city: str
    market_type: str
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(6))
    beta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sigma2_floor: float = 1.0
    training_rows: int = 0
    training_crps: float = 0.0
    is_fitted: bool = False
```

Also add at the top of `src/ngr.py` imports if missing:
```python
import math
import numpy as np
from scipy.stats import norm
import pandas as pd
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ngr.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ngr.py tests/test_ngr.py
git commit -m "feat(ngr): add NGRCalibrator dataclass and feature builder"
```

---

## Task 3 — NGRCalibrator.fit (CRPS optimization)

**Files:**
- Modify: `src/ngr.py`
- Modify: `tests/test_ngr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ngr.py`:

```python
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

    df = pd.DataFrame({
        "forecast_high_f": forecast,
        "actual_high_f": actual,
        "ensemble_high_std_f": spread,
        "forecast_lead_days": lead_h / 24,
        "date": [f"2025-{((d-1)//30)%12+1:02d}-{(d-1)%30+1:02d}" for d in doy],
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ngr.py::test_ngr_fit_recovers_known_relationship tests/test_ngr.py::test_ngr_fit_raises_on_too_few_rows -v`
Expected: FAIL — `fit` and `predict` methods not implemented.

- [ ] **Step 3: Write implementation**

Add to `NGRCalibrator` in `src/ngr.py`:

```python
from scipy.optimize import minimize


def _design_matrix_mu(feats: pd.DataFrame) -> np.ndarray:
    """[1, f, s, sin_doy, cos_doy, lead_h] — 6 columns."""
    return np.column_stack([
        np.ones(len(feats)),
        feats["forecast_f"].to_numpy(dtype=float),
        feats["spread_f"].to_numpy(dtype=float),
        feats["sin_doy"].to_numpy(dtype=float),
        feats["cos_doy"].to_numpy(dtype=float),
        feats["lead_h"].to_numpy(dtype=float),
    ])


def _design_matrix_sigma2(feats: pd.DataFrame) -> np.ndarray:
    """[1, s^2, lead_h] — 3 columns."""
    s = feats["spread_f"].to_numpy(dtype=float)
    return np.column_stack([
        np.ones(len(feats)),
        s * s,
        feats["lead_h"].to_numpy(dtype=float),
    ])


def _ngr_objective(params: np.ndarray, X_mu, X_sig, y, sigma2_floor) -> float:
    alpha = params[:6]
    beta = params[6:]
    mu = X_mu @ alpha
    sigma2_raw = X_sig @ beta
    sigma2 = np.maximum(sigma2_raw, sigma2_floor)
    sigma = np.sqrt(sigma2)
    return float(np.mean(gaussian_crps(mu, sigma, y)))


class NGRCalibrator:  # extend dataclass (keep fields from Task 2)
    ...

    def fit(self, df: pd.DataFrame, min_rows: int = 20) -> "NGRCalibrator":
        feats = build_ngr_features(df, self.market_type)
        if len(feats) < min_rows:
            raise ValueError(
                f"Need at least {min_rows} training rows for {self.city} {self.market_type} NGR, got {len(feats)}"
            )

        y = feats["actual_f"].to_numpy(dtype=float)
        X_mu = _design_matrix_mu(feats)
        X_sig = _design_matrix_sigma2(feats)

        # Empirical sigma^2 floor = 10th percentile of squared residuals vs raw forecast
        raw_residual = y - feats["forecast_f"].to_numpy(dtype=float)
        self.sigma2_floor = float(max(0.04, np.quantile(raw_residual ** 2, 0.10)))

        # Initial guess: identity mean, variance initialized to overall residual variance
        alpha0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        beta0 = np.array([float(np.var(raw_residual)), 0.0, 0.0])
        params0 = np.concatenate([alpha0, beta0])

        result = minimize(
            _ngr_objective,
            params0,
            args=(X_mu, X_sig, y, self.sigma2_floor),
            method="L-BFGS-B",
            options={"maxiter": 200},
        )

        if not result.success:
            # Retry with a different initialization before falling over
            alpha0_retry = np.zeros(6)
            alpha0_retry[1] = 1.0
            beta0_retry = np.array([1.0, 0.0, 0.0])
            result = minimize(
                _ngr_objective,
                np.concatenate([alpha0_retry, beta0_retry]),
                args=(X_mu, X_sig, y, self.sigma2_floor),
                method="L-BFGS-B",
                options={"maxiter": 500},
            )
            if not result.success:
                raise RuntimeError(
                    f"NGR optimization failed for {self.city} {self.market_type}: {result.message}"
                )

        self.alpha = result.x[:6]
        self.beta = result.x[6:]
        self.training_rows = int(len(feats))
        self.training_crps = float(result.fun)
        self.is_fitted = True
        return self

    def predict(self, forecast_f: float, spread_f: float, lead_h: float, doy: int) -> tuple[float, float]:
        """Return (mu, sigma). If unfitted, returns (forecast_f, max(spread_f, 1.0))."""
        if not self.is_fitted:
            return float(forecast_f), max(float(spread_f), 1.0)

        sin_d = math.sin(2.0 * math.pi * float(doy) / 365.0)
        cos_d = math.cos(2.0 * math.pi * float(doy) / 365.0)
        x_mu = np.array([1.0, float(forecast_f), float(spread_f), sin_d, cos_d, float(lead_h)])
        x_sig = np.array([1.0, float(spread_f) ** 2, float(lead_h)])

        mu = float(x_mu @ self.alpha)
        sigma2 = float(x_sig @ self.beta)
        sigma2 = max(sigma2, self.sigma2_floor)
        # Hard sanity bounds to catch pathological extrapolation
        sigma = math.sqrt(max(sigma2, 0.0625))  # floor at 0.25F
        sigma = min(sigma, 12.0)  # cap at 12F
        return mu, sigma
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ngr.py -v`
Expected: 8 tests PASS. Fit should complete in < 2 seconds.

- [ ] **Step 5: Commit**

```bash
git add src/ngr.py tests/test_ngr.py
git commit -m "feat(ngr): add fit() via L-BFGS-B CRPS minimization and predict()"
```

---

## Task 4 — NGRCalibrator persistence

**Files:**
- Modify: `src/ngr.py`
- Modify: `tests/test_ngr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ngr.py`:

```python
def test_ngr_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "forecast_high_f": rng.uniform(60, 90, n),
        "actual_high_f": rng.uniform(60, 90, n),
        "ensemble_high_std_f": rng.uniform(0.5, 2.5, n),
        "forecast_lead_days": [1] * n,
        "date": [f"2025-{(i%12)+1:02d}-{(i%28)+1:02d}" for i in range(n)],
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ngr.py::test_ngr_save_and_load_round_trip -v`
Expected: FAIL — `save`/`load` not implemented.

- [ ] **Step 3: Write implementation**

Append to `NGRCalibrator` class in `src/ngr.py`:

```python
    def save(self, path) -> "Path":
        from pathlib import Path
        import pickle
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, path) -> "NGRCalibrator":
        from pathlib import Path
        import pickle
        with open(Path(path), "rb") as f:
            return pickle.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ngr.py -v`
Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ngr.py tests/test_ngr.py
git commit -m "feat(ngr): add pickle persistence for NGRCalibrator"
```

---

## Task 5 — Integrate NGR into train_city_models

**Files:**
- Modify: `src/calibration.py:314-362` (the `train_city_models` function)
- Modify: `tests/test_calibration.py` (add new test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calibration.py`:

```python
def test_train_city_models_fits_ngr_when_enough_rows(tmp_path):
    import numpy as np
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_calibration.py::test_train_city_models_fits_ngr_when_enough_rows -v`
Expected: FAIL — `trained_ngr` key missing.

- [ ] **Step 3: Write implementation**

In `src/calibration.py`, modify `train_city_models` to fit NGR after the EMOS + isotonic block. Add these imports at top of file:

```python
from src.ngr import NGRCalibrator
```

Replace the outcome dict initialization and end of `train_city_models` loop body:

```python
        outcome: dict[str, object] = {
            "rows": int(len(prepared)),
            "emos_path": None,
            "isotonic_path": None,
            "ngr_path": None,
            "trained_emos": False,
            "trained_isotonic": False,
            "trained_ngr": False,
            "ngr_training_crps": None,
            "status": "skipped",
            "reason": "",
        }
```

And after the isotonic block (before `results[market_type] = outcome`):

```python
        # NGR — uses the full df (not `prepared`) because it needs date + lead columns
        ngr_min_rows = 20
        try:
            ngr_calibrator = NGRCalibrator(city=city, market_type=market_type).fit(
                training_df, min_rows=ngr_min_rows
            )
            ngr_path = calibration_model_path(city, market_type, "ngr", model_dir=model_dir)
            ngr_calibrator.save(ngr_path)
            outcome["trained_ngr"] = True
            outcome["ngr_path"] = str(ngr_path)
            outcome["ngr_training_crps"] = float(ngr_calibrator.training_crps)
        except (ValueError, RuntimeError) as exc:
            logger.warning("NGR fit skipped for %s %s: %s", city, market_type, exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_calibration.py -v`
Expected: All existing tests still pass, new NGR test passes.

- [ ] **Step 5: Commit**

```bash
git add src/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): fit NGR alongside EMOS and isotonic in train_city_models"
```

---

## Task 6 — Extend train_calibration.py CLI output

**Files:**
- Modify: `train_calibration.py:63-120`

- [ ] **Step 1: Update script**

Modify `train_calibration.py`:

```python
    total_city_models = 0
    total_isotonic_models = 0
    total_ngr_models = 0
    skipped_cities = 0

    log.info("Training calibration models for %d cities (window=%d days)...", len(cities), days)

    for city in cities:
        training_df = build_training_set(...)
        # ... existing code ...
        city_trained = 0
        city_isotonic = 0
        city_ngr = 0
        for market_type, outcome in results.items():
            if outcome.get("trained_emos"):
                city_trained += 1
                total_city_models += 1
            if outcome.get("trained_isotonic"):
                city_isotonic += 1
                total_isotonic_models += 1
            if outcome.get("trained_ngr"):
                city_ngr += 1
                total_ngr_models += 1

            log.info(
                "%s %s: status=%s rows=%s ngr_crps=%s reason=%s",
                city,
                market_type,
                outcome.get("status"),
                outcome.get("rows"),
                outcome.get("ngr_training_crps"),
                outcome.get("reason", ""),
            )

        if city_trained == 0:
            skipped_cities += 1
            log.info("%s: insufficient overlapping rows for training", city)
        else:
            log.info(
                "%s: trained %d EMOS, %d isotonic, %d NGR",
                city, city_trained, city_isotonic, city_ngr,
            )

    log.info(
        "Training complete: %d EMOS, %d isotonic, %d NGR, %d skipped",
        total_city_models, total_isotonic_models, total_ngr_models, skipped_cities,
    )
```

- [ ] **Step 2: Smoke test**

Run: `.\.venv\Scripts\python.exe train_calibration.py --city Austin --days 365`
Expected: Log line includes `NGR` count and `ngr_crps=<float>`. An `austin_high_ngr.pkl` file appears in `data/calibration_models/`.

- [ ] **Step 3: Commit**

```bash
git add train_calibration.py
git commit -m "feat(train_calibration): report NGR model counts and training CRPS"
```

---

## Task 7 — CalibrationManager.predict_distribution

**Files:**
- Modify: `src/calibration.py` (class `CalibrationManager`, near `correct_forecast`)
- Modify: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_calibration.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_calibration.py::test_calibration_manager_predict_distribution_uses_ngr_when_available -v`
Expected: FAIL — `predict_distribution` not defined.

- [ ] **Step 3: Write implementation**

In `src/calibration.py`, add to `CalibrationManager`:

```python
    def _load_ngr(self, city: str, market_type: str):
        from src.ngr import NGRCalibrator
        key = (city, market_type)
        cache_attr = "_ngr_cache"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache = getattr(self, cache_attr)
        if key in cache:
            return cache[key]

        path = calibration_model_path(city, market_type, "ngr", model_dir=self.model_dir)
        if not path.exists():
            cache[key] = None
            return None
        try:
            model = NGRCalibrator.load(path)
        except Exception as exc:
            logger.warning("Failed loading NGR model %s: %s", path, exc)
            model = None
        cache[key] = model
        return model

    def predict_distribution(
        self,
        city: str,
        market_type: str,
        forecast_f: float,
        spread_f: float,
        lead_h: float,
        doy: int,
    ) -> tuple[float, float, str]:
        """Return (mu, sigma, source) for the probability computation.

        Source ordering (first applicable):
          - raw_selective_fallback — pair in selective list; uses raw mu, NGR sigma if available
          - ngr — use NGR predictive distribution
          - emos — fall back to legacy EMOS (mu only); sigma = max(spread_f, 1.0)
          - raw — no models available; return forecast + max(spread_f, 1.0)
        """
        selective = is_selective_raw_fallback_pair(city, market_type)

        ngr_model = self._load_ngr(city, market_type)
        if ngr_model is not None and ngr_model.is_fitted:
            mu_ngr, sigma_ngr = ngr_model.predict(forecast_f, spread_f, lead_h, doy)
            if selective:
                # Keep raw mu but take NGR sigma for better uncertainty
                return float(forecast_f), float(sigma_ngr), SELECTIVE_RAW_FALLBACK_SOURCE
            return float(mu_ngr), float(sigma_ngr), "ngr"

        # No NGR — try legacy EMOS for mu
        emos_model = self._load_emos(city, market_type)
        if emos_model is not None and emos_model.is_fitted:
            if selective:
                return float(forecast_f), max(float(spread_f), 1.0), SELECTIVE_RAW_FALLBACK_SOURCE
            return float(emos_model.correct(forecast_f, spread_f)), max(float(spread_f), 1.0), "emos"

        # Nothing — raw passthrough
        return float(forecast_f), max(float(spread_f), 1.0), "raw"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_calibration.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): add CalibrationManager.predict_distribution with NGR→EMOS→raw chain"
```

---

## Task 8 — Matcher uses predict_distribution (behind config flag)

**Files:**
- Modify: `src/matcher.py:337-502` (the `match_kalshi_markets` function)
- Modify: `src/matcher.py:505-652` (the `match_polymarket_markets` function)
- Modify: `tests/test_matcher.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_matcher.py`:

```python
def test_matcher_uses_ngr_when_flag_enabled(monkeypatch):
    """When use_ngr_calibration=True, matcher should call predict_distribution and use its sigma."""
    from src.matcher import match_kalshi_markets

    class StubManager:
        def __init__(self):
            self.calls = []

        def predict_distribution(self, city, market_type, forecast_f, spread_f, lead_h, doy):
            self.calls.append((city, market_type, forecast_f, spread_f, lead_h, doy))
            return forecast_f + 0.5, 2.5, "ngr"  # mu, sigma, source

        def calibrate_probability(self, city, market_type, raw_prob):
            return raw_prob, "raw"

    stub = StubManager()
    forecasts = {
        "Austin": {
            "timezone": "America/Chicago",
            "hourly": {
                "time": ["2026-04-17T00:00"] + [f"2026-04-17T{h:02d}:00" for h in range(1, 24)],
                "temperature_2m": [20.0] + [22.0 + (h % 3) for h in range(1, 24)],
            },
        }
    }
    markets = [{
        "city": "Austin",
        "type": "high",
        "threshold": 75.0,
        "ticker": "KXHIGHTAUS-26APR17-T75",
        "title": "Will the maximum temperature be  >75° on Apr 17, 2026?",
        "yes_sub_title": "75 or above",
        "last_price": 0.50,
        "yes_bid": 0.48,
        "yes_ask": 0.52,
        "close_time": "2026-04-17T23:59:00+00:00",
        "volume_24h": 5000,
    }]

    opps = match_kalshi_markets(
        forecasts,
        markets,
        min_edge=0.0,
        uncertainty_std_f=2.0,
        calibration_manager=stub,
        use_ngr_calibration=True,
    )

    assert len(stub.calls) == 1
    city, mtype, _, _, lead_h, doy = stub.calls[0]
    assert city == "Austin"
    assert mtype == "high"
    assert lead_h > 0  # calculated from close_time
    assert 1 <= doy <= 366
    # Matcher used the sigma from predict_distribution, not ensemble clamp
    assert opps[0]["uncertainty_std_f"] == 2.5
    assert opps[0]["forecast_calibration_source"] == "ngr"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_matcher.py::test_matcher_uses_ngr_when_flag_enabled -v`
Expected: FAIL — `use_ngr_calibration` parameter not accepted.

- [ ] **Step 3: Write implementation**

In `src/matcher.py`:

Add helper near top:

```python
def _doy_from_date(target_date: Optional[str]) -> int:
    if not target_date:
        return 1
    try:
        return datetime.fromisoformat(target_date).timetuple().tm_yday
    except ValueError:
        return 1


def _lead_hours(close_time: str, now_utc: Optional[datetime]) -> float:
    close_dt = _parse_iso_datetime(close_time)
    if close_dt is None:
        return 24.0
    current = now_utc or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (close_dt.astimezone(timezone.utc) - current.astimezone(timezone.utc)).total_seconds() / 3600.0)
```

Update `match_kalshi_markets` signature to add parameter:

```python
def match_kalshi_markets(
    forecasts: dict,
    kalshi_markets: list[dict],
    min_edge: float = 0.05,
    uncertainty_std_f: float = 2.0,
    ensemble_data: Optional[dict] = None,
    calibration_manager: Optional["CalibrationManager"] = None,
    hrrr_data: Optional[dict] = None,
    hrrr_blend_horizon_hours: float = 18.0,
    now_utc: Optional[datetime] = None,
    use_ngr_calibration: bool = False,
) -> list[dict]:
```

Replace the `sigma_f, sigma_source = _resolve_temperature_uncertainty(...)` and `forecast_value, forecast_calibration_source = _apply_calibration(...)` block (around lines 375–401) with:

```python
        ensemble_sigma_f, sigma_source = _resolve_temperature_uncertainty(
            ensemble_data, city, market_date, mtype, uncertainty_std_f,
        )

        raw_forecast_value, forecast_blend_source, hrrr_forecast_value, hrrr_weight, hours_to_settlement = _apply_hrrr_blend(
            hrrr_data, city, market_date, mtype,
            str(m.get("close_time", "")),
            open_meteo_forecast_value,
            forecast.get("timezone"),
            hrrr_blend_horizon_hours, now_utc,
        )

        if use_ngr_calibration and calibration_manager is not None:
            lead_h = _lead_hours(str(m.get("close_time", "")), now_utc)
            doy = _doy_from_date(market_date)
            forecast_value, sigma_f, forecast_calibration_source = calibration_manager.predict_distribution(
                city, mtype, raw_forecast_value, ensemble_sigma_f, lead_h, doy,
            )
        else:
            sigma_f = ensemble_sigma_f
            forecast_value, forecast_calibration_source = _apply_calibration(
                calibration_manager, city, mtype, raw_forecast_value, sigma_f,
            )
```

Apply the same change to `match_polymarket_markets` (add `use_ngr_calibration` param, same replacement).

Update `scan_all` to thread `use_ngr_calibration` through.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_matcher.py -v`
Expected: New test passes, existing tests still pass (they call without `use_ngr_calibration`, which defaults to False).

- [ ] **Step 5: Commit**

```bash
git add src/matcher.py tests/test_matcher.py
git commit -m "feat(matcher): add use_ngr_calibration flag routing through predict_distribution"
```

---

## Task 9 — Opportunity log module

**Files:**
- Create: `src/opportunity_log.py`
- Create: `tests/test_opportunity_log.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_opportunity_log.py
"""Tests for opportunity archive logging."""

import pandas as pd
import pytest
from pathlib import Path

from src.opportunity_log import log_opportunities, OPPORTUNITY_ARCHIVE_COLUMNS


def test_log_opportunities_creates_file_with_expected_columns(tmp_path):
    scan_id = "2026-04-17T12:00:00+00:00"
    opps = [
        {
            "source": "kalshi",
            "ticker": "KXHIGHTAUS-26APR17-T75",
            "city": "Austin",
            "market_type": "high",
            "market_date": "2026-04-17",
            "outcome": ">75°F",
            "our_probability": 0.62,
            "raw_probability": 0.58,
            "market_price": 0.45,
            "edge": 0.17,
            "abs_edge": 0.17,
            "forecast_value_f": 77.5,
            "raw_forecast_value_f": 77.0,
            "uncertainty_std_f": 2.3,
            "forecast_blend_source": "open-meteo",
            "forecast_calibration_source": "ngr",
            "probability_calibration_source": "raw",
            "hours_to_settlement": 22.0,
            "volume24hr": 4200,
        }
    ]
    path = log_opportunities(scan_id, opps, archive_dir=tmp_path)
    assert path.exists()
    assert path.name == "2026-04-17.csv"

    frame = pd.read_csv(path)
    assert set(frame.columns) == set(OPPORTUNITY_ARCHIVE_COLUMNS)
    assert frame.iloc[0]["ticker"] == "KXHIGHTAUS-26APR17-T75"
    assert frame.iloc[0]["our_probability"] == 0.62


def test_log_opportunities_appends_on_second_scan_same_day(tmp_path):
    scan_id_a = "2026-04-17T12:00:00+00:00"
    scan_id_b = "2026-04-17T12:30:00+00:00"

    opp = {
        "source": "kalshi", "ticker": "X", "city": "A", "market_type": "high",
        "market_date": "2026-04-17", "outcome": ">75°F",
        "our_probability": 0.5, "raw_probability": 0.5, "market_price": 0.3,
        "edge": 0.2, "abs_edge": 0.2, "forecast_value_f": 76.0,
        "raw_forecast_value_f": 76.0, "uncertainty_std_f": 2.0,
        "forecast_blend_source": "open-meteo", "forecast_calibration_source": "ngr",
        "probability_calibration_source": "raw", "hours_to_settlement": 23.0,
        "volume24hr": 1000,
    }
    log_opportunities(scan_id_a, [opp], archive_dir=tmp_path)
    log_opportunities(scan_id_b, [opp], archive_dir=tmp_path)

    frame = pd.read_csv(tmp_path / "2026-04-17.csv")
    assert len(frame) == 2
    assert frame.iloc[0]["scan_id"] == scan_id_a
    assert frame.iloc[1]["scan_id"] == scan_id_b


def test_log_opportunities_empty_list_is_noop(tmp_path):
    path = log_opportunities("2026-04-17T12:00:00+00:00", [], archive_dir=tmp_path)
    assert path is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_opportunity_log.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write implementation**

```python
# src/opportunity_log.py
"""Append every scored opportunity to a daily CSV archive for post-hoc evaluation.

This is write-only — it does NOT feed back into trade selection. It exists
so calibration can be evaluated on hundreds of labeled (prob, outcome) pairs
per day instead of the 0-3 from actual paper trades.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("weather.opportunity_log")

DEFAULT_ARCHIVE_DIR = Path("data/opportunity_archive")

OPPORTUNITY_ARCHIVE_COLUMNS = [
    "scan_id",
    "recorded_at_utc",
    "source",
    "ticker",
    "city",
    "market_type",
    "market_date",
    "outcome",
    "our_probability",
    "raw_probability",
    "market_price",
    "edge",
    "abs_edge",
    "forecast_value_f",
    "raw_forecast_value_f",
    "uncertainty_std_f",
    "forecast_blend_source",
    "forecast_calibration_source",
    "probability_calibration_source",
    "hours_to_settlement",
    "volume24hr",
    "yes_outcome",
    "actual_value_f",
    "settled_at_utc",
]


def _archive_path_for_date(archive_dir: Path, date_str: str) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir / f"{date_str}.csv"


def _row_from_opportunity(scan_id: str, opp: dict) -> dict:
    return {
        "scan_id": scan_id,
        "recorded_at_utc": scan_id,
        "source": opp.get("source"),
        "ticker": opp.get("ticker"),
        "city": opp.get("city"),
        "market_type": opp.get("market_type"),
        "market_date": opp.get("market_date"),
        "outcome": opp.get("outcome"),
        "our_probability": opp.get("our_probability"),
        "raw_probability": opp.get("raw_probability"),
        "market_price": opp.get("market_price"),
        "edge": opp.get("edge"),
        "abs_edge": opp.get("abs_edge"),
        "forecast_value_f": opp.get("forecast_value_f"),
        "raw_forecast_value_f": opp.get("raw_forecast_value_f"),
        "uncertainty_std_f": opp.get("uncertainty_std_f"),
        "forecast_blend_source": opp.get("forecast_blend_source"),
        "forecast_calibration_source": opp.get("forecast_calibration_source"),
        "probability_calibration_source": opp.get("probability_calibration_source"),
        "hours_to_settlement": opp.get("hours_to_settlement"),
        "volume24hr": opp.get("volume24hr"),
        "yes_outcome": None,
        "actual_value_f": None,
        "settled_at_utc": None,
    }


def log_opportunities(
    scan_id: str,
    opportunities: list[dict],
    archive_dir: Path | str = DEFAULT_ARCHIVE_DIR,
) -> Optional[Path]:
    if not opportunities:
        return None
    archive_dir = Path(archive_dir)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    path = _archive_path_for_date(archive_dir, today)

    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OPPORTUNITY_ARCHIVE_COLUMNS)
        if write_header:
            writer.writeheader()
        for opp in opportunities:
            writer.writerow(_row_from_opportunity(scan_id, opp))
    logger.info("Logged %d opportunities to %s", len(opportunities), path)
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_opportunity_log.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/opportunity_log.py tests/test_opportunity_log.py
git commit -m "feat(opportunity_log): add daily-csv opportunity archive writer"
```

---

## Task 10 — Opportunity archive settlement

**Files:**
- Modify: `src/opportunity_log.py`
- Modify: `tests/test_opportunity_log.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_opportunity_log.py`:

```python
def test_settle_opportunity_archive_fills_outcome(tmp_path):
    from src.opportunity_log import settle_opportunity_archive

    archive_file = tmp_path / "2026-04-17.csv"
    df = pd.DataFrame([
        {
            "scan_id": "s1", "recorded_at_utc": "s1", "source": "kalshi",
            "ticker": "T1", "city": "Austin", "market_type": "high",
            "market_date": "2026-04-17", "outcome": ">75°F",
            "our_probability": 0.6, "raw_probability": 0.55, "market_price": 0.4,
            "edge": 0.2, "abs_edge": 0.2, "forecast_value_f": 77.0,
            "raw_forecast_value_f": 77.0, "uncertainty_std_f": 2.0,
            "forecast_blend_source": "open-meteo", "forecast_calibration_source": "ngr",
            "probability_calibration_source": "raw", "hours_to_settlement": 20.0,
            "volume24hr": 5000, "yes_outcome": None, "actual_value_f": None,
            "settled_at_utc": None,
        },
        {
            "scan_id": "s2", "recorded_at_utc": "s2", "source": "kalshi",
            "ticker": "T2", "city": "Austin", "market_type": "high",
            "market_date": "2026-04-17", "outcome": ">80°F",
            "our_probability": 0.3, "raw_probability": 0.25, "market_price": 0.5,
            "edge": -0.2, "abs_edge": 0.2, "forecast_value_f": 77.0,
            "raw_forecast_value_f": 77.0, "uncertainty_std_f": 2.0,
            "forecast_blend_source": "open-meteo", "forecast_calibration_source": "ngr",
            "probability_calibration_source": "raw", "hours_to_settlement": 20.0,
            "volume24hr": 5000, "yes_outcome": None, "actual_value_f": None,
            "settled_at_utc": None,
        },
    ])
    df.to_csv(archive_file, index=False)

    # Austin hit 78F: first opp (>75) YES, second (>80) NO
    station_actuals = {
        "Austin": pd.DataFrame([
            {"date": "2026-04-17", "tmax_f": 78.0, "tmin_f": 60.0},
        ]),
    }

    settled = settle_opportunity_archive(archive_dir=tmp_path, station_actuals=station_actuals)
    assert settled == 2

    after = pd.read_csv(archive_file)
    row_gt75 = after.loc[after["ticker"] == "T1"].iloc[0]
    row_gt80 = after.loc[after["ticker"] == "T2"].iloc[0]
    assert row_gt75["yes_outcome"] == 1
    assert row_gt80["yes_outcome"] == 0
    assert row_gt75["actual_value_f"] == 78.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_opportunity_log.py::test_settle_opportunity_archive_fills_outcome -v`
Expected: FAIL — `settle_opportunity_archive` missing.

- [ ] **Step 3: Write implementation**

Append to `src/opportunity_log.py`:

```python
def _parse_outcome_bounds(outcome: str) -> tuple[Optional[float], Optional[float]]:
    """Extract bounds from outcome labels like '>75°F', '<63°F', '66-67°F'."""
    if not outcome:
        return None, None
    cleaned = outcome.replace("°F", "").replace("°", "").strip()
    if cleaned.startswith(">"):
        return float(cleaned[1:]), None
    if cleaned.startswith("<"):
        return None, float(cleaned[1:])
    if "-" in cleaned:
        low_s, high_s = cleaned.split("-", 1)
        return float(low_s), float(high_s)
    return None, None


def _yes_outcome(low: Optional[float], high: Optional[float], actual_f: float, market_type: str) -> int:
    if low is not None and high is not None:
        return 1 if low <= actual_f <= high else 0
    if low is not None:
        return 1 if actual_f > low else 0   # threshold ">"
    if high is not None:
        return 1 if actual_f < high else 0  # threshold "<"
    return 0


def settle_opportunity_archive(
    archive_dir: Path | str,
    station_actuals: dict,
) -> int:
    """Join archive rows with station truth; fill yes_outcome/actual_value_f in-place.

    station_actuals: {city: pd.DataFrame with columns date, tmax_f, tmin_f}.
    Returns number of rows updated.
    """
    archive_dir = Path(archive_dir)
    if not archive_dir.exists():
        return 0

    now_iso = datetime.utcnow().isoformat() + "Z"
    total_updated = 0

    for csv_path in sorted(archive_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        frame_updated = False

        for idx, row in frame.iterrows():
            if not pd.isna(row.get("yes_outcome")):
                continue
            city = row["city"]
            actuals_df = station_actuals.get(city)
            if actuals_df is None or actuals_df.empty:
                continue
            match = actuals_df.loc[actuals_df["date"].astype(str) == str(row["market_date"])]
            if match.empty:
                continue
            actual_col = "tmax_f" if row["market_type"] == "high" else "tmin_f"
            actual_val = match.iloc[0][actual_col]
            if pd.isna(actual_val):
                continue
            low, high = _parse_outcome_bounds(str(row["outcome"]))
            if low is None and high is None:
                continue
            yo = _yes_outcome(low, high, float(actual_val), row["market_type"])
            frame.at[idx, "yes_outcome"] = yo
            frame.at[idx, "actual_value_f"] = float(actual_val)
            frame.at[idx, "settled_at_utc"] = now_iso
            frame_updated = True
            total_updated += 1

        if frame_updated:
            frame.to_csv(csv_path, index=False)

    logger.info("Settled %d archived opportunities across %s", total_updated, archive_dir)
    return total_updated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_opportunity_log.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/opportunity_log.py tests/test_opportunity_log.py
git commit -m "feat(opportunity_log): add settlement pass that joins station actuals"
```

---

## Task 11 — Wire opportunity logging into main.py scan

**Files:**
- Modify: `main.py` (scan function — search for where `all_opportunities` is built and reported)

- [ ] **Step 1: Find the wiring point**

```bash
grep -n "all_opportunities\|scan_timestamp\|scan_id" main.py | head -20
```

Look for the line where the scan produces its list of scored opportunities (before filtering). The opportunity log should receive ALL scored opps, not the filtered subset.

- [ ] **Step 2: Add import and call**

At the top of `main.py`:

```python
from src.opportunity_log import log_opportunities
```

After `all_opportunities` is built (before the policy filter), add:

```python
    if config.get("opportunity_archive_enabled", True):
        archive_dir = config.get("opportunity_archive_dir", "data/opportunity_archive")
        try:
            log_opportunities(
                scan_id=scan_timestamp_iso,  # use whatever timestamp var main already has
                opportunities=all_opportunities,
                archive_dir=archive_dir,
            )
        except Exception as exc:
            log.warning("Opportunity archive write failed: %s", exc)
```

(Use whatever variable names match the existing `main.py`. The scan already has a timestamp for the ledger; reuse it.)

- [ ] **Step 3: Smoke test**

Run: `.\.venv\Scripts\python.exe main.py --once`
Expected: Log line `Logged N opportunities to data/opportunity_archive/<today>.csv`. File exists and is populated.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(main): write scored opportunities to daily archive each scan"
```

---

## Task 12 — Wire settlement into main.py --settle-paper-trades

**Files:**
- Modify: `main.py` (settlement code path)

- [ ] **Step 1: Locate the existing settlement flow**

```bash
grep -n "settle_paper_trades\|--settle-paper-trades\|station_actuals" main.py | head -20
```

The existing settlement already loads station actuals for paper trades. Reuse that dict.

- [ ] **Step 2: Add call after paper trade settlement**

Import at top:

```python
from src.opportunity_log import settle_opportunity_archive
```

In the settlement branch, after the paper-trade settlement completes:

```python
    if config.get("opportunity_archive_enabled", True):
        archive_dir = config.get("opportunity_archive_dir", "data/opportunity_archive")
        try:
            count = settle_opportunity_archive(archive_dir=archive_dir, station_actuals=station_actuals_map)
            log.info("Opportunity archive: settled %d rows", count)
        except Exception as exc:
            log.warning("Opportunity archive settlement failed: %s", exc)
```

(Use whatever variable holds the `{city: DataFrame}` station actuals dict in the existing settlement code.)

- [ ] **Step 3: Smoke test**

Run: `.\.venv\Scripts\python.exe main.py --settle-paper-trades`
Expected: Log line `Opportunity archive: settled N rows`. Re-read a CSV in `data/opportunity_archive/`; rows for settled dates have `yes_outcome` filled in.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(main): settle opportunity archive alongside paper trades"
```

---

## Task 13 — Wire NGR flag into main.py matcher call

**Files:**
- Modify: `main.py` (the call to `scan_all` / `match_kalshi_markets`)

- [ ] **Step 1: Update matcher call**

Locate the `scan_all(...)` or `match_kalshi_markets(...)` call in `main.py`. Add:

```python
    use_ngr = bool(config.get("use_ngr_calibration", False))
    opportunities = scan_all(
        forecasts,
        kalshi_markets=kalshi_markets,
        poly_markets=poly_markets,
        min_edge=min_edge,
        uncertainty_std_f=uncertainty_std_f,
        ensemble_data=ensemble_data,
        calibration_manager=calibration_manager,
        hrrr_data=hrrr_data,
        hrrr_blend_horizon_hours=hrrr_blend_horizon_hours,
        now_utc=now_utc,
        use_ngr_calibration=use_ngr,
    )
```

Also update `scan_all` in `src/matcher.py` to accept and thread `use_ngr_calibration` (done in Task 8 but double-check).

- [ ] **Step 2: Smoke test both modes**

With `use_ngr_calibration: false` in config (default):
```
.\.venv\Scripts\python.exe main.py --once
```
Expected: opportunity archive shows `forecast_calibration_source` = `emos`/`raw`/`raw_selective_fallback` as before.

Flip config to `use_ngr_calibration: true` and re-run:
```
.\.venv\Scripts\python.exe main.py --once
```
Expected: `forecast_calibration_source` = `ngr` for pairs that have trained NGR models.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat(main): thread use_ngr_calibration config flag into scan_all"
```

---

## Task 14 — Evaluate NGR in holdout comparison

**Files:**
- Modify: `evaluate_calibration.py`
- Test: new function in existing tests (no dedicated test file — existing tests cover the CLI)

- [ ] **Step 1: Locate the policy comparison code**

```bash
grep -n "policy\|_evaluate_policy\|broad\|selective_fallback" evaluate_calibration.py | head -30
```

There should be an existing policy-evaluation function that takes training data and produces per-pair metrics. Read enough context to understand its inputs.

- [ ] **Step 2: Add NGR policy path**

Add a helper near the existing policy evaluation:

```python
def _evaluate_ngr_policy(
    training_df: pd.DataFrame,
    city: str,
    market_type: str,
    model_dir: Path,
) -> dict:
    """Evaluate pair under NGR policy on the holdout slice.

    Assumes training_df has already been split; this function just predicts
    with the loaded NGR model and returns {"mae": ..., "brier": ..., "crps": ...}.
    """
    from src.ngr import NGRCalibrator, build_ngr_features, gaussian_crps
    path = CALIBRATION_MODELS_DIR / f"{_slugify_city(city)}_{market_type}_ngr.pkl"
    if model_dir:
        path = Path(model_dir) / f"{_slugify_city(city)}_{market_type}_ngr.pkl"
    if not path.exists():
        return {"mae": None, "brier": None, "crps": None, "available": False}

    model = NGRCalibrator.load(path)
    feats = build_ngr_features(training_df, market_type)
    if feats.empty:
        return {"mae": None, "brier": None, "crps": None, "available": True, "reason": "no holdout rows"}

    # Use existing pipeline to get raw predictions, then apply NGR.
    mus, sigmas, crps_vals = [], [], []
    for _, row in feats.iterrows():
        mu, sigma = model.predict(row["forecast_f"], row["spread_f"], row["lead_h"], int(row["doy"]))
        mus.append(mu)
        sigmas.append(sigma)
        crps_vals.append(gaussian_crps(mu, sigma, row["actual_f"]))

    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    y = feats["actual_f"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(y - mus)))
    crps_mean = float(np.mean(crps_vals))
    # Brier on a threshold placeholder: use median forecast as threshold
    threshold = float(np.median(feats["forecast_f"]))
    prob_over = 1.0 - norm.cdf(threshold, loc=mus, scale=sigmas)
    outcome = (y > threshold).astype(float)
    brier = float(np.mean((prob_over - outcome) ** 2))

    return {"mae": mae, "brier": brier, "crps": crps_mean, "available": True}
```

Wire into the main policy comparison block so the output JSON/text includes an `ngr` key alongside `raw` and `selective_fallback`. Add to per-pair comparison and aggregate summary.

- [ ] **Step 3: Run it manually**

Run: `.\.venv\Scripts\python.exe evaluate_calibration.py --days 400 --holdout-days 30`
Expected: Output includes NGR metrics per pair. At least some pairs show NGR available.

- [ ] **Step 4: Commit**

```bash
git add evaluate_calibration.py
git commit -m "feat(evaluate_calibration): add NGR policy in holdout comparison"
```

---

## Task 15 — compute_position_size (quarter-Kelly)

**Files:**
- Modify: `src/paper_trading.py`
- Create: `tests/test_kelly_sizing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kelly_sizing.py
"""Tests for quarter-Kelly contract sizing."""

import pytest
from src.paper_trading import compute_position_size


def test_zero_edge_returns_zero_contracts():
    n = compute_position_size(
        edge=0.0, price=0.30, side="BUY",
        kelly_fraction=0.25, bankroll_dollars=100.0,
        max_order_cost_dollars=10.0, hard_cap_contracts=20,
    )
    assert n == 0


def test_negative_edge_buy_returns_zero():
    # Negative edge on BUY means market overpriced — don't buy.
    n = compute_position_size(
        edge=-0.05, price=0.30, side="BUY",
        kelly_fraction=0.25, bankroll_dollars=100.0,
        max_order_cost_dollars=10.0, hard_cap_contracts=20,
    )
    assert n == 0


def test_positive_edge_buy_sizes_up():
    # BUY @ 0.30 with our_prob = 0.50 (edge = 0.20)
    # kelly_full = (0.50 - 0.30) / (1 - 0.30) = 0.2857
    # stake = 0.25 * 100 * 0.2857 ~= $7.14
    # contracts at $0.30 = floor(7.14 / 0.30) = 23; capped by hard_cap=20
    n = compute_position_size(
        edge=0.20, price=0.30, side="BUY",
        kelly_fraction=0.25, bankroll_dollars=100.0,
        max_order_cost_dollars=100.0, hard_cap_contracts=20,
    )
    assert n == 20


def test_max_order_cost_caps_size():
    n = compute_position_size(
        edge=0.20, price=0.10, side="BUY",
        kelly_fraction=0.25, bankroll_dollars=1000.0,
        max_order_cost_dollars=10.0, hard_cap_contracts=1000,
    )
    # max 10.0 / 0.10 = 100 contracts max regardless of Kelly result
    assert n <= 100


def test_sell_side_uses_inverse_effective_price():
    # SELL at 0.80 means paying 0.20 for NO; our_prob=0.50 -> edge for NO is 0.30
    # kelly_full = (0.80 - 0.50) / 0.80 = 0.375
    # stake = 0.25 * 100 * 0.375 = $9.375
    # contracts at effective_price 0.20 = floor(9.375 / 0.20) = 46; capped by hard_cap=20
    n = compute_position_size(
        edge=-0.30, price=0.80, side="SELL",
        kelly_fraction=0.25, bankroll_dollars=100.0,
        max_order_cost_dollars=100.0, hard_cap_contracts=20,
    )
    assert n == 20


def test_extreme_price_guarded():
    # price=0 or price=1 should be handled
    n = compute_position_size(
        edge=0.5, price=0.0, side="BUY",
        kelly_fraction=0.25, bankroll_dollars=100.0,
        max_order_cost_dollars=10.0, hard_cap_contracts=20,
    )
    assert n == 0
    n = compute_position_size(
        edge=0.0, price=1.0, side="BUY",
        kelly_fraction=0.25, bankroll_dollars=100.0,
        max_order_cost_dollars=10.0, hard_cap_contracts=20,
    )
    assert n == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_kelly_sizing.py -v`
Expected: FAIL — `compute_position_size` not defined.

- [ ] **Step 3: Write implementation**

Append to `src/paper_trading.py`:

```python
import math


def compute_position_size(
    edge: float,
    price: float,
    side: str,
    kelly_fraction: float,
    bankroll_dollars: float,
    max_order_cost_dollars: float,
    hard_cap_contracts: int,
) -> int:
    """Quarter-Kelly contract sizing for binary YES/NO markets.

    side: 'BUY' (paying `price` for YES) or 'SELL' (paying 1-price for NO).
    edge = our_prob - market_price (same sign convention as matcher output).
    """
    if price <= 0.0 or price >= 1.0:
        return 0

    side = str(side).upper()
    if side == "BUY":
        if edge <= 0.0:
            return 0
        kelly_full = edge / (1.0 - price)
        effective_price = price
    elif side == "SELL":
        # For SELL, positive position is justified when edge < 0 (market overpriced).
        # Kelly on NO side: (price - our_prob) / price = -edge / price
        if edge >= 0.0:
            return 0
        kelly_full = (-edge) / price
        effective_price = 1.0 - price
    else:
        return 0

    kelly_full = max(0.0, min(1.0, kelly_full))  # never size over full Kelly
    stake_dollars = kelly_fraction * bankroll_dollars * kelly_full
    if stake_dollars <= 0.0:
        return 0

    by_stake = math.floor(stake_dollars / effective_price)
    by_order_cap = math.floor(max_order_cost_dollars / effective_price)
    contracts = min(by_stake, by_order_cap, hard_cap_contracts)
    return max(0, int(contracts))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_kelly_sizing.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/paper_trading.py tests/test_kelly_sizing.py
git commit -m "feat(paper_trading): add quarter-Kelly compute_position_size"
```

---

## Task 16 — Wire Kelly sizing into log_paper_trades

**Files:**
- Modify: `src/paper_trading.py:476+` (the `log_paper_trades` function)

- [ ] **Step 1: Update function signature and body**

Modify `log_paper_trades` to optionally compute sizing per trade:

```python
def log_paper_trades(
    opportunities: list[dict],
    *,
    scan_timestamp: str | datetime | None = None,
    ledger_path: str | Path | None = None,
    contracts: float = 1.0,  # kept for back-compat; ignored if sizing enabled in config
    config: Optional[dict] = None,
) -> dict:
    ...
    for opp in opportunities:
        side = _position_side(opp)
        if config and config.get("execution", {}).get("sizing") == "quarter_kelly":
            n = compute_position_size(
                edge=_coerce_float(opp.get("edge")) or 0.0,
                price=_coerce_float(opp.get("market_price")) or 0.0,
                side=opp.get("direction", side),
                kelly_fraction=float(config.get("kelly_fraction", 0.25)),
                bankroll_dollars=float(config.get("bankroll_dollars", 100.0)),
                max_order_cost_dollars=float(config.get("execution", {}).get("max_order_cost_dollars", 10.0)),
                hard_cap_contracts=int(config.get("max_contracts_hard_cap", 20)),
            )
            trade_contracts = float(n)
        else:
            trade_contracts = contracts

        if trade_contracts <= 0:
            continue
        # ... existing row-building code, but pass trade_contracts instead of contracts
```

(Read the actual function body in the current code first; adapt the patch so it fits the existing loop structure. The key change is: per-opp contracts is now computed from Kelly instead of always being 1.0.)

- [ ] **Step 2: Update the existing tests**

Existing tests pass `contracts=1.0` and don't set `execution.sizing`, so default behavior is unchanged. Add one new test proving sizing kicks in:

```python
# tests/test_paper_trading.py — append
def test_log_paper_trades_sizes_by_kelly_when_configured(tmp_path):
    ledger = tmp_path / "ledger.csv"
    opp = {
        "source": "kalshi", "ticker": "T1",
        "market_question": "q", "city": "Austin", "market_type": "high",
        "market_date": "2026-04-17", "outcome": ">75°F",
        "direction": "BUY",
        "our_probability": 0.50, "market_price": 0.30,
        "edge": 0.20, "abs_edge": 0.20,
        "forecast_value_f": 77.0, "uncertainty_std_f": 2.0,
        "settlement_rule": "gt", "settlement_low_f": 75.0, "settlement_high_f": None,
        "volume24hr": 5000, "yes_bid": 0.28, "yes_ask": 0.32,
    }
    config = {
        "execution": {"sizing": "quarter_kelly", "max_order_cost_dollars": 10.0},
        "kelly_fraction": 0.25,
        "bankroll_dollars": 100.0,
        "max_contracts_hard_cap": 20,
        "paper_trade_entry_fee_per_contract": 0.0,
        "paper_trade_settlement_fee_per_contract": 0.0,
    }
    result = log_paper_trades([opp], ledger_path=ledger, config=config)
    assert result["added"] == 1
    frame = pd.read_csv(ledger)
    assert frame.iloc[0]["contracts"] > 1.0  # Kelly sized up
```

- [ ] **Step 3: Run tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_paper_trading.py tests/test_kelly_sizing.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/paper_trading.py tests/test_paper_trading.py
git commit -m "feat(paper_trading): use Kelly sizing when execution.sizing=quarter_kelly"
```

---

## Task 17 — Policy v4 JSON + v3 backup

**Files:**
- Create: `strategy/strategy_policy_v3.json` (backup of current file)
- Modify: `strategy/strategy_policy.json` (v4 payload)

- [ ] **Step 1: Back up the existing policy**

```bash
cp strategy/strategy_policy.json strategy/strategy_policy_v3.json
```

- [ ] **Step 2: Write v4 payload**

Overwrite `strategy/strategy_policy.json` with:

```json
{
  "policy_version": 4,
  "status": "active",
  "generated_at_utc": "2026-04-16T18:00:00Z",
  "generated_by": "ngr_calibration_upgrade",
  "objective": "Take threshold trades with >= 8% calibrated edge, size by quarter-Kelly, expand candidate pool to accumulate paper data.",
  "selection": {
    "sources": ["kalshi"],
    "min_abs_edge": 0.08,
    "min_volume24hr": 2000,
    "max_candidates_per_scan": 8,
    "max_hours_to_settlement": 48,
    "allowed_market_types": ["high", "low"],
    "allowed_settlement_rules": ["lte", "gt"],
    "allowed_cities": [],
    "blocked_cities": ["Washington DC"]
  },
  "execution": {
    "sizing": "quarter_kelly",
    "max_contracts_per_trade": 20,
    "max_new_orders_per_day": 10,
    "max_order_cost_dollars": 10.0,
    "time_in_force": "fill_or_kill"
  },
  "rationale": {
    "v4_changes": "Lowered min_abs_edge 0.15 -> 0.08 now that NGR sigma is calibrated per city+regime. Widened lead horizon 24h -> 48h. Bumped max_candidates_per_scan 3 -> 8 and max_new_orders_per_day 3 -> 10 to accumulate paper data faster. Switched execution sizing to quarter-Kelly with hard cap 20 contracts."
  }
}
```

- [ ] **Step 3: Update strategy/CLAUDE.md documentation**

In `strategy/CLAUDE.md`, bump the documented version and thresholds to match. Add a note that v3 is saved as `strategy_policy_v3.json` for rollback.

- [ ] **Step 4: Commit**

```bash
git add strategy/strategy_policy.json strategy/strategy_policy_v3.json strategy/CLAUDE.md
git commit -m "feat(strategy): policy v4 — 8% edge, 48h window, quarter-Kelly sizing"
```

---

## Task 18 — Config keys

**Files:**
- Modify: `config.example.json`

- [ ] **Step 1: Add new keys**

Add these keys to `config.example.json` (keep existing structure, just insert near related keys):

```json
{
  "use_ngr_calibration": false,
  "ngr_min_training_rows": 20,
  "opportunity_archive_enabled": true,
  "opportunity_archive_dir": "data/opportunity_archive",
  "kelly_fraction": 0.25,
  "bankroll_dollars": 100.0,
  "max_contracts_hard_cap": 20
}
```

`use_ngr_calibration` defaults false — operator flips to true after Phase-2 validation. All other defaults are safe even when NGR and Kelly are off.

- [ ] **Step 2: Update README or AGENTS config section if one exists**

```bash
grep -n "use_ngr\|kelly_fraction\|opportunity_archive" AGENTS.md README.md config.example.json
```

If the README has a config-keys table, add rows for the new keys.

- [ ] **Step 3: Commit**

```bash
git add config.example.json README.md AGENTS.md
git commit -m "feat(config): add NGR, Kelly, and opportunity-archive config keys"
```

---

## Task 19 — Gitignore opportunity archive initially

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add archive directory to ignore**

Append to `.gitignore`:

```
# Opportunity archive — can grow and is not a long-term artifact like forecast_archive
data/opportunity_archive/
```

(Rationale: `forecast_archive/` is small and committed for calibration reproducibility, but `opportunity_archive/` can balloon and is regenerable from forecasts + actuals.)

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore opportunity_archive runtime dir"
```

---

## Task 20 — End-to-end smoke test

**Files:** none (verification only)

- [ ] **Step 1: Retrain models with NGR**

```
.\.venv\Scripts\python.exe train_calibration.py --days 365
```

Expected: Log reports `X NGR` models alongside EMOS + isotonic. `.pkl` files with `_ngr` suffix appear in `data/calibration_models/`.

- [ ] **Step 2: Run evaluation**

```
.\.venv\Scripts\python.exe evaluate_calibration.py --days 400 --holdout-days 30 --output data/evaluation_reports/eval_post_ngr.json
```

Expected: Report includes an NGR policy block with per-pair CRPS/Brier/MAE. At least 20 of 40 pairs should show NGR available (depends on training data).

- [ ] **Step 3: Run scanner with NGR OFF (safe default)**

Ensure `config.json` has `use_ngr_calibration: false`. Run:

```
.\.venv\Scripts\python.exe main.py --once
```

Expected: Scan runs; opportunity archive file appears for today; `forecast_calibration_source` column is `emos`/`raw`/`raw_selective_fallback` (legacy path).

- [ ] **Step 4: Flip NGR ON, re-run scanner**

Set `use_ngr_calibration: true` in `config.json`. Run:

```
.\.venv\Scripts\python.exe main.py --once
```

Expected: Same scan, but archive rows now show `forecast_calibration_source` = `ngr` for trained pairs. Kelly sizing active if policy v4 is in place — `contracts` column in ledger varies per trade (not all 1.0).

- [ ] **Step 5: Run full test suite**

```
.\.venv\Scripts\python.exe -m pytest tests -v
```

Expected: all existing tests still pass plus all new tests pass. Total should be ≥ 167 (current) + ~25 (new).

- [ ] **Step 6: Verify settlement**

```
.\.venv\Scripts\python.exe main.py --settle-paper-trades
```

Expected: Log includes `Opportunity archive: settled N rows`. Spot-check an archive CSV from 24h+ ago — rows have `yes_outcome` filled.

- [ ] **Step 7: Commit completion marker**

No code change. This task is the verification gate before handing back to operator for Phase-2 shadow observation window.

---

## Self-Review Summary

**Spec coverage:**
- NGR model ✓ Tasks 1–4
- CRPS fit ✓ Task 3
- σ²_floor empirical ✓ Task 3
- Selective fallback kept ✓ Task 7
- predict_distribution API ✓ Task 7
- Matcher integration (flag) ✓ Task 8
- Opportunity archive log + settle ✓ Tasks 9–10
- Main.py wire log + settle ✓ Tasks 11–12
- Main.py wire NGR flag ✓ Task 13
- Evaluation updates ✓ Task 14
- Kelly sizing ✓ Tasks 15–16
- Policy v4 ✓ Task 17
- Config keys ✓ Task 18
- Rollback path ✓ Task 17 (v3 backup) + Task 18 (flags default off)

**Known caveats for the implementer:**
- `main.py` wiring tasks (11, 12, 13) reference variable names like `scan_timestamp_iso` and `station_actuals_map` — the implementer must read the existing `main.py` and use the actual names. The patches are structural, not literal.
- Existing tests for `paper_trading.log_paper_trades` may need `config=` passed if they relied on defaults; adjust as needed.
- Kelly BUY / SELL sign convention matches the matcher's `edge = our_prob - market_price` and `direction ∈ {"BUY","SELL"}` — the test cases in Task 15 lock this in.
