"""Rain calibration: logistic bias correction + isotonic probability recalibration.

Designed as a generic binary-outcome calibration stack that P2 can reuse
for temperature tails (e.g., "high > climate mean + 2 sigma"). The rain use
case feeds it raw Open-Meteo rain probability + actual wet-day outcomes.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.station_truth import CALIBRATION_MODELS_DIR, _slugify_city

logger = logging.getLogger("weather.rain_calibration")

_MIN_PROB = 0.001
_MAX_PROB = 0.999


def _clip(p: float) -> float:
    return max(_MIN_PROB, min(_MAX_PROB, float(p)))


class LogisticRainCalibrator:
    """Per-city logistic bias correction for binary rain outcomes."""

    def __init__(self, city: str):
        self.city = city
        self._model: Optional[LogisticRegression] = None

    def fit(self, raw_probs, outcomes) -> None:
        x = np.asarray(raw_probs, dtype=float).reshape(-1, 1)
        y = np.asarray(outcomes, dtype=int).reshape(-1)
        if len(np.unique(y)) < 2:
            # Degenerate - caller's data has no contrast; refuse to fit
            self._model = None
            return
        self._model = LogisticRegression().fit(x, y)

    def predict(self, raw_prob: float) -> float:
        if self._model is None:
            return _clip(raw_prob)
        prob = float(self._model.predict_proba(np.array([[float(raw_prob)]]))[0, 1])
        return _clip(prob)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"city": self.city, "model": self._model}, f)

    @classmethod
    def load(cls, path) -> "LogisticRainCalibrator":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        obj = cls(city=data["city"])
        obj._model = data["model"]
        return obj


class IsotonicRainCalibrator:
    """Per-city isotonic probability recalibrator."""

    def __init__(self, city: str):
        self.city = city
        self._model: Optional[IsotonicRegression] = None

    def fit(self, predicted_probs, outcomes) -> None:
        x = np.asarray(predicted_probs, dtype=float).reshape(-1)
        y = np.asarray(outcomes, dtype=int).reshape(-1)
        if len(np.unique(y)) < 2:
            self._model = None
            return
        self._model = IsotonicRegression(out_of_bounds="clip").fit(x, y)

    def predict(self, predicted_prob: float) -> float:
        if self._model is None:
            return _clip(predicted_prob)
        return _clip(float(self._model.predict([float(predicted_prob)])[0]))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"city": self.city, "model": self._model}, f)

    @classmethod
    def load(cls, path) -> "IsotonicRainCalibrator":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        obj = cls(city=data["city"])
        obj._model = data["model"]
        return obj


# SECURITY: pickle.load trusts any file in model_dir. Never expose
# RainCalibrationManager(model_dir=...) to untrusted input (CLI flags,
# config values influenced by users). model_dir is internal-only.
def _rain_model_path(city: str, kind: str, model_dir=None) -> Path:
    """Return the on-disk path for a rain calibration model.

    kind is one of "logistic" or "isotonic".
    """
    directory = Path(model_dir) if model_dir is not None else CALIBRATION_MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slugify_city(city)}_rain_binary_{kind}.pkl"


class RainCalibrationManager:
    """Load and apply rain calibration models at scan time.

    Exposes the `calibrate_rain_probability(city, raw_prob)` entrypoint
    that `src/rain_matcher.py:match_kalshi_rain` consumes.
    """

    def __init__(self, model_dir=None):
        self._model_dir = Path(model_dir) if model_dir else CALIBRATION_MODELS_DIR
        # Each cache entry is (mtime_or_None, model_or_None). The mtime lets
        # us detect pkl files rewritten by autopilot_weekly or freshly-appeared
        # models after a cold start that cached None.
        self._logistic_cache: dict[str, tuple[Optional[float], Optional[LogisticRainCalibrator]]] = {}
        self._isotonic_cache: dict[str, tuple[Optional[float], Optional[IsotonicRainCalibrator]]] = {}

    @staticmethod
    def _path_mtime(path: Path) -> Optional[float]:
        try:
            return path.stat().st_mtime if path.exists() else None
        except OSError:
            return None

    def _logistic(self, city: str) -> Optional[LogisticRainCalibrator]:
        path = _rain_model_path(city, "logistic", self._model_dir)
        current_mtime = self._path_mtime(path)
        cached = self._logistic_cache.get(city)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]
        model = LogisticRainCalibrator.load(path) if current_mtime is not None else None
        self._logistic_cache[city] = (current_mtime, model)
        return model

    def _isotonic(self, city: str) -> Optional[IsotonicRainCalibrator]:
        path = _rain_model_path(city, "isotonic", self._model_dir)
        current_mtime = self._path_mtime(path)
        cached = self._isotonic_cache.get(city)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]
        model = IsotonicRainCalibrator.load(path) if current_mtime is not None else None
        self._isotonic_cache[city] = (current_mtime, model)
        return model

    def calibrate_rain_probability(
        self, city: str, raw_prob: float
    ) -> Optional[dict]:
        """Apply logistic -> isotonic chain. Returns None when neither model
        exists for this city (caller falls through to raw)."""
        logistic = self._logistic(city)
        isotonic = self._isotonic(city)
        if logistic is None and isotonic is None:
            return None

        p = _clip(raw_prob)
        forecast_src = "raw"
        prob_src = "raw"
        if logistic is not None:
            p = logistic.predict(p)
            forecast_src = "logistic"
        if isotonic is not None:
            p = isotonic.predict(p)
            prob_src = "isotonic"
        return {
            "calibrated_prob": p,
            "forecast_calibration_source": forecast_src,
            "probability_calibration_source": prob_src,
        }
