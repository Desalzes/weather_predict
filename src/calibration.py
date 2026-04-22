"""
Forecast calibration models for Weather Signals.

Implements:
- EMOS-style linear forecast correction per city/market type
- isotonic probability calibration over derived threshold/bucket examples
- model persistence and runtime loading helpers
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from src.station_truth import CALIBRATION_MODELS_DIR, _slugify_city
from src.ngr import NGRCalibrator

logger = logging.getLogger("weather.calibration")

_MIN_SPREAD_F = 1.0
_MIN_PROB = 0.001
_MAX_PROB = 0.999
SELECTIVE_RAW_FALLBACK_SOURCE = "raw_selective_fallback"
SELECTIVE_RAW_FALLBACK_TARGETS: tuple[tuple[str, str], ...] = (
    ("Boston", "low"),
    ("Minneapolis", "low"),
    ("Philadelphia", "low"),
    ("New Orleans", "high"),
    ("San Francisco", "low"),
)
_SELECTIVE_RAW_FALLBACK_TARGET_SET = frozenset(
    (" ".join(city.split()).casefold(), market_type.casefold())
    for city, market_type in SELECTIVE_RAW_FALLBACK_TARGETS
)


def configure_calibration_constants(
    min_spread_f: float | None = None,
    selective_raw_fallback_targets: list | None = None,
) -> None:
    """Override module-level calibration constants from config at startup."""
    global _MIN_SPREAD_F, SELECTIVE_RAW_FALLBACK_TARGETS, _SELECTIVE_RAW_FALLBACK_TARGET_SET
    if min_spread_f is not None:
        _MIN_SPREAD_F = float(min_spread_f)
    if selective_raw_fallback_targets is not None:
        SELECTIVE_RAW_FALLBACK_TARGETS = tuple(
            (str(pair[0]), str(pair[1])) for pair in selective_raw_fallback_targets
        )
        _SELECTIVE_RAW_FALLBACK_TARGET_SET = frozenset(
            (" ".join(city.split()).casefold(), market_type.casefold())
            for city, market_type in SELECTIVE_RAW_FALLBACK_TARGETS
        )


def _bucket_bounds_from_bucket_start(bucket_start: float) -> tuple[float, float]:
    """Return the continuous bounds for a Kalshi-style 2-degree bucket.

    A market like ``B66.5`` settles to the integer pair ``66-67``. During
    training we represent that same event using the lower bucket integer.
    """
    start = float(bucket_start)
    return start - 0.5, start + 1.5


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    z = (x - mu) / sigma
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _clip_probability(value: float) -> float:
    return max(_MIN_PROB, min(_MAX_PROB, float(value)))


def is_selective_raw_fallback_pair(city: str, market_type: str) -> bool:
    normalized_city = " ".join(str(city).split()).casefold()
    normalized_market_type = str(market_type).strip().casefold()
    return (normalized_city, normalized_market_type) in _SELECTIVE_RAW_FALLBACK_TARGET_SET


def compute_temperature_probability(
    forecast_value_f: float,
    outcome_low: Optional[float],
    outcome_high: Optional[float],
    uncertainty_std_f: float,
) -> float:
    """Compute P(actual temp falls in [low, high]) with a normal forecast model."""
    if outcome_low is not None and outcome_high is not None:
        prob = _normal_cdf(outcome_high, forecast_value_f, uncertainty_std_f) - _normal_cdf(
            outcome_low,
            forecast_value_f,
            uncertainty_std_f,
        )
    elif outcome_low is not None:
        prob = 1.0 - _normal_cdf(outcome_low, forecast_value_f, uncertainty_std_f)
    elif outcome_high is not None:
        prob = _normal_cdf(outcome_high, forecast_value_f, uncertainty_std_f)
    else:
        return 0.0

    return _clip_probability(prob)


def calibration_model_path(
    city: str,
    market_type: str,
    model_kind: str,
    model_dir: Optional[Path | str] = None,
) -> Path:
    directory = Path(model_dir) if model_dir is not None else CALIBRATION_MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slugify_city(city)}_{market_type}_{model_kind}.pkl"


def _market_columns(market_type: str) -> tuple[str, str, str]:
    if market_type not in {"high", "low"}:
        raise ValueError(f"Unsupported market_type: {market_type}")

    forecast_col = f"forecast_{market_type}_f"
    actual_col = f"actual_{market_type}_f"
    spread_col = f"ensemble_{market_type}_std_f"
    return forecast_col, actual_col, spread_col


def prepare_training_frame(df: pd.DataFrame, market_type: str) -> pd.DataFrame:
    """Normalize training data into forecast/actual/spread columns."""
    if {"forecast_f", "actual_f", "spread_f"}.issubset(df.columns):
        frame = df[["forecast_f", "actual_f", "spread_f"]].copy()
        frame["forecast_f"] = pd.to_numeric(frame["forecast_f"], errors="coerce")
        frame["actual_f"] = pd.to_numeric(frame["actual_f"], errors="coerce")
        frame["spread_f"] = pd.to_numeric(frame["spread_f"], errors="coerce")
        frame = frame.dropna(subset=["forecast_f", "actual_f"]).copy()
        if frame.empty:
            return pd.DataFrame(columns=["forecast_f", "actual_f", "spread_f"])
        spread_series = frame["spread_f"].dropna()
        fallback_spread = float(spread_series.median()) if not spread_series.empty else _MIN_SPREAD_F
        frame["spread_f"] = frame["spread_f"].fillna(fallback_spread).clip(lower=_MIN_SPREAD_F)
        return frame.reset_index(drop=True)

    forecast_col, actual_col, spread_col = _market_columns(market_type)
    if forecast_col not in df.columns or actual_col not in df.columns:
        return pd.DataFrame(columns=["forecast_f", "actual_f", "spread_f"])

    frame = df.copy()
    if spread_col not in frame.columns:
        spread_col = "ensemble_std_f"

    if spread_col not in frame.columns:
        frame["spread_f"] = _MIN_SPREAD_F
    else:
        frame["spread_f"] = pd.to_numeric(frame[spread_col], errors="coerce")

    frame["forecast_f"] = pd.to_numeric(frame[forecast_col], errors="coerce")
    frame["actual_f"] = pd.to_numeric(frame[actual_col], errors="coerce")
    frame = frame.dropna(subset=["forecast_f", "actual_f"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["forecast_f", "actual_f", "spread_f"])

    spread_series = frame["spread_f"].dropna()
    fallback_spread = float(spread_series.median()) if not spread_series.empty else _MIN_SPREAD_F
    frame["spread_f"] = frame["spread_f"].fillna(fallback_spread).clip(lower=_MIN_SPREAD_F)

    return frame[["forecast_f", "actual_f", "spread_f"]].reset_index(drop=True)


@dataclass
class EMOSCalibrator:
    """Linear bias corrector: actual = a + b*forecast + c*spread."""

    city: str
    market_type: str
    a: float = 0.0
    b: float = 1.0
    c: float = 0.0
    training_rows: int = 0
    is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "EMOSCalibrator":
        training = prepare_training_frame(df, self.market_type)
        if len(training) < 2:
            raise ValueError(
                f"Need at least 2 training rows for {self.city} {self.market_type} EMOS, got {len(training)}"
            )

        model = LinearRegression()
        x = training[["forecast_f", "spread_f"]].to_numpy(dtype=float)
        y = training["actual_f"].to_numpy(dtype=float)
        model.fit(x, y)

        self.a = float(model.intercept_)
        self.b = float(model.coef_[0])
        self.c = float(model.coef_[1]) if len(model.coef_) > 1 else 0.0
        self.training_rows = int(len(training))
        self.is_fitted = True
        return self

    def correct(self, forecast_f: float, spread_f: float) -> float:
        if not self.is_fitted:
            return float(forecast_f)
        return float(self.a + self.b * forecast_f + self.c * max(float(spread_f), _MIN_SPREAD_F))

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "EMOSCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)


@dataclass
class IsotonicCalibrator:
    """Maps raw probabilities to calibrated probabilities."""

    city: str
    market_type: str
    training_examples: int = 0
    is_fitted: bool = False
    _model: IsotonicRegression = field(
        default_factory=lambda: IsotonicRegression(out_of_bounds="clip"),
        repr=False,
    )

    def fit(self, raw_probs: np.ndarray, actual_outcomes: np.ndarray) -> "IsotonicCalibrator":
        probs = np.asarray(raw_probs, dtype=float).reshape(-1)
        outcomes = np.asarray(actual_outcomes, dtype=float).reshape(-1)
        if probs.size < 10:
            raise ValueError(
                f"Need at least 10 isotonic examples for {self.city} {self.market_type}, got {probs.size}"
            )
        if len(np.unique(outcomes)) < 2:
            raise ValueError(
                f"Need both positive and negative outcomes for {self.city} {self.market_type} isotonic fit"
            )

        self._model.fit(probs, outcomes)
        self.training_examples = int(probs.size)
        self.is_fitted = True
        return self

    def calibrate(self, raw_prob: float) -> float:
        if not self.is_fitted:
            return _clip_probability(raw_prob)
        calibrated = float(self._model.predict(np.asarray([raw_prob], dtype=float))[0])
        return _clip_probability(calibrated)

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self, f)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "IsotonicCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_isotonic_examples(
    df: pd.DataFrame,
    market_type: str,
    emos_model: Optional[EMOSCalibrator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive yes/no calibration examples from historical forecast/actual pairs."""
    training = prepare_training_frame(df, market_type)
    if training.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    raw_probs: list[float] = []
    outcomes: list[float] = []

    for row in training.itertuples(index=False):
        corrected = emos_model.correct(row.forecast_f, row.spread_f) if emos_model else float(row.forecast_f)
        spread_f = max(float(row.spread_f), _MIN_SPREAD_F)
        actual_f = float(row.actual_f)

        radius = max(4.0, spread_f * 2.0)
        lower = int(math.floor(min(corrected, actual_f) - radius))
        upper = int(math.ceil(max(corrected, actual_f) + radius))

        for threshold in range(lower, upper + 1, 2):
            if market_type == "high":
                over_prob = _clip_probability(1.0 - _normal_cdf(threshold, corrected, spread_f))
                over_outcome = 1.0 if actual_f >= threshold else 0.0
            else:
                over_prob = _clip_probability(_normal_cdf(threshold + 0.99, corrected, spread_f))
                over_outcome = 1.0 if actual_f <= threshold else 0.0

            raw_probs.append(over_prob)
            outcomes.append(over_outcome)

            bucket_low, bucket_high = _bucket_bounds_from_bucket_start(threshold)
            bucket_prob = compute_temperature_probability(corrected, bucket_low, bucket_high, spread_f)
            bucket_outcome = 1.0 if bucket_low <= actual_f <= bucket_high else 0.0
            raw_probs.append(bucket_prob)
            outcomes.append(bucket_outcome)

    return np.asarray(raw_probs, dtype=float), np.asarray(outcomes, dtype=float)


def train_city_models(
    city: str,
    training_df: pd.DataFrame,
    *,
    model_dir: Optional[Path | str] = None,
    min_training_rows: int = 10,
) -> dict[str, dict]:
    """Train and persist EMOS/isotonic models for one city."""
    results: dict[str, dict] = {}

    for market_type in ("high", "low"):
        prepared = prepare_training_frame(training_df, market_type)
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

        if len(prepared) < min_training_rows:
            outcome["reason"] = f"need at least {min_training_rows} overlapping rows"
            results[market_type] = outcome
            continue

        emos = EMOSCalibrator(city=city, market_type=market_type).fit(prepared)
        emos_path = calibration_model_path(city, market_type, "emos", model_dir=model_dir)
        emos.save(emos_path)
        outcome["trained_emos"] = True
        outcome["emos_path"] = str(emos_path)

        raw_probs, actual_outcomes = build_isotonic_examples(prepared, market_type, emos_model=emos)
        if raw_probs.size >= 10 and len(np.unique(actual_outcomes)) >= 2:
            isotonic = IsotonicCalibrator(city=city, market_type=market_type).fit(raw_probs, actual_outcomes)
            isotonic_path = calibration_model_path(city, market_type, "isotonic", model_dir=model_dir)
            isotonic.save(isotonic_path)
            outcome["trained_isotonic"] = True
            outcome["isotonic_path"] = str(isotonic_path)
            outcome["isotonic_examples"] = int(raw_probs.size)
            outcome["status"] = "trained"
        else:
            outcome["reason"] = "insufficient isotonic example diversity"
            outcome["status"] = "trained_emos_only"

        # NGR uses the full training_df (needs date + lead columns)
        try:
            ngr_calibrator = NGRCalibrator(city=city, market_type=market_type).fit(
                training_df, min_rows=20
            )
            ngr_path = calibration_model_path(city, market_type, "ngr", model_dir=model_dir)
            ngr_calibrator.save(ngr_path)
            outcome["trained_ngr"] = True
            outcome["ngr_path"] = str(ngr_path)
            outcome["ngr_training_crps"] = float(ngr_calibrator.training_crps)
        except Exception as exc:
            logger.warning("NGR fit skipped for %s %s: %s", city, market_type, exc)

        results[market_type] = outcome

    return results


class CalibrationManager:
    """Load calibration models on demand and apply them in matcher flows."""

    def __init__(self, model_dir: Optional[Path | str] = None):
        self.model_dir = Path(model_dir) if model_dir is not None else CALIBRATION_MODELS_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Each cache entry is (mtime_or_None, model_or_None). mtime invalidation
        # catches retrained pkls (autopilot_weekly rewrites models in place) and
        # newly-appeared pkls after a cold start that cached None.
        self._emos_cache: dict[tuple[str, str], tuple[Optional[float], Optional[EMOSCalibrator]]] = {}
        self._isotonic_cache: dict[tuple[str, str], tuple[Optional[float], Optional[IsotonicCalibrator]]] = {}
        self._ngr_cache: dict[tuple[str, str], tuple[Optional[float], Optional[NGRCalibrator]]] = {}

    @staticmethod
    def _path_mtime(path: Path) -> Optional[float]:
        try:
            return path.stat().st_mtime if path.exists() else None
        except OSError:
            return None

    def _load_emos(self, city: str, market_type: str) -> Optional[EMOSCalibrator]:
        key = (city, market_type)
        path = calibration_model_path(city, market_type, "emos", model_dir=self.model_dir)
        current_mtime = self._path_mtime(path)
        cached = self._emos_cache.get(key)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]

        if current_mtime is None:
            self._emos_cache[key] = (None, None)
            return None

        try:
            model = EMOSCalibrator.load(path)
        except Exception as exc:
            logger.warning("Failed loading EMOS model %s: %s", path, exc)
            model = None

        self._emos_cache[key] = (current_mtime, model)
        return model

    def _load_isotonic(self, city: str, market_type: str) -> Optional[IsotonicCalibrator]:
        key = (city, market_type)
        path = calibration_model_path(city, market_type, "isotonic", model_dir=self.model_dir)
        current_mtime = self._path_mtime(path)
        cached = self._isotonic_cache.get(key)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]

        if current_mtime is None:
            self._isotonic_cache[key] = (None, None)
            return None

        try:
            model = IsotonicCalibrator.load(path)
        except Exception as exc:
            logger.warning("Failed loading isotonic model %s: %s", path, exc)
            model = None

        self._isotonic_cache[key] = (current_mtime, model)
        return model

    def _load_ngr(self, city: str, market_type: str) -> Optional[NGRCalibrator]:
        key = (city, market_type)
        path = calibration_model_path(city, market_type, "ngr", model_dir=self.model_dir)
        current_mtime = self._path_mtime(path)
        cached = self._ngr_cache.get(key)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]

        if current_mtime is None:
            self._ngr_cache[key] = (None, None)
            return None

        try:
            model = NGRCalibrator.load(path)
        except Exception as exc:
            logger.warning("Failed loading NGR model %s: %s", path, exc)
            model = None

        self._ngr_cache[key] = (current_mtime, model)
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
        """Return (mu, sigma, source) for probability computation.

        Priority: selective_raw_fallback + NGR σ → NGR → selective + EMOS →
        EMOS → raw. Selective-fallback pairs keep raw mu but take NGR σ when
        available so uncertainty is still calibrated.
        """
        selective = is_selective_raw_fallback_pair(city, market_type)

        ngr_model = self._load_ngr(city, market_type)
        if ngr_model is not None and ngr_model.is_fitted:
            mu_ngr, sigma_ngr = ngr_model.predict(forecast_f, spread_f, lead_h, doy)
            if selective:
                return float(forecast_f), float(sigma_ngr), SELECTIVE_RAW_FALLBACK_SOURCE
            return float(mu_ngr), float(sigma_ngr), "ngr"

        emos_model = self._load_emos(city, market_type)
        if emos_model is not None and emos_model.is_fitted:
            if selective:
                return float(forecast_f), max(float(spread_f), 1.0), SELECTIVE_RAW_FALLBACK_SOURCE
            return float(emos_model.correct(forecast_f, spread_f)), max(float(spread_f), 1.0), "emos"

        return float(forecast_f), max(float(spread_f), 1.0), "raw"

    def correct_forecast(
        self,
        city: str,
        market_type: str,
        forecast_f: float,
        spread_f: float,
    ) -> tuple[float, str]:
        model = self._load_emos(city, market_type)
        if not model or not model.is_fitted:
            return float(forecast_f), "raw"
        if is_selective_raw_fallback_pair(city, market_type):
            return float(forecast_f), SELECTIVE_RAW_FALLBACK_SOURCE
        return float(model.correct(forecast_f, spread_f)), "emos"

    def calibrate_probability(
        self,
        city: str,
        market_type: str,
        raw_prob: float,
    ) -> tuple[float, str]:
        model = self._load_isotonic(city, market_type)
        if not model or not model.is_fitted:
            return _clip_probability(raw_prob), "raw"
        return model.calibrate(raw_prob), "isotonic"

    def available_model_count(self) -> int:
        return len(list(self.model_dir.glob("*.pkl")))
