"""Tail calibration for temperature markets.

Composes P1's LogisticRainCalibrator + IsotonicRainCalibrator into a two-stage
chain specialized per (city, market_type, direction). Model pickles live in
data/calibration_models/ with distinct naming (suffix `_tail_`) so the existing
EMOS / NGR / isotonic stack stays operable and comparable.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from src.rain_calibration import (
    LogisticRainCalibrator,
    IsotonicRainCalibrator,
    _clip,
)
from src.station_truth import CALIBRATION_MODELS_DIR, _slugify_city

logger = logging.getLogger("weather.tail_calibration")


# ========================================================================
# Threshold-tail calibration (per-direction high/low markets)
# ========================================================================


def _tail_model_path(
    city: str,
    market_type: str,
    direction: str,
    kind: str,
    model_dir: Optional[Path] = None,
) -> Path:
    """Per-pair tail model path.

    kind is one of "logistic" or "isotonic". Returned path is
    {model_dir}/{slug}_{market_type}_{direction}_tail_{kind}.pkl.
    """
    directory = Path(model_dir) if model_dir is not None else CALIBRATION_MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slugify_city(city)}_{market_type}_{direction}_tail_{kind}.pkl"


class TailBinaryCalibrator:
    """Two-stage tail calibration: logistic bias correction + isotonic
    probability recalibration. Per (city, market_type, direction).

    Save format: three sidecar pickle files at a single path prefix:
      {prefix}_logistic.pkl  — {"city": str, "model": LogisticRegression | None}
      {prefix}_isotonic.pkl  — {"city": str, "model": IsotonicRegression | None}
      {prefix}_meta.pkl      — {"city": str, "market_type": str, "direction": str}
    """

    def __init__(self, city: str, market_type: str, direction: str):
        self.city = city
        self.market_type = market_type
        self.direction = direction
        self._logistic = LogisticRainCalibrator(city=city)
        self._isotonic = IsotonicRainCalibrator(city=city)

    def fit(self, raw_probs, outcomes) -> None:
        import numpy as np
        self._logistic.fit(raw_probs, outcomes)
        raw_arr = np.asarray(raw_probs, dtype=float)
        logistic_preds = np.array([self._logistic.predict(p) for p in raw_arr])
        self._isotonic.fit(logistic_preds, outcomes)

    def predict(self, raw_prob: float) -> float:
        p = _clip(raw_prob)
        p = self._logistic.predict(p)
        p = self._isotonic.predict(p)
        return _clip(p)

    def save(self, path_prefix) -> None:
        """Save both stages plus metadata sidecar to disk.

        path_prefix is a path without extension; three sidecar files are
        created alongside it.
        """
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

    @classmethod
    def load(cls, path_prefix) -> "TailBinaryCalibrator":
        """Reconstruct a TailBinaryCalibrator from sidecar pickles."""
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
            logger.warning(
                "Tail calibration meta missing at %s; synthesizing "
                "market_type/direction as 'unknown'. This model will not match "
                "TailCalibrationManager key lookups once Task 4 lands.",
                meta_path,
            )
            meta = {
                "city": logistic_data.get("city", "unknown"),
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


# ========================================================================
# Bucket calibration (P(temp in [lo, hi]) for narrow bucket markets)
# ========================================================================


def _bucket_model_path(
    city: str,
    market_type: str,
    kind: str,
    model_dir: Optional[Path] = None,
) -> Path:
    """Per-pair bucket model path.

    kind is one of "logistic" or "isotonic". Returned path is
    {model_dir}/{slug}_{market_type}_bucket_{kind}.pkl.
    """
    directory = Path(model_dir) if model_dir is not None else CALIBRATION_MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slugify_city(city)}_{market_type}_bucket_{kind}.pkl"


class BucketDistributionalCalibrator:
    """Two-stage calibration for bucket market probabilities. Per
    (city, market_type) — no direction dimension since buckets are
    inherently two-sided (the outcome is "temp in [lo, hi]" vs not).

    Save format mirrors TailBinaryCalibrator: three sidecar pickles at a
    single path prefix:
      {prefix}_logistic.pkl — {"city": str, "model": LogisticRegression | None}
      {prefix}_isotonic.pkl — {"city": str, "model": IsotonicRegression | None}
      {prefix}_meta.pkl     — {"city": str, "market_type": str}
    """

    def __init__(self, city: str, market_type: str):
        self.city = city
        self.market_type = market_type
        self._logistic = LogisticRainCalibrator(city=city)
        self._isotonic = IsotonicRainCalibrator(city=city)

    def fit(self, raw_bucket_probs, outcomes) -> None:
        import numpy as np
        self._logistic.fit(raw_bucket_probs, outcomes)
        raw_arr = np.asarray(raw_bucket_probs, dtype=float)
        logistic_preds = np.array([self._logistic.predict(p) for p in raw_arr])
        self._isotonic.fit(logistic_preds, outcomes)

    def predict(self, raw_bucket_prob: float) -> float:
        p = _clip(raw_bucket_prob)
        p = self._logistic.predict(p)
        p = self._isotonic.predict(p)
        return _clip(p)

    def save(self, path_prefix) -> None:
        """Save both stages plus metadata sidecar."""
        prefix = Path(path_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self._logistic.save(Path(f"{prefix}_logistic.pkl"))
        self._isotonic.save(Path(f"{prefix}_isotonic.pkl"))
        meta = {"city": self.city, "market_type": self.market_type}
        with Path(f"{prefix}_meta.pkl").open("wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path_prefix) -> "BucketDistributionalCalibrator":
        """Reconstruct from sidecar pickles."""
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
            logger.warning(
                "Bucket calibration meta missing at %s; synthesizing "
                "market_type='unknown'. This model will not match "
                "TailCalibrationManager key lookups once Task 4 lands.",
                meta_path,
            )
            meta = {
                "city": logistic_data.get("city", "unknown"),
                "market_type": "unknown",
            }
        obj = cls(city=meta["city"], market_type=meta["market_type"])
        obj._logistic._model = logistic_data["model"]
        obj._isotonic._model = isotonic_data["model"]
        return obj


# ========================================================================
# Runtime manager (mtime-invalidating cache for both calibrator types)
# ========================================================================


class TailCalibrationManager:
    """Per-pair loader + cache for tail and bucket calibration models.
    mtime-based invalidation so retrains on disk are picked up without
    process restart.

    Entry point: calibrate_tail_probability(city, market_type, direction,
    is_bucket, raw_prob) -> {"calibrated_prob", "source"} | None.
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
        """Latest mtime across the three sidecar files for a prefix.
        Returns None if no sidecar files exist.
        """
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
        """Returns {"calibrated_prob", "source"} or None if no model exists.

        For is_bucket=True the direction arg is ignored.
        """
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
