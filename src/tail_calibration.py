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
            # Fallback for pre-meta saves — unknown market_type/direction
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
