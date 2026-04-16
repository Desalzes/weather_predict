"""Non-homogeneous Gaussian Regression calibration.

Fits a predictive normal distribution whose mean and variance are both
functions of forecast features. Trained by minimizing the closed-form
Gaussian CRPS (Gneiting et al. 2005).
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm
import pandas as pd
from dataclasses import dataclass, field


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
