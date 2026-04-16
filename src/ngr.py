"""Non-homogeneous Gaussian Regression calibration.

Fits a predictive normal distribution whose mean and variance are both
functions of forecast features. Trained by minimizing the closed-form
Gaussian CRPS (Gneiting et al. 2005).
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
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
        # Guard against pathological mu drift (e.g. alpha corruption): reject
        # corrections more than 30F away from the raw forecast.
        if abs(mu - float(forecast_f)) > 30.0:
            mu = float(forecast_f)
        return mu, sigma

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
