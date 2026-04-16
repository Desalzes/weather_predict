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
