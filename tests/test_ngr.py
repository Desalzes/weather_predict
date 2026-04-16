"""Tests for Non-homogeneous Gaussian Regression."""

import math
import numpy as np
import pytest

from src.ngr import gaussian_crps


def test_gaussian_crps_closed_form_zero_at_perfect_forecast():
    assert gaussian_crps(mu=10.0, sigma=1e-6, y=10.0) == pytest.approx(0.0, abs=1e-3)


def test_gaussian_crps_matches_known_value():
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
