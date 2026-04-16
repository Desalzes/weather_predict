"""Tests for quarter-Kelly contract sizing (src.sizing)."""
from __future__ import annotations

import pytest

from src.sizing import compute_position_size


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
    # price=0 or price=1 should be handled gracefully
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
