"""Position-sizing helper for weather-market betting.

Provides a single entry point, ``compute_position_size``, that translates a
probability edge and market price into a whole-number contract count using a
fractional-Kelly criterion.  The module is intentionally free of side effects
so that any consumer (paper trading, live execution, back-testing) can import
it without pulling in API or I/O dependencies.

Kelly convention used here
--------------------------
``edge = our_probability - market_price``

* BUY  — positive edge means the market underprices the YES leg.
* SELL — negative edge means the market overprices the YES leg
          (equivalently, the NO leg is underpriced at ``1 - price``).

The fractional-Kelly stake is computed as::

    kelly_full  = |edge| / (1 - effective_price)   # for BUY
                = |edge| / price                    # for SELL
    stake       = kelly_fraction * bankroll * clamp(kelly_full, 0, 1)
    contracts   = floor(stake / effective_price)

The result is then capped by ``max_order_cost_dollars / effective_price`` and
``hard_cap_contracts`` before being returned.
"""
from __future__ import annotations

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
    """Return the number of contracts to trade under a fractional-Kelly criterion.

    Parameters
    ----------
    edge:
        ``our_probability - market_price``.  Positive means we think YES is
        underpriced; negative means YES is overpriced (NO is underpriced).
    price:
        Market price of the YES contract, in [0, 1].
    side:
        ``"BUY"`` (buying YES) or ``"SELL"`` (selling YES / buying NO).
    kelly_fraction:
        Fraction of full Kelly to stake (e.g. 0.25 for quarter-Kelly).
    bankroll_dollars:
        Total bankroll available for sizing.
    max_order_cost_dollars:
        Hard cap on the dollar cost of a single order.
    hard_cap_contracts:
        Hard cap on the number of contracts regardless of other limits.

    Returns
    -------
    int
        Number of contracts (>= 0).  Returns 0 for zero or wrong-sign edge,
        degenerate prices, or when any limit resolves to zero.
    """
    # Guard degenerate prices that cause division by zero or nonsensical Kelly.
    if price <= 0.0 or price >= 1.0:
        return 0

    if side == "BUY":
        if edge <= 0.0:
            return 0
        kelly_full = edge / (1.0 - price)
        effective_price = price
    elif side == "SELL":
        if edge >= 0.0:
            return 0
        kelly_full = (-edge) / price
        effective_price = 1.0 - price
    else:
        return 0

    # Clamp kelly_full to [0, 1] — never bet more than the full-Kelly fraction.
    kelly_full = max(0.0, min(1.0, kelly_full))

    stake_dollars = kelly_fraction * bankroll_dollars * kelly_full
    if stake_dollars <= 0.0:
        return 0

    by_stake = math.floor(stake_dollars / effective_price)
    by_order_cap = math.floor(max_order_cost_dollars / effective_price)
    contracts = min(by_stake, by_order_cap, hard_cap_contracts)
    return max(0, int(contracts))
