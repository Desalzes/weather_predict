# Rain Vertical Rules

## Scope

P1 ships KXRAIN binary any-rain markets (≥ 0.01 in threshold) only.
Moderate thresholds (≥ 0.25 in) and heavy thresholds (≥ 1.0 in) are
P1.5 or later — `src/rain_matcher.py`'s parser explicitly returns None
for any non-`-T0` ticker suffix.

## NYC-Only Scope for P1

The 2026-04-20 discovery pass (`data/rain_market_inventory.csv`) found that
only New York has active KXRAIN markets on Kalshi. The other watchlist
cities (Chicago, Miami, Houston, Seattle, Philadelphia, Boston, Atlanta,
Washington DC, New Orleans) have zero open markets. The rain policy
(`strategy/rain_policy_v1.json`) restricts `allowed_cities` to `["New York"]`
and sets `min_volume24hr: 0` because the listed NYC markets have blank
yes_ask and zero volume_24h. Calibration models for all 20 cities are still
trained and stored so that any future expansion of Kalshi's market list is
immediately supported.

## Routing

Rain opportunities carry `market_category="rain"` and `direction="BUY"`.
`src/strategy_policy.py` filters by category when a policy declares
`market_category: "rain"`. Legacy (temperature) opportunities without
`market_category` default to `"temperature"`. The category filter runs
before the other selection filters so candidate caps see the correct
per-category universe.

## Calibration

Two-stage: `LogisticRainCalibrator` (bias correction) +
`IsotonicRainCalibrator` (probability recalibration). Models live at
`data/calibration_models/{city_slug}_rain_binary_{logistic|isotonic}.pkl`
and are trained by `train_rain_calibration.py` against a rolling
`rain_calibration_window_days` (default 90) of archived forecast-vs-actual
pairs.

Baseline scorecard (2026-04-20 via
`evaluate_rain_calibration.py --days 400 --holdout-days 30`):
- 18/20 cities beat the climatology baseline on Brier
- New York (P1 target): 0.1137 calibrated vs 0.1983 climatology, ~43% improvement
- Phoenix and Las Vegas do not beat climatology; expected for dry climates
  where near-zero climatology is hard to improve on

## HRRR Blend

Same-day rain markets (hours_to_settlement < `rain_hrrr_blend_horizon_hours`,
default 12) blend HRRR APCP into the calibrated probability. Linear ramp
weight, cap 0.7. Note: HRRR-precip wiring is deliberately NOT active in
`main.py` yet (Task 12 wires `hrrr_data=None` into `match_kalshi_rain`).
Live HRRR-precip integration is a future task.

## Bankroll

Rain trades draw from a separate 20% slice of the Kelly bankroll
(`rain_bankroll_fraction`). Temperature Kelly is untouched. If realized
30-day daily-PnL Pearson correlation between rain and temperature exceeds
0.4, manually dial `rain_bankroll_fraction` down — `summary.json`'s
`category_breakdown` block carries the correlation figure.

## Promotion Gate

Paper-only for at least 30 days. Criteria to begin real-money discussion:
1. Rain calibration log-loss strictly less than climatology baseline on holdout
2. Paper ROI ≥ 0 net of fees
3. Pearson correlation of daily settled-PnL series (rain vs temperature) < 0.4

## Known Scanner-Side Issue

`src/fetch_kalshi.py::fetch_weather_markets` uses the `/markets/trades`
endpoint, which only surfaces *recently-traded* markets. Untraded KXRAIN
listings therefore slip through the default scanner's weather-prefix filter.
For rain-vertical paper trading to fire, the scanner would need to fetch
`/markets?series_ticker={X}` per-city — see `scripts/dump_rain_market_inventory.py`
for the correct endpoint shape. Task 12's wiring still uses the existing
`kalshi_markets` from the trades endpoint, so live rain-opportunity flow is
blocked until this is fixed.
