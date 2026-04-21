# Rain Vertical P1 — Go-Live Log

## Status as of 2026-04-21

**Feature branch:** `feature/rain-vertical-p1`
**Plan:** [2026-04-20-rain-vertical-p1.md](2026-04-20-rain-vertical-p1.md)
**Spec:** [2026-04-20-rain-vertical-p1-design.md](../specs/2026-04-20-rain-vertical-p1-design.md)

**Smoke test:** passed (exit 0) via `scripts/smoke_test_rain_vertical.py`.
Full pipeline wired end-to-end: rain matcher loads, calibration manager
initializes, rain policy loads, category-aware filter produces rain-scoped
candidate list. Suite at 229 passing + 1 deselected (pre-existing unrelated).

## NYC-Only Scope

Per Task 6 discovery (2026-04-20), only New York has open KXRAIN markets
on Kalshi. `strategy/rain_policy_v1.json` sets `allowed_cities: ["New York"]`
and `min_volume24hr: 0` (NYC rain markets currently have blank yes_ask /
zero volume_24h).

## Blocker before live paper trading

`src/fetch_kalshi.py::fetch_weather_markets` calls `/markets/trades`, which
only returns recently-traded markets. The current NYC KXRAIN listings have
never traded, so they do not appear in the scanner's `kalshi_markets` input.
Rain vertical pipeline runs but currently sees zero rain markets — verified
in the 2026-04-21 smoke test log: `Rain: 0 opportunities with edge >= 15%`.

**Fix:** scan per-city via `/markets?series_ticker=KXRAINNYC&status=open`
(same shape as `scripts/dump_rain_market_inventory.py`). Spawned as a
follow-up task — landing that fix is a prerequisite for starting the 30-day
paper evaluation window.

## Go-Live Checklist (to fill in when live)

- [ ] Upstream Kalshi fetcher change landed and verified (per-city series
      scan returns KXRAINNYC markets)
- [ ] `enable_rain_vertical: true` set in local `config.json`
- [ ] `rain_bankroll_fraction: 0.20` set in local `config.json`
- [ ] Paper-trade ledger has at least one `market_category=rain` row
      recorded from a live scan (any outcome; just proof of end-to-end flow)
- [ ] `summary.json` shows a populated `category_breakdown.rain.trade_count`
      after the first settled rain trade
- [ ] Go-live UTC: <TBD when live>
- [ ] Commit at go-live: <TBD when live>

## 30-Day Evaluation Window

- Start: <go-live UTC>
- End: <go-live UTC + 30 days>
- Promotion gate to real money requires:
  1. Rain calibration log-loss strictly less than climatology on holdout
  2. Paper ROI ≥ 0 net of fees
  3. Pearson correlation of daily settled-PnL series (rain vs temperature) < 0.4
- Re-run `evaluate_rain_calibration.py --days 400 --holdout-days 30` at the
  end of the window for an up-to-date scorecard
