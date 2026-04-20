# Rain Vertical — Phase 1 Design

**Date:** 2026-04-20
**Status:** Approved design, ready for implementation planning
**Author:** Claude (brainstorming skill) with Gabriel

## Context

The project currently trades Kalshi temperature threshold markets only. Policy v4
(2026-04-20) is BUY-only after SELL-side produced −50.6% ROI on 35 trades, and
blocks bucket markets, Washington DC, and Polymarket. The modeling stack is
well-developed (Open-Meteo ensemble → EMOS → NGR → isotonic → HRRR blend) but
is applied exclusively to temperature.

This spec defines **Phase 1 of a three-phase program** that expands the project
into new territory in a staged, diversification-first order:

- **P1 (this spec):** Rain vertical — a parallel pipeline that trades Kalshi
  `KXRAIN` markets with a separate calibration stack, separate policy, and its
  own bankroll slice. Paper-only for 30 days.
- **P2 (future):** Port tail-calibration techniques developed for rare-rain
  events back to temperature → rehab SELL-side and bucket markets.
- **P3 (future):** Multi-model ensemble upgrade (ECMWF + GFS + ICON stacked
  beside Open-Meteo's blended output). Benefits both verticals.

P1 is designed so P2 and P3 are drop-ins, not rewrites.

## Goals

1. Add a trading vertical whose P&L is statistically uncorrelated with the
   current temperature book, so a bad temperature week does not compound.
2. Build the tail-calibration tools that P2 needs, in an uncorrelated sandbox
   where mistakes don't threaten the working temperature strategy.
3. Keep blast radius small: paper-only for 30 days, separate bankroll slice,
   category-aware telemetry.

## Non-Goals

- Snow markets, wind markets, multi-day composites — out of scope for P1.
- Multi-model ensemble stacking — P3 concern.
- SELL-side or bucket-market rehabilitation on temperature — P2 concern.
- Enabling Polymarket — unchanged from current state.
- Any real-money execution before the 30-day paper evaluation gate passes.

## Architecture

### Reused Without Modification

| Module | Role |
|---|---|
| `src/kalshi_client.py` | Authenticated REST client |
| `src/fetch_kalshi.py` | Already includes `KXRAIN` in `_WEATHER_PREFIXES` |
| `src/station_truth.py` | Already parses `precip_in` and `precip_trace` from NWS CLI |
| `src/kalshi_client.py`, `src/logging_setup.py`, `src/config.py` | Infra |
| `src/deepseek_worker.py` | Optional gate, consumes policy JSON; category-agnostic |
| `src/sizing.py` | Quarter-Kelly calculation, applied per-bankroll-slice |
| `src/opportunity_log.py` | Per-scan opportunity archive |

### Modified

| Module | Change |
|---|---|
| `src/paper_trading.py` | Add `market_category` column to ledger (values: `temperature`, `rain`). Backfill legacy rows as `temperature`. Extend `summary.json` with a `category_breakdown` block (per-category PnL / ROI / trade count / realized correlation over trailing 30 days). |
| `src/strategy_policy.py` | Category-aware filter: a policy file carries an implicit or explicit `market_category` scope; filtering routes opportunities to the matching policy. |
| `main.py` | After the existing temperature `match_*` passes, run an analogous rain matcher pass when `enable_rain_vertical` is `true`. Rain opportunities flow through the rain policy and are logged alongside temperature trades with `market_category=rain`. |
| `config.example.json` | Add rain-vertical keys (all default off). |
| `data/CLAUDE.md`, `strategy/CLAUDE.md`, `src/CLAUDE.md`, `.claude/rules/` | Document the new vertical per the repo's docs conventions. |

### New Modules

| Module | Purpose |
|---|---|
| `src/fetch_precipitation.py` | Open-Meteo precipitation fetch (deterministic + ensemble) and HRRR APCP point extraction at station coordinates. Mirrors `fetch_forecasts.py` + `fetch_hrrr.py` patterns. Accepts a `models` parameter (default: Open-Meteo blend) so P3 can pass an explicit model list. |
| `src/rain_matcher.py` | Parse `KXRAIN` ticker / outcome strings, compute per-contract probability from calibrated rain probability, return opportunity dicts with the same shape as `matcher.py` opportunities. |
| `src/rain_calibration.py` | `LogisticRainCalibrator` (bias correction) + `IsotonicRainCalibrator` (probability recalibration). Same save/load pattern as `src/calibration.py`. Interface deliberately generic over any binary-outcome probability input, so P2 can reuse it for temperature tails. |
| `train_rain_calibration.py` | CLI entrypoint. Trains per-city logistic + isotonic on the rolling precipitation archive. Analogous to `train_calibration.py`. |
| `evaluate_rain_calibration.py` | CLI entrypoint. `--days 400 --holdout-days 30` chronological split evaluation; reports Brier, reliability, log-loss vs climatology baseline. Analogous to `evaluate_calibration.py`. |
| `strategy/rain_policy_v1.json` | Rain-specific policy (see Section: Policy). |

## Rain Market Coverage

**Watchlist (10 cities):** New York, Chicago, Boston, Philadelphia, Miami,
Atlanta, Houston, Seattle, New Orleans, Washington DC.

Rationale: climatologically wet enough for meaningful calibration sample sizes
(~30%+ wet days), established Kalshi temperature liquidity suggests rain
liquidity, and all 10 already have full station-truth history. Dry cities
(Phoenix, Las Vegas, Los Angeles, Denver) are excluded from P1 — their
rare-rain tail is precisely what P2 will tackle once calibration tools exist.
DC is included on the rain side even though it's temperature-blocked; the
block was a temperature-specific miscalibration.

**Market scope at launch:** binary any-rain contracts only (`≥ 0.01 in`
threshold). Moderate thresholds (`≥ 0.25 in`) and heavy thresholds (`≥ 1.0 in`)
are out of scope for P1. Adding them is a Phase 1.5 enhancement after binary
is proven.

**Discovery pass (pre-implementation):** before any rain-pipeline code is
written, run the existing `fetch_kalshi.py` with `KXRAIN` filtering to produce
a one-off `data/rain_market_inventory.csv` of ticker + outcome + 24 h volume
across the watchlist. This sets policy thresholds from real data, not
assumptions.

## Calibration Stack

Five layers, parallel to the temperature stack:

1. **Raw probability.** Extract `precipitation_probability_max` for the target
   date from the existing Open-Meteo ensemble call, blended 50/50 with
   **wet-fraction** from ensemble members:
   `count(member_precip_total >= 0.01 in) / total_members`.
2. **Logistic bias correction.** Per-city logistic regression on historical
   `(raw_prob, actual_wet_0_1)`. Class: `LogisticRainCalibrator`. Fit on the
   rolling `calibration_window_days` (default 90) of archived
   forecast-vs-actual pairs.
3. **Isotonic probability calibration.** Same pattern as the existing
   temperature isotonic step. Class: `IsotonicRainCalibrator`.
4. **HRRR same-day blend.** Within `rain_hrrr_blend_horizon_hours` (default
   12) of settlement, blend HRRR APCP over the target day window into the
   probability. Blend weight ramps linearly with proximity to close, capped at
   0.7 (HRRR never fully overrides the calibrated value).
5. **Selective raw fallback.** Same mechanism as the temperature pipeline:
   cities where calibration empirically hurts holdout metrics are bypassed.
   Populated after ~60 days of holdout data, not pre-declared.

### Evaluation

`evaluate_rain_calibration.py --days 400 --holdout-days 30` does chronological
split evaluation and reports:

- **Brier score** per city and aggregate, baseline = climatology
- **Log-loss** vs climatology baseline
- **Reliability diagram** (10-bin) per city
- **Targeted-pair deltas** if selective fallback is populated

Retraining cadence: weekly (Sunday night), same as the existing temperature
weekly retrain timer.

## Policy (`strategy/rain_policy_v1.json`)

Conservative starting values — mirrors the cautious posture of temperature v4:

```json
{
  "policy_version": 1,
  "status": "active",
  "market_category": "rain",
  "generated_at_utc": "<ISO8601 on creation>",
  "objective": "Paper-only evaluation of binary any-rain markets on 10-city watchlist. 30-day gate before any real-money consideration.",
  "selection": {
    "sources": ["kalshi"],
    "min_abs_edge": 0.15,
    "min_volume24hr": 500,
    "max_candidates_per_scan": 2,
    "max_hours_to_settlement": 24,
    "allowed_market_types": ["rain_binary"],
    "allowed_position_sides": ["yes"],
    "allowed_cities": ["New York", "Chicago", "Boston", "Philadelphia", "Miami", "Atlanta", "Houston", "Seattle", "New Orleans", "Washington DC"],
    "blocked_cities": []
  },
  "execution": {
    "max_contracts_per_trade": 1,
    "max_new_orders_per_day": 2,
    "max_order_cost_dollars": 10.0,
    "time_in_force": "fill_or_kill"
  },
  "rationale": {
    "v1_notes": "Initial launch of rain vertical. Thresholds intentionally conservative to mirror temperature v4's BUY-only lottery-ticket posture while calibration evidence accumulates. min_volume24hr set lower than temperature (500 vs 2000) because rain markets are expected to be smaller."
  }
}
```

Discovery-pass inventory may motivate a one-time adjustment to
`min_volume24hr` before v1 is locked in.

## Risk, Bankroll, Correlation Telemetry

- **`rain_bankroll_fraction: 0.20`** — rain gets a 20% slice of the total
  bankroll at P1. Quarter-Kelly sizing from `src/sizing.py` applies against
  this slice only; temperature Kelly is untouched.
- **Paper-only gate:** minimum 30 trading days before any real-money
  consideration. Promotion criteria:
  1. Log-loss strictly less than climatology baseline on holdout.
  2. Paper ROI ≥ 0 net of fees.
  3. Pearson correlation of the daily settled-PnL time series between
     `market_category=rain` and `market_category=temperature` < 0.4 over the
     evaluation window.
- **Correlation monitoring:** `summary.json` gains a `category_breakdown`
  block (correlation is Pearson on the trailing-30-day daily settled-PnL
  series; reported alongside sample size because the statistic is noisy at
  small n):
  ```json
  "category_breakdown": {
    "temperature": {"trade_count": ..., "pnl": ..., "roi": ..., "fees": ...},
    "rain": {"trade_count": ..., "pnl": ..., "roi": ..., "fees": ...},
    "correlation_30d": 0.12,
    "correlation_sample_size": 48
  }
  ```
  If `correlation_30d > 0.4` for two consecutive summaries, dial
  `rain_bankroll_fraction` down (manual, not automatic in P1).

## Data & Config Schema

### New Directories

- `data/precip_archive/{city_slug}.csv`
  ```
  as_of_utc, date, forecast_prob_any_rain, forecast_amount_mm,
  ensemble_wet_fraction, ensemble_amount_std_mm, forecast_model,
  forecast_lead_days
  ```
- `data/calibration_models/{city_slug}_rain_binary_logistic.pkl`
- `data/calibration_models/{city_slug}_rain_binary_isotonic.pkl`

### Ledger Change

Add `market_category` column to `data/paper_trades/ledger.csv`. Legacy rows
backfill deterministically as `temperature` (same pattern as
`legacy_unknown` route backfill). New rain rows carry `market_category=rain`.

### Settlement Rule

```python
wet_day = (precip_in >= 0.01) or (precip_trace and market_threshold == 0.01)
```

Trace-day handling is configurable in anticipation of future
non-binary-threshold markets.

### Config Additions (`config.example.json`, all default off)

```json
"enable_rain_vertical": false,
"rain_watchlist": [
  "New York", "Chicago", "Boston", "Philadelphia", "Miami",
  "Atlanta", "Houston", "Seattle", "New Orleans", "Washington DC"
],
"rain_hrrr_blend_horizon_hours": 12,
"rain_bankroll_fraction": 0.20,
"rain_strategy_policy_path": "strategy/rain_policy_v1.json",
"rain_min_edge_threshold": 0.15,
"rain_min_volume24hr": 500,
"rain_max_candidates_per_scan": 2,
"rain_calibration_window_days": 90
```

## P2 / P3 Hooks

**P2 (SELL-side and bucket-market rehab via tail calibration):**

- `LogisticRainCalibrator` and `IsotonicRainCalibrator` interfaces accept any
  binary-outcome probability input. P2 instantiates them against temperature
  tail events (e.g., "high > climate mean + 2σ") without new class code.
- The `evaluate_rain_calibration.py` metric suite (Brier, reliability,
  log-loss vs climatology) is written to be dataset-agnostic; P2 feeds it
  temperature tail data.

**P3 (multi-model ensemble):**

- `fetch_precipitation.py` and (retrospectively) `fetch_forecasts.py` expose
  a `models` parameter. Default remains Open-Meteo's blend; P3 supplies an
  explicit list like `["ecmwf_ifs04", "gfs_seamless", "icon_seamless"]`.
- Ensemble σ / wet-fraction calculations already accept a members array; P3
  supplies a bigger array with a `model` dimension, and the calibration code
  gains a `per_model_weight` parameter without breaking the existing call
  sites.

## Rollout Order

1. **Discovery pass.** Run existing fetcher with `KXRAIN` enabled end-to-end;
   dump `data/rain_market_inventory.csv` (ticker, outcome, 24 h volume, city,
   date). First empirical reality check; may adjust `min_volume24hr` in v1
   policy.
2. **`src/fetch_precipitation.py`** — Open-Meteo precip (deterministic +
   ensemble) and HRRR APCP extraction at station points.
3. **Archive immediately** — precipitation forecast snapshots start landing in
   `data/precip_archive/` the moment the fetcher is wired, building training
   data while the rest of P1 is being written.
4. **`src/rain_calibration.py` + `train_rain_calibration.py`** — classes and
   CLI trainer. Until ~60 days of archived precip + actuals exist, the live
   pipeline uses pass-through raw Open-Meteo probability.
5. **`src/rain_matcher.py`** — probability → contract edge, mirroring
   `matcher.py` opportunity shape.
6. **`strategy/rain_policy_v1.json` + category-aware filter in
   `src/strategy_policy.py`.**
7. **Ledger & telemetry changes** — `market_category` column, backfill legacy
   rows, `category_breakdown` + correlation fields in `summary.json`.
8. **Tests** — mirror existing temperature coverage: matcher, calibration,
   policy filter, settlement, telemetry, end-to-end paper-trade round trip.
9. **Wire into `main.py`** behind `enable_rain_vertical` flag.
10. **Go live (paper).** 30-day evaluation window begins.
11. **Evaluation gate** — review Brier, log-loss, ROI, and correlation. If all
    three promotion criteria pass, begin real-money consideration discussion.
    If any fail, diagnose and iterate before a v2 policy.

## Success Criteria

At the end of the 30-day paper evaluation window:

- ✅ Rain calibration log-loss beats climatology baseline per city on the
  rolling holdout.
- ✅ Paper ROI ≥ 0 net of fees.
- ✅ Pearson correlation of daily settled-PnL series (rain vs temperature)
  over the 30-day evaluation window < 0.4.
- ✅ No data leakage / lookahead detected in calibration holdout splits.
- ✅ Settlement audit: every rain paper trade has a matched station-truth row
  within the cutoff, or is explicitly surfaced as a truth blocker.

If all five pass, P2 (tail-calibration rehab on temperature) begins. If one or
more fail, iterate on the rain pipeline before starting P2.

## Open Questions

These are pinned here rather than resolved, because each depends on empirical
data from the discovery pass or the first weeks of archiving:

1. **Kalshi rain ticker structure** — exact outcome-string format for
   `KXRAIN*` markets is verified during the discovery pass, not pre-assumed.
   The `rain_matcher.py` parser should be written after the inventory CSV is
   in hand.
2. **Rain market liquidity** — if discovery shows 24 h volume chronically
   below 500 across the watchlist, `min_volume24hr` is adjusted or the
   watchlist is trimmed before policy v1 is committed.
3. **Trace-day handling** — binary `≥ 0.01` markets may or may not resolve
   YES on trace-only days. This is verified against at least one Kalshi
   settlement example before the matcher is finalized; configurable flag is
   left in place regardless.

## References

- `UPGRADE_PLAN.md` — context for the existing temperature modeling stack.
- `.claude/rules/calibration.md` — calibration routing order, selective raw
  fallback mechanism.
- `.claude/rules/paper-trading.md` — fee model, settlement flow, truth
  blocker reporting.
- `strategy/strategy_policy.json` — current temperature policy (v4).
- `data/CLAUDE.md` — data directory schemas and safety rules.
