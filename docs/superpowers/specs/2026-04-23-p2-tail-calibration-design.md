# P2 — Temperature Tail Calibration Design

**Date:** 2026-04-23
**Status:** Approved design, ready for implementation planning
**Author:** Claude (brainstorming skill) with Gabriel

## Context

Temperature policy v4 (2026-04-20) is BUY-only after SELL-side evidence
produced −50.6 % ROI on 35 settled paper trades (8W/27L). Bucket markets were
blocked earlier in v3 after 8 of 9 settled bucket trades lost. The v4
rationale diagnoses the failure as Gaussian-tail miscalibration: the EMOS /
NGR / isotonic stack underestimates rare events, so SELL-side bets at
"impossible" probabilities lose at a higher rate than the model predicts.

P2 is the middle phase of the three-phase diversification program laid out
in the 2026-04-20 rain vertical design:

- **P1 (done):** Rain vertical. Shipped `LogisticRainCalibrator` and
  `IsotonicRainCalibrator` as deliberately generic binary-outcome calibrators
  that P2 would reuse.
- **P2 (this spec):** Reclaim blocked temperature SELL-side and bucket
  markets by adding a tail-specific calibration layer on top of the existing
  chain. Evidence-gated unblock per (city, market_type, direction) pair.
- **P3 (future):** Multi-model ensemble (ECMWF + GFS + ICON) stacked alongside
  Open-Meteo's current blend. Benefits both verticals.

## Goals

1. Reclaim the blocked SELL-side and bucket-market trade surface without
   destabilizing the working BUY book.
2. Reuse P1's `LogisticRainCalibrator` / `IsotonicRainCalibrator` classes
   verbatim — P1 was explicitly designed with the generic interface this
   phase needs.
3. Unblock trades only on evidence. Per-pair holdout gate, mirroring the
   existing `SELECTIVE_RAW_FALLBACK_TARGETS` mechanism.
4. Keep probation risk low: new unblocks draw from a separate 10 % bankroll
   slice for their first 30 days of paper trading.

## Non-Goals

- Replacing the Gaussian base distribution with Student-t (option A from
  brainstorm).
- Empirical σ inflation at the distribution layer (option B).
- Changing the existing BUY-side probability chain. Anything that would touch
  `calibrate_probability` on the existing `CalibrationManager` is out of
  scope — P2 adds a parallel branch, never modifies the current one.
- Automated unblocking. Every pair unblock is a human policy edit referencing
  a scorecard file.
- Multi-model ensemble (that's P3).

## Architecture

### Current flow (unchanged for BUY side)

```
forecast + σ  →  EMOS bias  →  NGR distribution  →  Normal CDF
                                     ↓
                              isotonic prob cal (disabled during HRRR-blend regime)
                                     ↓
                                  our_prob
                                     ↓
                              edge vs market → direction = BUY if edge > 0 else SELL
```

### P2 adds a parallel tail branch

```
                                      ┌─ BUY, non-bucket, edge > 0
                                      │     → existing chain → our_prob
raw_threshold_or_bucket_prob  →  route├─ SELL side
                                      │     → TailBinaryCalibrator(city, mtype, direction) → our_prob_tail
                                      └─ bucket market
                                            → BucketDistributionalCalibrator(city, mtype) → our_prob_tail
```

Both `our_prob` (isotonic) and `our_prob_tail` are always carried on the
opportunity dict when tail models exist. The policy decides which number
drives the trade decision.

### Reused from P1 verbatim

- `LogisticRainCalibrator` — generic over any binary-outcome probability
  input. Imported, not duplicated.
- `IsotonicRainCalibrator` — same.
- Manager pattern with mtime-based cache invalidation (from cleanup-pass
  commit `80dd53a`).
- `build_rain_training_set` data-join pattern — generalized as
  `build_tail_training_set`.
- Per-pair selective evaluation mirrored after the existing
  `SELECTIVE_RAW_FALLBACK_TARGETS` mechanism.

### New modules

| Module | Responsibility |
|---|---|
| `src/tail_calibration.py` | `TailBinaryCalibrator` (threshold events), `BucketDistributionalCalibrator` (bucket events), `TailCalibrationManager` (loads / caches models). |
| `src/tail_training_data.py` | `build_tail_training_set(city, market_type, direction)` — (raw_prob, actual_exceeded_0_1) pairs from archived forecasts + station actuals. |
| `train_tail_calibration.py` | CLI trainer. Per-pair, rolling 180-day window. Respects sample-size gate; skips + logs pairs that don't qualify. |
| `evaluate_tail_calibration.py` | Chronological-holdout evaluator. Writes scorecard JSON with `qualifies_for_unblock: bool` per pair. |

### Modified modules

| File | Change |
|---|---|
| `src/matcher.py` | One conditional branch: after computing `our_probability`, call `TailCalibrationManager.calibrate_tail_probability(...)` and attach `our_probability_tail` + `edge_tail` + `tail_calibration_source` when models exist. ~25 lines. |
| `src/strategy_policy.py` | New `apply_tail_unblocks(opportunity, policy)` filter. For SELL/bucket opportunities, consults `tail_unblocks` list and swaps in `our_probability_tail` / `edge_tail` as decision values. Unknown pairs stay blocked. |
| `strategy/strategy_policy.json` | Bump to v5. Adds `tail_unblocks` (empty list at launch), `bankroll_slices` block. Existing v4 fields unchanged. |
| `src/paper_trading.py` | New `bankroll_slice` column (values: `temperature_buy` \| `rain_binary` \| `probation`). Legacy rows backfill as `temperature_buy`. `summary.json` gains per-slice PnL breakdown (same mechanism as P1's `category_breakdown`). |
| `src/sizing.py` | Quarter-Kelly accepts a `bankroll_slice` parameter; computes against the slice's fraction of total bankroll. Default `"temperature_buy"` preserves current behavior. |
| `scripts/autopilot_weekly.py` | Adds `train_tail_calibration.py` + `evaluate_tail_calibration.py` steps after the existing weekly temp retrain. |
| `config.example.json` | `enable_tail_calibration: false`, `tail_calibration_window_days: 180`, `tail_bankroll_fraction: 0.10`. |

## Training Data + Calibration Math

### Training row shape (per pair)

```
threshold markets: (raw_prob, actual_exceeded_0_1)
bucket markets:    (raw_bucket_prob, actual_in_bucket_0_1)
```

`raw_prob` is synthesized at training time from the archived forecast value
and σ via the same Gaussian CDF the live matcher uses. This preserves the
causal chain: the calibrator learns the correction that applies to live
opportunities.

### No tail filter on training

The calibrator sees the full probability range. Isotonic regression needs
anchor points in the center to constrain its monotone shape; filtering to
tail-only data overfits sparse extremes. Tail specialization happens at the
**evaluation gate**, not the training filter.

### Window

Rolling **180 days** (`tail_calibration_window_days: 180`), wider than the
current 90-day temperature window. Tails need more samples to fit reliably.

### Sample-size gate (per pair, before the pair is eligible for evaluation)

- ≥ 30 samples in the tail region (raw_prob < 0.25 OR raw_prob > 0.75) within
  the training window
- ≥ 2 actual tail events on each side (can't fit logistic on degenerate
  outcome vectors)

Pairs below these thresholds produce no model; the scorecard marks them
`status: insufficient_data`.

### Calibration chain

```
raw_prob → LogisticRainCalibrator(city, mtype, direction) → p_logistic
p_logistic → IsotonicRainCalibrator(city, mtype, direction) → p_final
```

Same two-stage chain as P1. Fallback semantics identical: degenerate fits
become pass-through with a source tag that reflects it (`logistic`,
`isotonic`, or `raw`).

### Bucket markets

Same chain, different binary outcome. For a bucket `[T−0.5, T+0.5]`, training
row is `(F(T+0.5) − F(T−0.5), actual_temp ∈ [T−0.5, T+0.5])`. One calibrator
pair per (city, market_type) — no `direction` dimension.

### Evaluation gate

A pair qualifies for unblock only if, on the chronological holdout:

1. Tail-region log-loss (raw_prob < 0.25 OR > 0.75) **strictly beats
   climatology**.
2. Tail-region log-loss with the tail calibrator **strictly beats** tail-region
   log-loss with the existing raw isotonic calibrator.

The scorecard lists both values per pair plus `qualifies_for_unblock: bool`.

## Policy v5 Schema

```json
{
  "policy_version": 5,
  "status": "active",
  "generated_at_utc": "<iso8601 on creation>",
  "selection": {
    // ... all v4 fields unchanged ...
    "allowed_position_sides": ["yes"],
    "allowed_settlement_rules": ["lte", "gt"]
  },
  "tail_unblocks": {
    "threshold_sell": [
      {
        "city": "<City>",
        "market_type": "high|low",
        "direction": "above|below",
        "unblocked_at_utc": "<iso8601>",
        "scorecard_ref": "data/evaluation_reports/tail_eval_YYYY-MM-DD.json",
        "bankroll_slice": "probation"
      }
    ],
    "bucket": [
      {
        "city": "<City>",
        "market_type": "high|low",
        "unblocked_at_utc": "<iso8601>",
        "scorecard_ref": "...",
        "bankroll_slice": "probation"
      }
    ]
  },
  "bankroll_slices": {
    "temperature_buy": 0.70,
    "rain_binary": 0.20,
    "probation": 0.10
  }
}
```

`tail_unblocks` is **additive, not replacement**. `allowed_position_sides:
["yes"]` and `allowed_settlement_rules: ["lte", "gt"]` still blanket-block
SELL and bucket markets; the per-pair entries in `tail_unblocks` override
those blocks on a case-by-case basis.

`tail_unblocks` is empty at launch. First real unblock requires two
consecutive weekly scorecards that both say `qualifies_for_unblock: true`
for the same pair — at minimum ~7 days of forward time, plus human policy
edit time.

## Routing Logic in Matcher

```python
# Existing flow — unchanged
raw_prob = compute_temperature_probability(...)
our_prob = isotonic_calibrator.calibrate(raw_prob)   # existing path
market_price = ...
edge = our_prob - market_price
direction = "BUY" if edge > 0 else "SELL"

opp = {
    "our_probability": round(our_prob, 4),
    "edge": round(edge, 4),
    "direction": direction,
    # ...
}

# P2 addition — attach tail probability when models exist
if tail_manager is not None:
    tail_result = tail_manager.calibrate_tail_probability(
        city, market_type, direction, is_bucket, raw_prob,
    )
    if tail_result is not None:
        opp["our_probability_tail"] = round(tail_result["calibrated_prob"], 4)
        opp["edge_tail"] = round(opp["our_probability_tail"] - market_price, 4)
        opp["tail_calibration_source"] = tail_result["source"]
```

BUY-side isotonic path is untouched. Every existing `test_matcher.py`
assertion still holds. The opportunity dict simply gains optional fields.

## Policy Filter in `strategy_policy.py`

`apply_tail_unblocks(opp, policy)` runs after the existing filter chain,
specifically on opportunities that would otherwise be dropped by the
`allowed_position_sides` / `allowed_settlement_rules` filters:

1. Identify opportunity as SELL-threshold, SELL-bucket, or BUY-bucket (BUY
   thresholds aren't tail-routed — they use the existing isotonic
   `our_probability`).
2. Look up (city, market_type, direction or bucket-flag) in
   `policy.tail_unblocks`.
3. If matched: replace `our_probability` ← `our_probability_tail`, `edge` ←
   `edge_tail`, and the opportunity passes through. Tag the emitted trade
   with `bankroll_slice: "probation"`.
4. If not matched: opportunity is dropped (policy block stands).

## Bankroll Slices

- `temperature_buy`: 0.70 — unchanged BUY book.
- `rain_binary`: 0.20 — P1 rain vertical slice, unchanged.
- `probation`: 0.10 — all newly-unblocked tail pairs draw from here for their
  first 30 days of paper trading.

Quarter-Kelly sizing computes against the slice's fraction of the total
bankroll (not the total bankroll directly). So a pair under probation with
an edge that would normally size to $0.25 instead sizes to $0.036 ($0.25 ×
0.1 / 0.7). Promotion from probation to `temperature_buy` is a manual policy
edit after 30 days of probation with ROI ≥ 0.

## Data & Disk Schema

### Calibration model files

```
data/calibration_models/
  {city_slug}_{market_type}_{direction}_tail_logistic.pkl    # threshold events
  {city_slug}_{market_type}_{direction}_tail_isotonic.pkl
  {city_slug}_{market_type}_bucket_logistic.pkl               # bucket events
  {city_slug}_{market_type}_bucket_isotonic.pkl
```

Names deliberately distinct from the existing
`{city}_{market_type}_{emos|isotonic|ngr}.pkl` so the old calibration stack
stays operable and comparable.

### Evaluation scorecard

`data/evaluation_reports/tail_eval_YYYY-MM-DD.json`:

```json
{
  "generated_at_utc": "...",
  "args": {"days": 400, "holdout_days": 30},
  "pairs": [
    {
      "city": "New York",
      "market_type": "high",
      "direction": "above",
      "status": "ok" | "insufficient_data",
      "tail_region_n": 42,
      "tail_region_wet_n": 5,
      "log_loss_raw": 0.38,
      "log_loss_isotonic": 0.36,
      "log_loss_tail": 0.31,
      "log_loss_climatology": 0.34,
      "qualifies_for_unblock": true
    }
  ]
}
```

### Paper-trade ledger

Add `bankroll_slice` column. Legacy rows backfill as `temperature_buy`.
`summary.json` `category_breakdown` gains a nested `slice_breakdown` with
per-slice PnL/ROI/fee totals.

## Rollout Order

1. `src/tail_training_data.py` with `build_tail_training_set` — TDD, pure
   data join.
2. `src/tail_calibration.py` — `TailBinaryCalibrator`,
   `BucketDistributionalCalibrator`, `TailCalibrationManager`. Tests mirror
   `test_rain_calibration.py` patterns including mtime-invalidation tests.
3. `train_tail_calibration.py` — CLI trainer. Run once against committed
   archive; commit the produced pkls.
4. `evaluate_tail_calibration.py` — chronological-holdout evaluator. Run
   once; commit the baseline scorecard. Individual pairs may score
   `qualifies_for_unblock: true` on this first run, but the policy-side
   rule (two consecutive weekly scorecards must both agree) means no pair
   is actually unblocked until the next week's scorecard confirms.
5. `src/matcher.py` — conditional tail branch, attaches
   `our_probability_tail` to opportunities. Gated behind
   `enable_tail_calibration: true` in config. All existing tests stay green.
6. `src/strategy_policy.py` — `apply_tail_unblocks` filter.
7. `strategy/strategy_policy.json` → v5 (empty unblock lists, bankroll
   slices).
8. `src/paper_trading.py` — `bankroll_slice` column with legacy migration.
   `summary.json` slice breakdown.
9. `src/sizing.py` — slice-aware quarter-Kelly.
10. `scripts/autopilot_weekly.py` — weekly retrain + rescore integration.
11. Docs: `.claude/rules/tail-calibration.md`, module-map updates in
    `src/CLAUDE.md`, policy section in `strategy/CLAUDE.md`.
12. Merge to `main` with flag off. No trade-behavior change yet.
13. Operator flips `enable_tail_calibration: true` in local `config.json`.
14. First real qualifying-pair unblock ~30 + days later. Requires two
    consecutive weekly scorecards that agree.

## Tests

### New test files

- `tests/test_tail_calibration.py` — class-level fit/predict/save/load,
  degenerate-outcome fall-through, partial-model paths, mtime invalidation.
- `tests/test_tail_training_data.py` — join correctness, tail-region
  filtering, empty-archive fallback.
- `tests/test_matcher_tail_routing.py` — four scenarios: BUY path unaffected,
  SELL-unblocked path uses tail, SELL-not-unblocked drops, bucket-unblocked
  uses bucket calibrator.

### Extensions to existing tests

- `tests/test_strategy_policy.py` — `test_tail_unblocks_routes_sell_correctly`,
  `test_tail_unblocks_rejects_untriangulated_pairs`.
- `tests/test_paper_trading.py` — `test_bankroll_slice_column_migrates_legacy`.

### Regression guarantees

- `tests/test_matcher.py` unchanged; any failure indicates an unintended
  BUY-path regression.
- Expected final count: ~243 baseline + ~30 new = ~273 passing, 0 deselected.

## P2 Project Success Criteria

- ✅ Code merged to `main` behind `enable_tail_calibration` flag (flag off by
  default in `config.example.json`).
- ✅ Initial training + evaluation produce calibration artifacts + baseline
  scorecard.
- ✅ Full test suite green; zero regressions in `test_matcher.py`.
- ✅ `tail_unblocks` list empty at launch.
- ✅ ADR written via `mcp__codebase-memory-mcp__manage_adr` (and mirrored to
  `docs/adr/`) documenting the P2 architecture and the
  evidence-gated-unblock pattern.

## Per-Pair Promotion Criteria (post-launch)

Tracked separately in `_state.md`. A pair moves from blocked → probation →
promoted when:

- **blocked → probation:** two consecutive weekly scorecards both report
  `qualifies_for_unblock: true`. Policy v6+ adds the pair to
  `tail_unblocks.*` with `bankroll_slice: "probation"`.
- **probation → promoted:** 30 days of paper trading with ROI ≥ 0 net of
  fees and correlation with existing BUY book < 0.4. Policy update changes
  the pair's `bankroll_slice` to `"temperature_buy"`.
- **any → re-blocked:** if a promoted or probation pair posts ROI < −5 % on
  its 30-day rolling window, remove from `tail_unblocks` and capture the
  failure in `_learnings.md`.

## P3 Hooks Pre-Wired

- Tail chain input is a single float `raw_prob`; P3 swap of Open-Meteo for a
  multi-model ensemble changes how that float is computed upstream, not the
  chain itself.
- `tail_calibration_source` is a string tag; P3 can add values like
  `"ensemble_tail_logistic+isotonic"` without schema migration.
- `build_tail_training_set` reads the forecast archive through existing
  helpers; P3's multi-model extensions of those helpers are inherited
  automatically.

## Open Questions

Pinned, not blocking:

1. **Bucket calibrator's 3-class degeneracy.** A bucket with very narrow
   width (2 °F) and climatologically rare occupancy (e.g., Phoenix in
   `60–62 °F` in July) will fail the sample-size gate on most cities. The
   scorecard should track which buckets are structurally untrainable so we
   don't keep re-scoring them.
2. **Cross-pair correlation during probation.** Two probation pairs in the
   same climate zone (e.g., Boston low and New York low) will correlate.
   10 % probation bankroll is the blast-radius cap, but we should monitor
   and consider a per-zone sub-cap after the first 5 pairs unblock.
3. **Probation → promotion automation.** First cycle is manual. If the
   pattern holds, a future session can write a small script that reads
   `ledger.csv`, computes per-pair 30-day ROI, and proposes promotion diffs
   to the policy. Not in P2 scope.

## References

- `docs/superpowers/specs/2026-04-20-rain-vertical-p1-design.md` — P1 design
  with reusable calibrator interfaces.
- `.claude/rules/calibration.md` — existing temperature calibration routing
  and selective raw fallback mechanism.
- `.claude/rules/paper-trading.md` — fee model, ledger schema, settlement
  flow.
- `strategy/strategy_policy.json` v4 rationale — the SELL-side
  −50.6 % / bucket 8-of-9-loss evidence that motivates P2.
