# ADR-0001: Tail calibration reuses P1 binary-outcome calibrators

Status: accepted
Date: 2026-04-23
Deciders: Gabriel + Claude

## Context

Temperature policy v4 blocks SELL-side (35 trades, -50.6% ROI under v3) and
bucket markets (8 of 9 lost) because the Gaussian-based probability chain
underestimates rare events. P1 rain vertical built `LogisticRainCalibrator`
and `IsotonicRainCalibrator` in `src/rain_calibration.py` as deliberately
generic binary-outcome calibrators explicitly noted for P2 reuse.

## Decision

P2 adds a parallel tail-calibration branch that composes P1's calibrators via
`TailBinaryCalibrator` (threshold events, per city/market_type/direction) and
`BucketDistributionalCalibrator` (bucket events, per city/market_type).
The BUY-side isotonic path is untouched. Per-pair holdout evidence — two
consecutive weekly scorecards with `qualifies_for_unblock: true` — gates
individual unblocks; newly-unblocked pairs run on a separate 10% probation
bankroll slice for 30 days before promotion to the main temperature_buy
slice.

Key train/serve parity invariants locked in by regression tests:
1. `_normal_cdf` imported directly from `src.matcher` (not duplicated)
2. `+0.99` offset on threshold for direction="above" (Kalshi resolution convention)
3. Sigma clipped to [1.0, 6.0] before CDF
4. NaN ensemble sigma filled with `uncertainty_std_f` (default 2.0) before clipping

Minimum improvement margin `_MIN_IMPROVEMENT_NATS = 1e-3` added to the
unblock gate after the first scorecard revealed 23 of 37 qualifying pairs
were floating-point-noise qualifications where the logistic stage had
degenerated to approximate identity.

## Alternatives considered

- **Student-t base distribution** — fatter tails by construction. Rejected
  because the change would affect every downstream probability including
  currently-profitable BUY trades.
- **Empirical sigma inflation** — simpler but broadens the center of the
  distribution to fix the tails; regresses BUY accuracy.
- **Per-bin reliability correction** — simpler than a calibrator chain but
  doesn't compose as cleanly with the existing isotonic pipeline.

## Consequences

- Positive: Surgical fix, BUY book untouched, reuses P1 code verbatim,
  evidence-gated unblocks mirror the existing `selective_raw_fallback`
  pattern, parity tests lock in train/serve invariants so future refactors
  can't silently drift.
- Negative: Two calibration stacks live side-by-side in
  `data/calibration_models/`; filename discipline (`_tail_` vs existing
  `_emos_`/`_ngr_`/`_isotonic_`) is load-bearing. Policy v5 grew from 30
  lines to ~55 with `tail_unblocks` and `bankroll_slices` — more surface
  for operator edits.
- Neutral: Unblock cadence is manual (two-scorecard rule, human policy
  edit). Automation can come later if the pattern holds.

## Revisit if

- More than 10 pairs unblocked and probation management becomes
  operationally heavy → consider automating promotion.
- Tail calibrator holdout log-loss beats raw isotonic by less than 5% on
  average across unblocked pairs → the chain isn't pulling its weight;
  reconsider Student-t (option A).
- A second caller of `TailBinaryCalibrator` / `BucketDistributionalCalibrator`
  appears — refactor save/load boilerplate into a shared helper (deferred
  per Task 3 code review recommendation).
