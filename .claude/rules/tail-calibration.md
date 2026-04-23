# Tail Calibration Rules (P2)

## Scope

P2 reclaims blocked temperature SELL-side and bucket-market trades via a
tail-specific calibration layer composed on top of the existing chain.
Activated by `enable_tail_calibration: true` in config.

## Routing

- BUY-side thresholds: use existing isotonic path unchanged.
- SELL-side thresholds: route through `TailCalibrationManager` if models
  exist for (city, market_type, direction). Trade only if pair listed in
  `strategy_policy.json` `tail_unblocks.threshold_sell`.
- Bucket markets: route through `BucketDistributionalCalibrator`. Trade
  only if pair listed in `tail_unblocks.bucket`.

## Calibration chain (per pair)

```
raw_prob → LogisticRainCalibrator (bias correction)
        → IsotonicRainCalibrator (probability recalibration)
        → calibrated_prob
```

Same two-stage chain as P1 rain, composed via
`src/tail_calibration.TailBinaryCalibrator` /
`BucketDistributionalCalibrator`.

## Training

- `train_tail_calibration.py` — per-pair, rolling
  `tail_calibration_window_days` (default 180).
- Sample-size gate: >=30 tail-region rows (raw_prob < 0.25 or > 0.75) and
  >=2 actual tail events on each side of the raw_prob distribution.
  Pairs below the gate produce no model.
- `uncertainty_std_f` config key (default 2.0) is used as the fallback sigma
  when `ensemble_*_std_f` is NaN in the forecast archive — mirrors the
  serving-side fallback in `src/matcher.py`.

## Train/serve parity

Training must match serving exactly to avoid silent calibration bias. The
current parity invariants (locked in by regression tests):

1. Gaussian CDF `_normal_cdf` is imported directly from `src.matcher`
   (not duplicated).
2. Threshold markets with direction "above" use the `+0.99` offset on
   `threshold` in BOTH `_raw_prob_above` and `_threshold_exceeded`
   (matches `_kalshi_threshold_yes_probability("above", ...)`).
3. Sigma is clipped to `[_ENSEMBLE_SIGMA_FLOOR_F, _ENSEMBLE_SIGMA_CAP_F]`
   (1.0-6.0) before CDF computation, matching
   `_ensemble_sigma_for_date` in the matcher.
4. NaN sigma in the archive is filled with `uncertainty_std_f` (default
   2.0) before clipping, matching the serving fallback.

Parity tests live in `tests/test_tail_training_data.py` and will fail if
any of these drifts.

## Evaluation gate

A pair qualifies for unblock only if on the chronological holdout:

1. Tail-region log-loss strictly beats climatology baseline by at least
   `_MIN_IMPROVEMENT_NATS = 1e-3`.
2. Tail-region log-loss with the tail calibrator strictly beats tail-region
   log-loss with the existing raw isotonic calibrator by at least
   `_MIN_IMPROVEMENT_NATS = 1e-3`.

The minimum-improvement margin prevents noise qualifications where the
tail calibrator's logistic stage degenerates to approximate identity.
Scorecards written to
`data/evaluation_reports/tail_eval_YYYY-MM-DD.json`.

## Unblock workflow

- Unblock requires **two consecutive weekly scorecards** both reporting
  `qualifies_for_unblock: true` for the same pair.
- Policy update is manual: human adds the pair to
  `strategy_policy.json` `tail_unblocks.*` with
  `bankroll_slice: "probation"` and a `scorecard_ref`. Bump
  `policy_version`.

## Probation

Newly-unblocked pairs draw from the `probation` bankroll slice (10% of
total, per `strategy_policy.json` `bankroll_slices`) for the first 30
days of paper trading. Kelly sizing is scaled by the slice fraction
(via `bankroll_fraction_multiplier` on `compute_position_size`).
Promotion from probation → `temperature_buy` requires 30 days of ROI
>= 0 on the pair; manual policy edit to change the pair's
`bankroll_slice` value.

## Rollback

Remove the pair from `tail_unblocks` (one-line JSON edit, bump
`policy_version`) and append the failure to `_learnings.md`.
