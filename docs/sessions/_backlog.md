# Backlog

Priority order. Items are never deleted; completed ones get
`[x]` and a completion date. This is the project's memory.

## P0 — Must do soon

- [ ] Merge `feature/p2-tail-calibration` → main + push to origin once
      user approves. 21 commits, 278 passing, behind flag.
- [ ] Enable `enable_tail_calibration: true` in local `config.json` to
      start accumulating weekly scorecards. Flag-on is a manual operator
      step per the P2 spec; autopilot will then run tail retrain + eval
      every Sunday.

## P1 — Should do

- [ ] After first two weekly tail scorecards agree on any (city,
      market_type, direction) pair with `qualifies_for_unblock: true`,
      bump `strategy/strategy_policy.json` to v6 and add the pair to
      `tail_unblocks.threshold_sell` (or `.bucket`) with
      `bankroll_slice: "probation"` and the scorecard_ref.
- [ ] After 30 days of ROI >= 0 on a probation pair, promote it to
      `bankroll_slice: "temperature_buy"` via policy v7.
- [ ] Rain vertical 30-day paper evaluation — still accumulating since
      Apr 22 Kalshi discovery pass. Evaluate against the promotion gate
      from the P1 spec (log-loss vs climatology, ROI >= 0, correlation
      with temperature < 0.4).

## P2 — Nice to have

- [ ] Add `*,` keyword-only barrier to `match_kalshi_markets` signature
      (Task 7 review minor item M1). Prevents positional-arg regressions
      when new kwargs are added later.
- [ ] Extract shared save/load helper for `TailBinaryCalibrator` /
      `BucketDistributionalCalibrator` — ~40 lines of near-identical
      boilerplate. Wait for a third consumer before refactoring.
- [ ] Extend tail-calibration to Polymarket (`match_polymarket_markets`).
      Deferred in Task 7 because Polymarket outcome shapes (range
      strings) differ materially from Kalshi thresholds.
- [ ] Sweep Kalshi-observed thresholds per city instead of training on a
      single median threshold per market_type. Currently
      `train_tail_calibration.py` picks `thresholds[len//2]` as the
      representative fit; per-city sweeps would broaden coverage.
- [ ] Start P3: multi-model ensemble (ECMWF + GFS + ICON) stacked
      alongside Open-Meteo. Benefits both verticals (temp + rain).
      Spec + plan not yet written.

## P3 — Ideas parking lot

- [ ] Automate probation→promoted transition via a weekly script that
      reads `ledger.csv`, computes per-pair 30-day ROI, proposes policy
      diffs. Only worth building after the first 3–5 pairs have gone
      through promotion manually.
- [ ] Tighter sample-size gate for bucket calibrator for dry-climate
      cities (Phoenix, Las Vegas) where bucket occupancy is rare — may
      not be trainable at current archive depth.
- [ ] Cross-pair probation correlation tracking: two probation pairs in
      same climate zone will correlate; 10% slice is the blast cap but
      per-zone sub-cap may be warranted after 5+ unblocks.
- [ ] `_normal_cdf` vectorization in `src/tail_training_data.py` —
      `.apply(lambda r: ...)` is row-by-row; numpy-vectorized would
      scale better for bigger archives. Not urgent at 20 cities × ~500
      days.

## Completed

- [x] 2026-04-23 — P2 tail-calibration P2 implementation (15 tasks, 21
      commits, 278 tests) on `feature/p2-tail-calibration`
- [x] 2026-04-23 — ADR-0001: Tail calibration reuses P1 binary-outcome
      calibrators (stored in codebase-memory-mcp + `docs/adr/`)
- [x] 2026-04-22 — llm-pm session-orientation scaffold installed +
      merged + pushed (commit `111b6bb` on main)
- [x] 2026-04-22 — Cleanup pass (4 fixups: brittle date fixture,
      archive_previous_run_precipitation polish, fetch_precipitation
      exception narrowing, calibration-cache mtime invalidation) merged
      to main
- [x] 2026-04-22 — Rain vertical P1 (14 tasks) merged + pushed; scanner
      wired for KXRAIN per-series discovery; 30-day paper-eval window
      begins
