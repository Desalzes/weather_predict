# 2026-04-23 17:00 — P2 tail-calibration complete

## Context

Full P2 tail-calibration implementation — 15 tasks executed via subagent-
driven development over this session. Reclaims blocked temperature SELL-side
and bucket-market trades via a parallel calibration layer composed on top
of the existing chain. Gated behind `enable_tail_calibration: false` so
flag-off behavior is identical to pre-P2. 21 commits on
`feature/p2-tail-calibration`, 278 tests passing, 0 deselected.

## What changed

**New modules:**
- `src/tail_training_data.py` — `build_tail_training_set`,
  `build_bucket_training_set` (forecast-archive × station-actuals joins
  with train/serve parity guarantees)
- `src/tail_calibration.py` — `TailBinaryCalibrator`,
  `BucketDistributionalCalibrator`, `TailCalibrationManager` (mtime-
  invalidating cache)
- `train_tail_calibration.py` — CLI trainer (produced 83 initial models
  across 20 cities after the NaN-sigma fallback fix)
- `evaluate_tail_calibration.py` — chronological-holdout evaluator with
  `_MIN_IMPROVEMENT_NATS = 1e-3` margin

**Modified:**
- `src/matcher.py` — `tail_calibration_manager` kwarg + conditional
  `our_probability_tail`/`edge_tail`/`tail_calibration_source` attachment
- `src/strategy_policy.py` — `apply_tail_unblocks` filter
- `strategy/strategy_policy.json` — bumped to v5 with `tail_unblocks`
  (empty at launch) + `bankroll_slices`
- `src/paper_trading.py` — `bankroll_slice` ledger column with
  category-aware defaults
- `src/sizing.py` — `bankroll_fraction_multiplier` param on
  `compute_position_size`
- `main.py` — wires `TailCalibrationManager` + `_tail_preroute` behind
  `enable_tail_calibration` flag
- `scripts/autopilot_weekly.py` — runs tail retrain + holdout eval weekly
- `config.example.json` — `enable_tail_calibration: false`,
  `tail_calibration_window_days: 180`, `tail_bankroll_fraction: 0.10`

**Docs:**
- `.claude/rules/tail-calibration.md` — rules for scope, routing,
  parity invariants, evaluation gate, unblock workflow, probation
- `src/CLAUDE.md`, `strategy/CLAUDE.md` — module-map + v5 policy section
- `docs/adr/0001-tail-calibration-reuses-p1.md` + mirror in
  codebase-memory-mcp

**21 commits on `feature/p2-tail-calibration`:**
`3e39316` → `cb20eb2`. Full list via `git log main..HEAD --oneline`.

## Current state

**Working:**
- Full test suite: 278 passing, 0 deselected
- Smoke test: all P2 wiring resolves; Atlanta high above tail calibration
  returns `{calibrated_prob: 0.667, source: "logistic+isotonic"}` on
  `raw_prob=0.1`; `apply_tail_unblocks` correctly returns `None` for empty
  `tail_unblocks`
- Policy v5 loads: `bankroll_slices` sum to 1.0
- 249 initial calibration pkl artifacts committed at `6552f92`

**Not committed in this session:**
- Nothing pending from P2. Branch ready to merge.

**Uncommitted working-tree drift (unrelated to P2, pre-existing):**
- `data/forecast_archive/*.csv` — 20 modified (left over from live-scan
  snapshots)
- `strategy/strategy_policy.json` — no (merged clean into v5 commit)
- `.claude/settings.local.json` — local settings drift
- Untracked: `WEATHER_PROJECT_BRIEFING.md`, `codex_loop/`, `data/test_runs/`,
  `llm-pm/` (pre-extraction debris)

## Next action

**Merge `feature/p2-tail-calibration` into main.**

Specific commands:

```bash
git checkout main
git merge --no-ff feature/p2-tail-calibration -m "Merge branch 'feature/p2-tail-calibration' into main"
git push origin main
git branch -d feature/p2-tail-calibration
```

After merge, the operator-facing step is to flip
`enable_tail_calibration: true` in local `config.json` to start
accumulating weekly scorecards. Autopilot will handle tail retrain +
holdout eval on Sundays. No pair gets unblocked until two consecutive
weekly scorecards agree on `qualifies_for_unblock: true`.

## Watch out for

- **Train/serve parity is load-bearing.** Three separate skews were found
  and fixed during this session — see `_learnings.md` for all three. The
  parity tests in `tests/test_tail_training_data.py` will fail if any
  drifts. Do NOT refactor `src/tail_training_data.py` without
  understanding why each invariant exists.
- **Minimum-improvement margin (`_MIN_IMPROVEMENT_NATS = 1e-3`) in the
  evaluator.** First scorecard had 37 "qualifying" pairs but 23 were
  floating-point noise. The margin brought it to 13 real qualifications.
  Do not remove or loosen this.
- **`match_kalshi_markets` signature has no `*,` barrier before the
  keyword-only-by-convention params.** This is latent footgun — a
  future caller passing positional args after position 4 could silently
  misroute. Task 7 review flagged this as Minor. Fix is a one-character
  edit; documented as a P2-priority backlog item.
- **Weather project in codebase-memory-mcp uses the name
  `C-Users-desal-Desktop-Projects-_Betting_Markets-Weather`**, NOT
  "Weather". The skills hint "Weather" but that's wrong — use
  `list_projects` to find the real name.

## ADRs touched

- Created: ADR-0001 — Tail calibration reuses P1 binary-outcome
  calibrators (accepted 2026-04-23). Stored in codebase-memory-mcp
  and mirrored to `docs/adr/0001-tail-calibration-reuses-p1.md`.
