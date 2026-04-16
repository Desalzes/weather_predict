# Session Worklog: 2026-04-16

**Duration:** Full session (~3 hours)
**Scope:** Project review, documentation, test coverage, strategy fixes, evaluation improvements, VPS deployment

---

## Starting State

The Weather Signals project was at a functional but under-documented stage with several known gaps:
- Phases 1-4 of the upgrade plan complete (ensemble, station truth, calibration, HRRR)
- Phase 5 (GOES satellite) not started
- Phase 6 (evaluation) partially complete
- 6 test files covering core logic but leaving major modules untested
- No CLAUDE.md documentation in subdirectories
- `.claude/rules/` scaffolded but empty
- DeepSeek worker enabled but no longer needed
- Paper trading running but with a critical bug: strategy policy filter was bypassed when DeepSeek was disabled
- No VPS deployment

---

## What Was Done

### 1. Project Review

Comprehensive codebase audit covering architecture, upgrade plan status, strengths, and weaknesses. Key findings:

- **Architecture is solid** — clean separation between fetch, calibrate, match, trade, review layers
- **Calibration pipeline is well-engineered** — selective fallback routing, route attribution, lead-time regime diagnostics
- **Test gaps were significant** — HRRR, Polymarket, DeepSeek failure modes, and analyze.py had zero test coverage
- **Strategy policy was being bypassed** — the most critical bug found (details in section 5)

### 2. Documentation (8 files)

Created agent-oriented CLAUDE.md files in every major subdirectory and populated `.claude/rules/` with topic-specific rules:

| File | Content |
|------|---------|
| `src/CLAUDE.md` | Data flow diagram, 15-module responsibility table, key constants table, change rules |
| `tests/CLAUDE.md` | Test file map (updated as new tests were added), run commands, conventions, coverage gaps |
| `data/CLAUDE.md` | Directory layout with committed/gitignored status, CSV/JSON schemas for all data files, safety rules |
| `strategy/CLAUDE.md` | Policy v2->v3 thresholds, how DeepSeek and fallback filter consume it, Polymarket enablement path |
| `.claude/rules/calibration.md` | EMOS -> isotonic -> HRRR routing order, 5 selective fallback pairs, validation command |
| `.claude/rules/paper-trading.md` | Fee model formula, settlement flow, route attribution columns, truth blocker contract |
| `.claude/rules/secrets.md` | Exhaustive list of secret files, env vars, config inheritance pattern |
| Root `CLAUDE.md` | Added Documentation Map section pointing to all subdirectory docs |

**Why this matters:** Any agent (Claude, Codex, or human) entering a subdirectory now gets immediate context about what they're looking at and what constraints apply. Previously you had to read the entire README.md or trace through source code to understand module responsibilities.

### 3. Test Coverage (8 new test files, 97 new tests)

Went from 6 test files / 70 tests to 14 test files / 167 tests:

| File | Tests | What it covers |
|------|-------|----------------|
| `test_fetch_hrrr.py` | 19 | Temp conversion (K/C/F), UTC handling, init time resolution, location normalization, in-memory cache hit/miss/invalidation, xarray dataset extraction, high/low extraction from timeseries, Herbie import failure, empty locations |
| `test_fetch_polymarket.py` | 17 | Market type parsing from question strings, city alias resolution (all 20 cities), outcome range parsing (above/below/range/degrees), cache TTL behavior, API failure handling, deduplication across search paths, price type conversion |
| `test_deepseek_failure_modes.py` | 15 | Timeout returns failed (not crash), failed state persisted correctly, review_runs not incremented on failure, malformed JSON handling, empty choices (IndexError caught), missing decisions default to watch, HTTP 429/500/connection errors, disabled worker, no eligible opportunities, recovery after failure |
| `test_analyze.py` | 21 | Ensemble spread (multi-member stats, single-value zero spread, empty data, time key exclusion, normalized spread formula), weather event detection (rain amount + probability triggers, wind, temp swing, custom thresholds, missing fields gracefully handled), confidence scoring (horizon penalty increases, ensemble penalty reduces confidence, low precip = high confidence) |
| `test_fetch_kalshi.py` | 15 | Ticker parsing (high/low threshold/bucket, 2-4 char city codes, unknown codes), city code completeness (all 20 cities + alternates like DFW/DAL), weather prefix set, unauthenticated fetch with pagination + cursor, API failure returns partial, closed market filtering, group-by-city with threshold sorting |
| `test_fetch_goes.py` | 12 | Cloud adjustment math (clear sky = 0, full cloud at peak, off-peak = 0, partial cloud linear scaling, edge-of-peak reduction, custom max adjustment, cloud fraction clamped at 1.0), forecast adjustment wrapper (None data, clear sky, cloudy at peak, cloudy off-peak), scaffold returns None gracefully |
| `test_strategy_policy.py` | +2 | `allowed_settlement_rules` filter blocks bucket markets, empty rules allows all |

**Testing philosophy:** External API calls are mocked. Calibration math runs on real (small) datasets. Fee accounting is tested with exact arithmetic. All tests are deterministic and fast (167 tests in <2 seconds).

### 4. Pipeline Diagnosis

Ran `main.py --once` and `--settle-paper-trades` locally to see the live state:

**Scan results:**
- 20/20 forecasts + ensembles fetched from Open-Meteo
- 16 Kalshi markets found (authenticated client working)
- HRRR timed out after ~10 minutes on S3 read (gracefully degraded)
- NCEI CDO partially failed for San Antonio (retried, other cities succeeded)
- 12 opportunities with edge >= 5% found and paper-traded

**Paper trade state:**
- 21 total trades, 9 settled, 12 open
- PnL: +$0.32, ROI: 47%, Win rate: 11% (1 win out of 9)
- The single win (Boston SELL on 68-69F bucket, +$0.99) carried the entire book

**Route breakdown revealed a problem:**
- `emos`: 3 settled, 0 wins, PnL -$0.08, ROI -100%
- `legacy_unknown`: 6 settled, 1 win, PnL +$0.40, ROI 67%
- `raw_selective_fallback`: 0 settled yet

### 5. Critical Bug Fix: Strategy Policy Bypass

**Root cause:** `main.py` line 307 set `trade_opportunities = list(all_opportunities)` — when DeepSeek was disabled, ALL opportunities (anything with >= 5% edge) went straight to paper trading. The strategy policy filter (`min_abs_edge: 0.15`, `min_volume24hr: 2000`, etc.) was only applied inside the DeepSeek worker branch.

**Impact:** The scanner was paper-trading 12 opportunities per scan instead of the intended 0-3. Most were low-quality bucket market SELL positions at high prices that systematically lost.

**Fix:** Moved `filter_opportunities_for_policy()` to always run, regardless of DeepSeek status. The strategy policy is now the first gate. DeepSeek (if re-enabled) further narrows from there.

**Before:**
```
all_opportunities (12) -> paper_trading (12)  # NO FILTER
```

**After:**
```
all_opportunities (12) -> strategy_policy_filter (0-3) -> paper_trading (0-3)
```

### 6. Strategy Policy v3

Upgraded from v2 to v3 based on paper trade evidence:

**New filter: `allowed_settlement_rules: ["lte", "gt"]`**

This blocks `between_inclusive` (bucket) markets. Evidence:
- 8 of 9 settled bucket trades lost (1W/7L)
- The single bucket win was a high-variance NO bet
- Bucket markets have narrow 2F windows that are inherently low-probability
- The scanner was taking large SELL positions at high prices (avg entry $0.28) that lost frequently
- Threshold markets (`<84F`, `>82F`) better match the calibration model's distributional edge

Also implemented `allowed_settlement_rules` filter in `strategy_policy.py` with 2 new tests.

### 7. HRRR Reliability Improvements

The HRRR fetch timed out after ~10 minutes during our scan (S3 bucket read timeout). Added three configurable safety mechanisms:

1. **Overall timeout** (`hrrr_timeout_seconds`, default 300s): Hard deadline for the entire fetch. After 5 minutes, returns partial results instead of hanging indefinitely.

2. **Per-hour retries** (`hrrr_max_retries_per_hour`, default 2; `hrrr_retry_delay_seconds`, default 5s): Each lead hour gets up to 2 retries on transient failures before moving to the next hour.

3. **Consecutive failure abort**: If 3 lead hours fail in a row, remaining hours are skipped entirely (S3 is probably down).

All three are configurable in `config.json` and default to sensible values.

### 8. Constants to Config

Promoted hardcoded module-level constants to configurable values:

| Constant | Module | Config Key | Default |
|----------|--------|-----------|---------|
| `_ENSEMBLE_SIGMA_FLOOR_F` | matcher.py | `ensemble_sigma_floor_f` | 1.0 |
| `_ENSEMBLE_SIGMA_CAP_F` | matcher.py | `ensemble_sigma_cap_f` | 6.0 |
| `_MIN_SPREAD_F` | calibration.py | `emos_min_spread_f` | 1.0 |
| `SELECTIVE_RAW_FALLBACK_TARGETS` | calibration.py | `selective_raw_fallback_targets` | 5 pairs |

Implementation: Added `configure_sigma_bounds()` and `configure_calibration_constants()` functions that `main.py` calls at startup with config values. The module-level constants remain as defaults for backward compatibility.

### 9. Evaluation Framework Improvements

`evaluate_calibration.py` was already comprehensive (1480 lines) but output-only JSON to stdout. Added:

**Human-readable text summary:**
```
========================================================================
  Calibration Evaluation Report
  Window: 400 days, Holdout: 30 days, Cities: 20
========================================================================
  Policy: selective_raw_fallback
    Pairs evaluated: 40/40
    Temperature MAE: raw 3.519F -> policy 2.858F (better)
    Probability Brier: raw 0.2133 -> policy 0.1384 (better)
    Proxy trade selection: 8611 examples, 82.2% hit rate, PnL proxy +2776.5
    Temperature: 29 helped, 6 hurt
    Probability: 40 helped, 0 hurt
```

Plus per-city table, targeted fallback pair comparison, and net regression warnings.

**`--output` flag:** Save JSON reports to disk for historical tracking:
```
python evaluate_calibration.py --days 400 --holdout-days 30 --output data/evaluation_reports/eval_2026-04-16.json
```

**`--compare` flag:** Diff two saved reports:
```
python evaluate_calibration.py --compare eval_A.json eval_B.json
```
Shows side-by-side metrics with deltas for each policy, per-city Brier delta changes, and new/removed pairs.

**Calibration reliability bins:** Added `_reliability_bins()` function that bins predictions into 10 equal-width probability buckets with mean predicted vs. mean observed rates. Enables future calibration curve plotting.

**Ran full evaluation:** 40/40 city-market pairs evaluated. Selective fallback policy outperforms broad on every metric. All 5 targeted fallback pairs show improved Brier scores vs. the broad policy.

### 10. Phase 5: GOES Scaffold

Created `src/fetch_goes.py` with:
- Full API signatures matching the UPGRADE_PLAN.md spec
- `fetch_cloud_fraction()` and `fetch_cloud_fraction_multi()` (stubs returning None until goes2go installed)
- `compute_cloud_adjustment_f()` fully implemented — scales adjustment by cloud fraction and peak heating hour proximity
- `get_goes_forecast_adjustment()` wrapper with source labeling
- 12 tests covering all the math

The scaffold is ready for implementation once `goes2go` is added to requirements.

### 11. Polymarket Enablement Documentation

Polymarket integration is fully coded but disabled (`enable_polymarket: false`). Documented the enablement path in `strategy/CLAUDE.md`:
1. Set `enable_polymarket: true`
2. Add `"polymarket"` to strategy policy sources
3. Note `allowed_settlement_rules` compatibility consideration
4. Recommend 2+ weeks paper-trading alongside Kalshi before giving it real weight

### 12. VPS Deployment

Deployed the Weather scanner to the Hetzner VPS (`95.216.159.10`) alongside the existing Polymarket copy-trader bot.

**What was deployed:**
- Source code, calibration models, forecast archive, station actuals, paper trade ledger
- Python 3.12 venv with all dependencies (including Herbie for HRRR)
- Systemd timer units for scheduled scanning (every 30 min) and settlement (daily 6am UTC)

**VPS layout:**
```
/opt/polymarket/    <- existing bot (untouched)
/opt/weather/       <- Weather scanner (new, isolated venv)
  .venv/            <- 617 MB
  data/             <- 23 MB (models, archive, ledger)
  main.py, src/     <- application code
  deploy/           <- systemd units, setup scripts
```

**First live run on VPS:**
- 20/20 forecasts in 2 seconds (faster than local)
- 22 Kalshi markets found (authenticated)
- HRRR fetched successfully, BallTree cached
- 80 calibration models loaded
- Strategy policy correctly filtered to 0 trades (v3 policy: threshold-only + 15% edge + $2k volume)

**Disk usage:** 828 MB total (30 GB available on VPS)

**Deploy artifacts created:**
| File | Purpose |
|------|---------|
| `deploy/deploy_to_vps.ps1` | PowerShell script: rsync code + run setup |
| `deploy/vps_setup.sh` | Server-side: creates dirs, builds venv, installs systemd |
| `deploy/weather-scanner.service` | Oneshot: `main.py --once` |
| `deploy/weather-scanner.timer` | Every 30 minutes |
| `deploy/weather-settle.service` | Oneshot: `main.py --settle-paper-trades` |
| `deploy/weather-settle.timer` | Daily 6am UTC |
| `deploy/README.md` | Full deploy guide with monitoring commands |
| `deploy/DASHBOARD_INTEGRATION.md` | Brief for adding Weather page to Polymarket dashboard |

### 13. Dashboard Integration Brief

Wrote a detailed spec for adding a Weather monitoring page to the existing Polymarket Flask dashboard. Includes:
- Exact CSS design system variables and component classes to match
- Sidebar nav button HTML
- 4 card layouts (route breakdown, open positions, settled trades, scanner log)
- Complete Flask `/api/weather` route code
- JavaScript fetch/render pattern matching existing auto-refresh cycle
- `summary.json` schema and `ledger.csv` column reference
- File paths on VPS

### 14. Cleanup

- Disabled DeepSeek worker in `config.example.json`
- Removed stale git worktree `Weather__wt__task_0001-plan-the-next-calibration-upgrad` (orphaned from .orchestrator)
- Deleted the orphaned branch `orchestrator/task_0001-plan-the-next-calibration-upgrad`

---

## Key Metrics

| Metric | Before | After |
|--------|--------|-------|
| Test files | 6 | 14 |
| Test count | 70 | 167 |
| Test runtime | ~2s | ~2s |
| CLAUDE.md docs | 1 (root) | 8 (root + 4 subdirs + 3 rules) |
| Config keys | 33 | 41 |
| Strategy policy version | v2 | v3 |
| VPS deployment | None | Live, scanning every 30 min |
| Evaluation output | JSON only | JSON + text summary + save + compare |

---

## Architecture Decisions

### Why block bucket markets entirely?

The data is clear: 8/9 settled bucket trades lost. But the deeper reason is structural. Bucket markets (e.g., "will the high be 84-85F?") have a ~2F window that any temperature can fall in or out of easily. The calibration model outputs a normal distribution over temperature — it's good at saying "the high will be around 87F with sigma 1.3F" but it can't reliably predict whether the actual will land in a specific 2F bin. Threshold markets ("will it be above 84F?") directly map to the CDF of that distribution, which is what the model is actually calibrated to predict.

### Why always apply strategy policy?

The previous behavior was a design oversight, not intentional. The strategy policy was written to be the mechanical filter between scans and paper trading, but it was only wired into the DeepSeek branch. With DeepSeek disabled, the policy was dead code. The fix ensures the policy is always the first gate, which matches the documented intent in `strategy/CLAUDE.md`.

### Why systemd timers instead of cron?

The Polymarket bot uses cron, but systemd timers are better for this use case:
- `Persistent=true` means missed runs (e.g., during VPS restart) are caught up
- `RandomizedDelaySec` prevents exact-minute load spikes
- `journalctl -u weather-scanner` gives proper structured logging
- Timer status is visible via `systemctl list-timers`
- Easier to stop/start/disable without editing crontab

### Why not share the Polymarket venv?

Dependency isolation. The Weather scanner uses herbie-data, cfgrib, xarray, scikit-learn — heavy scientific packages that could conflict with the Polymarket bot's dependencies. Separate venvs cost ~600 MB but prevent any risk of breaking the production copy-trader.

---

## Current State

### What's Working
- Scanner runs every 30 min on VPS via systemd timer
- Settlement runs daily at 6am UTC
- Calibration models loaded (80 models for 20 cities x 2 types x 2 models)
- Strategy policy v3 correctly filtering to threshold-only, high-conviction trades
- HRRR blending available with timeout/retry protection
- Full test suite passing (167 tests)

### What to Watch
- **Paper trade accumulation:** The v3 policy is tight. It may take several days of scanning before enough threshold markets clear the 15% edge + $2k volume bar to get meaningful trade flow. This is by design — we want quality over quantity.
- **HRRR cache growth:** With HRRR enabled, the cache at `/opt/weather/data/hrrr_cache/` will grow ~200 MB/week. May want a periodic cleanup cron.
- **NCEI CDO reliability:** Station actual refresh had partial failures. The fallback to NWS CLI archive works but is slower. If CDO stays unreliable, consider adjusting retry parameters.

### What's Next
1. **Accumulate data** — Let the scanner run for 2-3 weeks under v3 policy. Need 30+ settled trades to properly evaluate.
2. **Retrain models** — Run `train_calibration.py` after another week of forecast archive data accumulates.
3. **Re-evaluate** — Run `evaluate_calibration.py --output` and `--compare` against today's baseline.
4. **Dashboard** — Hand `DASHBOARD_INTEGRATION.md` to whoever works on the HTML to add the Weather monitoring page.
5. **Consider loosening policy** — If the scanner is too quiet (0 trades most scans), consider lowering `min_volume24hr` from 2000 to 1000, or dropping `min_abs_edge` from 0.15 to 0.12.
6. **GOES implementation** — When ready, install `goes2go`, uncomment in requirements.txt, and implement `fetch_cloud_fraction()` in the scaffold.

---

## Files Changed/Created

### New Files (22)
```
src/CLAUDE.md
tests/CLAUDE.md
data/CLAUDE.md
strategy/CLAUDE.md
.claude/rules/calibration.md
.claude/rules/paper-trading.md
.claude/rules/secrets.md
tests/test_fetch_hrrr.py
tests/test_fetch_polymarket.py
tests/test_deepseek_failure_modes.py
tests/test_analyze.py
tests/test_fetch_kalshi.py
tests/test_fetch_goes.py
src/fetch_goes.py
data/evaluation_reports/eval_2026-04-16.json
deploy/deploy_to_vps.ps1
deploy/vps_setup.sh
deploy/weather-scanner.service
deploy/weather-scanner.timer
deploy/weather-settle.service
deploy/weather-settle.timer
deploy/README.md
deploy/DASHBOARD_INTEGRATION.md
.orchestrator/docs/worklogs/worklog_session_2026-04-16.md
```

### Modified Files (8)
```
CLAUDE.md                    (added Documentation Map)
config.example.json          (disabled DeepSeek, added HRRR/sigma/GOES config keys)
main.py                      (always apply strategy policy, wire config constants, HRRR config passthrough)
src/matcher.py               (added configure_sigma_bounds())
src/calibration.py           (added configure_calibration_constants())
src/fetch_hrrr.py            (timeout, retry, consecutive failure abort)
src/strategy_policy.py       (allowed_settlement_rules filter)
strategy/strategy_policy.json (v2 -> v3, added allowed_settlement_rules + rationale)
evaluate_calibration.py      (--output, --compare, text summary, reliability bins)
tests/test_strategy_policy.py (2 new tests for settlement rules)
```
