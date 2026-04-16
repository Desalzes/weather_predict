# tests/ — Test Suite

## Run

```powershell
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m pytest tests/test_training_regressions.py -v   # single file
```

## Test File Map

| File | Covers |
|------|--------|
| `test_training_regressions.py` | Calibration pipeline: archive parsing, training-set builds, chronological holdout splits, EMOS/isotonic metrics, selective fallback routing, lead-time regime diagnostics |
| `test_paper_trading.py` | Trade lifecycle: logging with routing metadata, deduplication, settlement against station truth, fee accounting, ledger migrations, truth-blocker reporting, PnL breakdown by route |
| `test_deepseek_worker.py` | DeepSeek review gateway: trigger interval, JSON response parsing, approve/watch/reject decisions, state persistence, graceful skip when API key missing |
| `test_strategy_policy.py` | Policy filtering: min_abs_edge, min_volume, max_candidates, settlement horizon, allowed/blocked cities, source filtering |
| `test_config_loading.py` | Config inheritance: example defaults, local overrides, explicit disable, inherited paper-trade paths and fees |
| `test_main_deepseek_integration.py` | End-to-end: run_scan() with DeepSeek paper_gate mode, fallback to policy-only between reviews, opportunity ID format |
| `test_fetch_hrrr.py` | HRRR module: temp conversion, UTC handling, init time resolution, location normalization, in-memory cache hit/miss, dataset extraction, high/low extraction, Herbie import failure |
| `test_fetch_polymarket.py` | Polymarket: market type parsing, city alias resolution, outcome range parsing, cache TTL, API failure handling, deduplication, price conversion |
| `test_deepseek_failure_modes.py` | DeepSeek failure paths: timeout, malformed JSON, empty choices, missing decisions, HTTP 429/500, connection error, disabled worker, recovery after failure |
| `test_analyze.py` | Analytics: ensemble spread stats (multi-member, single, empty), weather event detection (rain, wind, temp swing, custom thresholds), confidence scoring (horizon penalty, ensemble penalty) |
| `test_fetch_kalshi.py` | Kalshi: ticker parsing, city codes, weather prefixes, unauthenticated fetch, pagination, cache, market filtering, grouping by city |
| `test_fetch_goes.py` | GOES scaffold: cloud adjustment math (peak vs off-peak, scaling, clamping), forecast adjustment wrapper, scaffold returns None |

## Conventions

- **Chronological holdout splits** — test data is split by date, not randomly,
  to match production evaluation.
- **No mocking of calibration math** — EMOS and isotonic logic runs on real
  (small) datasets in tests; only external API calls are mocked.
- **Deterministic fee accounting** — fee tests verify exact arithmetic, not
  approximate ranges.
- **Legacy compatibility** — tests verify that ledgers missing newer columns are
  migrated correctly with `legacy_unknown` backfill.

## Known Coverage Gaps

- `fetch_hrrr.py` — Herbie integration test (requires heavy deps) not covered;
  unit-level helpers and cache logic are tested
- `fetch_kalshi.py` — authenticated client and market discovery not tested
- `logging_setup.py` — untested (low risk)
