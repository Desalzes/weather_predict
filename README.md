# Weather Predict

Short-horizon weather prediction and market scanning for Kalshi and Polymarket weather contracts.

## What the project does

- Fetches Open-Meteo forecasts for a 20-city watchlist.
- Optionally pulls ensemble spread to estimate forecast uncertainty.
- Scans live Kalshi and Polymarket weather markets.
- Converts forecast distributions into contract probabilities and highlights edge.
- Archives forecast snapshots and station truth for calibration.
- Trains per-city calibration models from historical forecast-vs-actual data.
- Blends in same-day HRRR guidance for near-settlement markets.

## Repo layout

```text
.
+-- main.py                    # main scanner entrypoint
+-- backfill_training_data.py  # backfill actuals + archived forecasts
+-- train_calibration.py       # train EMOS + isotonic calibration models
+-- config.example.json        # safe template for local config
+-- stations.json              # city -> station metadata map
+-- requirements.txt
+-- UPGRADE_PLAN.md
+-- data/
|   +-- calibration_models/    # trained model artifacts
|   +-- forecast_archive/      # archived forecast snapshots
|   +-- paper_trades/          # paper-trade ledger + settlement summary
|   +-- station_actuals/       # historical station truth
+-- src/
    +-- analyze.py             # confidence/event analytics helpers
    +-- calibration.py         # EMOS + isotonic training/runtime logic
    +-- config.py              # config resolution and secret loading
    +-- fetch_forecasts.py     # Open-Meteo forecast + previous-run history
    +-- fetch_hrrr.py          # HRRR fetch + batching/cache logic
    +-- fetch_kalshi.py        # Kalshi market discovery
    +-- fetch_polymarket.py    # Polymarket market discovery
    +-- kalshi_client.py       # authenticated Kalshi trading client
    +-- matcher.py             # forecast -> contract probability matching
    +-- station_truth.py       # NWS/NCEI truth ingestion + training joins
```

## Local setup

1. Create a Python 3.12 virtual environment.
2. Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Create a local config:

```powershell
Copy-Item config.example.json config.json
```

4. Fill in local-only secrets in `config.json` or environment variables:
   - `kalshi_api_key_id`
   - `ncei_api_token`
   - `kalshi_private_key_path`
   - `DEEPSEEK_API_KEY`

5. Keep the RSA private key file out of git. The default local path is `api-credentials.txt`.

## Common commands

```powershell
.\.venv\Scripts\python.exe main.py --once --kalshi-only
.\.venv\Scripts\python.exe main.py --once
.\.venv\Scripts\python.exe main.py --settle-paper-trades
.\.venv\Scripts\python.exe backfill_training_data.py --days 365
.\.venv\Scripts\python.exe train_calibration.py
.\.venv\Scripts\python.exe evaluate_calibration.py --days 400 --holdout-days 30
```

The `evaluate_calibration.py --days 400 --holdout-days 30` command is the exact rerun path for the current regression diagnostics and selective-fallback benchmark. It prints deterministic chronological split results for each city and `high`/`low` market type, plus policy-level summaries under `summary.broad_emos_isotonic` and `summary.selective_raw_fallback`, targeted-pair deltas under `comparison.targeted_pairs`, net-new regression checks, and an explicitly labeled even-odds proxy trade-selection readout.

Each city-market result now also carries `temperature_diagnostics.regime_diagnostics.lead_time_regime` when the retained holdout rows span both lead-time buckets: `day_ahead` (`forecast_lead_days == 1`) and `multi_day` (`forecast_lead_days >= 2`). Pairs that do not have enough usable lead-time diversity report an explicit per-pair insufficiency reason in both the pair diagnostics and the report-level limitation summary instead of the evaluator claiming the dimension is universally unavailable.

The live scanner now mirrors that benchmarked routing policy. If a fitted EMOS model exists, it is bypassed only for `Boston low`, `Minneapolis low`, `Philadelphia low`, `New Orleans high`, and `San Francisco low`, and those opportunities are labeled with `forecast_calibration_source=raw_selective_fallback`. The existing isotonic gate is unchanged, so near-settlement HRRR-blended flows still skip probability calibration.

## Baseline paper-trading flow

1. Local `config.json` values override `config.example.json`, but any missing top-level keys now inherit the committed example defaults. Older local configs therefore pick up the paper-trading defaults automatically unless they explicitly set `enable_paper_trading` to `false`.
2. Paper-trade fees are controlled by two top-level config keys:
   - `paper_trade_entry_fee_per_contract`: flat fee added once per contract when the paper position is opened
   - `paper_trade_settlement_fee_per_contract`: flat fee deducted once per contract when the paper trade settles
   Both default to `0.0` in `config.example.json`, so older local configs keep the prior zero-fee behavior until you set explicit values.
3. Run a scan:

```powershell
.\.venv\Scripts\python.exe main.py --once
```

4. New paper positions are written to `data/paper_trades/ledger.csv`. Each row now records the fee model (`flat_fee_per_contract_v1`), the per-contract fee assumptions applied, and the derived `entry_fee`, `settlement_fee`, and `total_fees` values used later for net profitability reporting. Rows also preserve the forecast-routing metadata needed to attribute forward results, including `forecast_calibration_source`, `probability_calibration_source`, `forecast_blend_source`, `hours_to_settlement`, `open_meteo_forecast_value_f`, `raw_forecast_value_f`, and `raw_probability`. Legacy ledgers without those columns remain readable; when they are rewritten, unknown legacy route metadata is backfilled deterministically as `legacy_unknown` instead of being guessed. The scanner only records one open position per unique market side until that position settles.
5. Settle and score the ledger:

```powershell
.\.venv\Scripts\python.exe main.py --settle-paper-trades
```

The settlement pass now attempts a bounded station-actual refresh for only the cities and open-trade dates that are eligible to settle, ending at `min(yesterday, --as-of-date)` when a cutoff is supplied. It uses NCEI CDO first when `ncei_api_token` is configured and otherwise uses the NWS CLI archive first. If the first pass still leaves required city/date truth missing, the refresh result retries only those still-missing windows against the alternate official source when one is available. The ledger is then updated in place and `data/paper_trades/summary.json` is rewritten. This path is paper-only and never places live orders.

Net paper-trade profitability is now computed with explicit deterministic fees:

- `entry_cost = (entry_price * contracts) + entry_fee`
- `pnl = payout - (entry_price * contracts) - total_fees`
- `roi = pnl / entry_cost`

`summary.json` reports `total_pnl` and `roi` on that net basis, and also includes `total_fees` so future strategy comparisons can be reproduced against the same fee assumptions. It now also includes `forecast_calibration_source_breakdown`, which summarizes total/open/settled trades plus settled PnL, ROI, fees, and win rate for each persisted forecast route (for example `emos`, `raw`, `raw_selective_fallback`, or `legacy_unknown` for older rows that predate route persistence).

When the refresh transport completes but the needed station-actual rows are still absent locally, the summary now exposes that as an explicit blocker instead of looking like a clean no-settlement run. Operator-facing fields include:

- `settlement_cutoff_date`: the bounded date through which settlement truth was expected for the run
- `eligible_open_trade_count`: remaining open trades whose market dates are within that cutoff
- `missing_required_truth_blocker`: `true` when realized PnL is still blocked by absent local truth rows
- `missing_required_truth_trade_count` / `missing_required_truth_date_count`: blocked trade count and distinct city/date requirements still missing
- `missing_required_truth_dates`: per-city, per-date missing requirements with `actual_field` and `latest_local_date`
- `missing_required_truth_city_windows`: grouped city/date windows showing the missing range and the latest local station-actual date on disk

The refresh result itself also reports the source order used for that run:

- `refresh_source`: the first-pass source attempted for the bounded refresh
- `fallback_attempted`: whether a second pass ran for still-missing truth windows
- `fallback_refresh_source`: the alternate source reserved for that second pass when available
- `fallback_city_windows`: the exact still-missing city/date windows retried on the fallback pass

## DeepSeek worker

The scanner now supports an intermittent DeepSeek review worker for trade selection. The committed defaults turn it on for this repo in `paper_gate` mode, which means:

- the full market scan still runs every pass
- DeepSeek reviews the top candidate opportunities on the first eligible scan, then again every `deepseek_review_interval_scans`
- when a DeepSeek review runs, paper trades are limited to its explicit `approve` decisions
- between scheduled DeepSeek reviews, the current `strategy/strategy_policy.json` filters still drive paper-trade selection so the paper trader does not go idle just to save tokens
- review artifacts are written to `data/deepseek_worker/reviews/`
- review state is persisted in `data/deepseek_worker/state.json`

Relevant config keys:

- `enable_deepseek_worker`
- `deepseek_trade_mode`: `review_only` or `paper_gate`
- `strategy_policy_path`
- `deepseek_model`
- `deepseek_review_interval_scans`
- `deepseek_max_opportunities_per_review`
- `deepseek_min_abs_edge`
- `deepseek_pull_kalshi_market_context`

The worker uses the official DeepSeek chat-completions API via `DEEPSEEK_API_KEY` from the environment. The committed strategy contract is [strategy_policy.json](/Users/desal/Desktop/Projects/_Betting_Markets/weather/strategy/strategy_policy.json); `codex_loop` can rewrite that file over time, and the DeepSeek paper gate consumes it on each scan before deciding which paper trades to take. This path remains paper-only.

The first evidence-backed policy revision landed on April 5, 2026. The fallback policy between DeepSeek review runs now requires at least `15%` absolute edge, at least `2000` in 24-hour market volume, caps fallback candidates at `3`, and temporarily blocks `Washington DC` while the loop gathers enough settled paper evidence to revisit that city.

## Git safety

The repository intentionally ignores:

- `config.json`
- `api-credentials.txt`
- `.venv/`
- `data/hrrr_cache/`
- Python cache files and IDE folders

That keeps local API credentials and bulky runtime cache files out of GitHub while still allowing the repo to ship the code, station map, archived training data, and trained calibration artifacts.
