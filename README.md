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

5. Keep the RSA private key file out of git. The default local path is `api-credentials.txt`.

## Common commands

```powershell
.\.venv\Scripts\python.exe main.py --once --kalshi-only
.\.venv\Scripts\python.exe backfill_training_data.py --days 365
.\.venv\Scripts\python.exe train_calibration.py
```

## Git safety

The repository intentionally ignores:

- `config.json`
- `api-credentials.txt`
- `.venv/`
- `data/hrrr_cache/`
- Python cache files and IDE folders

That keeps local API credentials and bulky runtime cache files out of GitHub while still allowing the repo to ship the code, station map, archived training data, and trained calibration artifacts.
