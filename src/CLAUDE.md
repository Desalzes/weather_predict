# src/ — Application Logic

## Data Flow

```
fetch_forecasts / fetch_hrrr  →  calibration (EMOS)  →  matcher (probability)
fetch_kalshi / fetch_polymarket                          ↓
                                                    paper_trading  →  deepseek_worker (gate)
                                                         ↑
                                                    station_truth (settlement actuals)
```

## Module Map

| Module | Responsibility |
|--------|----------------|
| `config.py` | Config loading with inheritance (`config.json` overrides `config.example.json`) |
| `logging_setup.py` | RotatingFileHandler (1 MB, 5 backups) to `logs/weather.log` |
| `fetch_forecasts.py` | Open-Meteo deterministic + ensemble + previous-runs APIs |
| `fetch_hrrr.py` | HRRR 2-m temp via Herbie (optional dep, degrades gracefully) |
| `fetch_kalshi.py` | Kalshi market discovery with 5-min cache, city-code mapping |
| `fetch_polymarket.py` | Polymarket Gamma API with 5-min cache, city-alias mapping |
| `fetch_precipitation.py` | Open-Meteo precip (deterministic + ensemble) + HRRR APCP extraction |
| `kalshi_client.py` | Authenticated Kalshi REST client (RSA-PSS SHA-256 signatures) |
| `matcher.py` | Normal-CDF probability for threshold/bucket markets, edge computation |
| `rain_calibration.py` | Logistic + Isotonic rain-probability calibrators + RainCalibrationManager |
| `rain_matcher.py` | Parse KXRAIN binary outcomes, compute probability & edge, mirror temp opportunity shape |
| `calibration.py` | EMOS (linear bias correction) + isotonic (probability recalibration) |
| `station_truth.py` | NWS CLI + NCEI CDO truth ingestion, training-set joins |
| `paper_trading.py` | Ledger logging, deterministic fee accounting, settlement |
| `deepseek_worker.py` | LLM trade-gate via DeepSeek chat-completions API |
| `strategy_policy.py` | Policy JSON loading, opportunity filtering, prompt serialization |
| `analyze.py` | Ensemble spread stats, weather-event detection thresholds |
| `fetch_goes.py` | GOES satellite cloud-fraction fetcher (Phase 5 scaffold, not yet implemented) |

## Key Constants

| Constant | File | Value | Purpose |
|----------|------|-------|---------|
| `_ENSEMBLE_SIGMA_FLOOR_F` | matcher.py | 1.0 | Min ensemble σ (°F) |
| `_ENSEMBLE_SIGMA_CAP_F` | matcher.py | 6.0 | Max ensemble σ (°F) |
| `_MIN_SPREAD_F` | calibration.py | 1.0 | EMOS spread floor (°F) |
| `_MIN_PROB` / `_MAX_PROB` | calibration.py | 0.001 / 0.999 | Probability clamps |
| `_SELECTIVE_RAW_FALLBACK_TARGETS` | calibration.py | 5 city-market pairs | Bypass EMOS for empirically weaker pairs |
| `PAPER_TRADE_FEE_MODEL` | paper_trading.py | `flat_fee_per_contract_v1` | Fee model identifier |
| `_CACHE_TTL` | fetch_kalshi.py | 300 s | Market list cache lifetime |
| `_CACHE_TTL_SECONDS` | fetch_polymarket.py | 300 s | Market list cache lifetime |

## Rules

- Changes to `matcher.py`, `calibration.py`, `paper_trading.py`, or
  `station_truth.py` require inspecting and updating the corresponding tests
  before merging.
- All probability paths must respect the `_MIN_PROB`/`_MAX_PROB` clamps.
- HRRR integration is optional — failures must degrade gracefully, never crash.
- Market API clients must respect rate limits and caching.
