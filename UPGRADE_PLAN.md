# Weather Signals Upgrade Plan

## Current State

The scanner fetches Open-Meteo hourly forecasts for 20 cities, discovers Kalshi weather markets, models temperature as `N(forecast, 2.0°F)`, and reports edge vs market price. The `uncertainty_std_f=2.0` is a static guess — no calibration, no ensemble input, no station truth, no same-day adjustment.

## Upgrade Stack (ordered by ROI)

```
1. Open-Meteo Ensemble API     → member-based σ replaces hardcoded 2°F
2. NWS Station Truth (CLI/CDO) → train calibration, measure real accuracy
3. Station-Specific Calibration → EMOS / isotonic regression per city
4. HRRR via Herbie             → same-day local adjustment (3km, hourly)
5. GOES via goes2go            → cloud/sun adjustment for daytime heating
```

---

## Phase 1: Ensemble-Based Uncertainty (replaces hardcoded σ)

**Goal:** Replace `uncertainty_std_f=2.0` with per-city, per-forecast dynamic σ from ensemble member spread.

### Files to change

| File | Change |
|---|---|
| `fetch_forecasts.py` | New `fetch_ensemble_multi_location()` that calls ensemble API for all 20 cities. Return member temps alongside deterministic forecast. |
| `matcher.py` | `match_kalshi_markets()` accepts optional `ensemble_data` dict. If present, compute σ from ensemble member max/min temps for the market date instead of using config `uncertainty_std_f`. |
| `main.py` | After fetching deterministic forecasts, also fetch ensemble. Pass both to matcher. |
| `config.json` | Add `"enable_ensemble": true`, keep `uncertainty_std_f` as fallback when ensemble unavailable. |

### Implementation Detail

```python
# In fetch_forecasts.py
def fetch_ensemble_multi_location(locations, hours=72):
    """Fetch ensemble forecasts for all locations. Returns dict[city_name] -> ensemble_data."""
    # Uses existing _ENSEMBLE_URL
    # Returns raw ensemble API response per city

# In matcher.py — new helper
def _ensemble_sigma_for_date(ensemble_data, target_date, market_type):
    """Compute σ from ensemble member spread for a specific date.

    1. Extract all member temps for target_date hours
    2. Per member: compute max (high market) or min (low market)
    3. σ = std(member_extremes)
    4. Floor at 1.0°F, cap at 6.0°F
    """
```

**Open-Meteo ensemble API** returns ~50 members (icon_seamless model). Each member has its own `temperature_2m_member01`, `temperature_2m_member02`, etc. The spread of daily-max across members directly measures forecast uncertainty for high-temp markets.

### Why this matters

A market where ensemble members agree on 72±0.5°F is very different from one where members range 68-76°F. The hardcoded 2°F misses both cases — overconfident when spread is wide, underconfident when spread is narrow.

---

## Phase 2: NWS Station Truth for Training & Backtesting

**Goal:** Collect actual daily high/low temps from the stations Kalshi uses for settlement, enabling calibration training and edge-resolution backtesting.

### Data Sources

1. **NWS CLI (Climatological Report)** — The official daily report Kalshi settles on.
   - URL pattern: `https://forecast.weather.gov/product.php?site=NWS&issuedby={STATION}&product=CLI`
   - Free, no key. Published daily after midnight local time.
   - Parse the text report for "MAXIMUM TEMPERATURE" and "MINIMUM TEMPERATURE" lines.

2. **NCEI CDO (Climate Data Online)** — Free historical daily data for backtesting.
   - `https://www.ncds.noaa.gov/cdo-web/api/v2/data`
   - Requires free API token (sign up at ncdc.noaa.gov).
   - Dataset: `GHCND`, data types: `TMAX`, `TMIN`.
   - Rate limit: 5 req/sec, 10k records/request.

### Files to create

| File | Purpose |
|---|---|
| `station_truth.py` | Fetch and parse NWS CLI reports + NCEI CDO historical data. Map Kalshi city codes → ICAO/GHCND station IDs. |
| `stations.json` | Station mapping: city name → `{icao, ghcnd_id, cli_station, lat, lon}` for all 20 cities. |

### Station Mapping (Kalshi cities → NWS stations)

```json
{
  "New York":       {"icao": "KNYC", "ghcnd": "USW00094728", "cli": "NYC"},
  "Los Angeles":    {"icao": "KLAX", "ghcnd": "USW00023174", "cli": "LAX"},
  "Chicago":        {"icao": "KORD", "ghcnd": "USW00094846", "cli": "ORD"},
  "Miami":          {"icao": "KMIA", "ghcnd": "USW00012839", "cli": "MIA"},
  "Houston":        {"icao": "KIAH", "ghcnd": "USW00012960", "cli": "IAH"},
  "Phoenix":        {"icao": "KPHX", "ghcnd": "USW00023183", "cli": "PHX"},
  "Denver":         {"icao": "KDEN", "ghcnd": "USW00003017", "cli": "DEN"},
  "Seattle":        {"icao": "KSEA", "ghcnd": "USW00024233", "cli": "SEA"},
  "Philadelphia":   {"icao": "KPHL", "ghcnd": "USW00013739", "cli": "PHL"},
  "Dallas":         {"icao": "KDFW", "ghcnd": "USW00003927", "cli": "DFW"},
  "Austin":         {"icao": "KAUS", "ghcnd": "USW00013904", "cli": "AUS"},
  "Boston":         {"icao": "KBOS", "ghcnd": "USW00014739", "cli": "BOS"},
  "Atlanta":        {"icao": "KATL", "ghcnd": "USW00013874", "cli": "ATL"},
  "Washington DC":  {"icao": "KDCA", "ghcnd": "USW00013743", "cli": "DCA"},
  "San Francisco":  {"icao": "KSFO", "ghcnd": "USW00023234", "cli": "SFO"},
  "Las Vegas":      {"icao": "KLAS", "ghcnd": "USW00023169", "cli": "LAS"},
  "Minneapolis":    {"icao": "KMSP", "ghcnd": "USW00014922", "cli": "MSP"},
  "San Antonio":    {"icao": "KSAT", "ghcnd": "USW00012921", "cli": "SAT"},
  "New Orleans":    {"icao": "KMSY", "ghcnd": "USW00012916", "cli": "MSY"},
  "Oklahoma City":  {"icao": "KOKC", "ghcnd": "USW00013967", "cli": "OKC"}
}
```

### Key Functions

```python
# station_truth.py

def fetch_cli_report(station: str) -> dict:
    """Parse today's NWS CLI report. Returns {date, tmax_f, tmin_f, precip_in}."""

def fetch_historical_daily(ghcnd_id: str, start: str, end: str, token: str) -> pd.DataFrame:
    """Fetch GHCND daily TMAX/TMIN from NCEI CDO. Returns DataFrame[date, tmax_f, tmin_f]."""

def build_training_set(city: str, days: int = 90) -> pd.DataFrame:
    """Join last N days of Open-Meteo forecasts with station actuals.
    Columns: date, forecast_high_f, actual_high_f, forecast_low_f, actual_low_f, ensemble_std_f
    """
```

### Data Storage

```
weather_signals/
  data/
    station_actuals/          # Daily CSV per city: date,tmax_f,tmin_f
    forecast_archive/         # Archived daily forecasts for backtesting
    calibration_models/       # Pickled calibration models per city
```

---

## Phase 3: Station-Specific Calibration (EMOS + Isotonic)

**Goal:** Train per-city correction models that transform raw ensemble probabilities into calibrated probabilities tuned to each station's bias patterns.

### Approach: Two-Stage Calibration

**Stage A — Bias Correction (EMOS-lite)**

EMOS (Ensemble Model Output Statistics) fits a linear correction:
```
actual_temp = a + b * ensemble_mean + c * ensemble_spread
```

Per city, fit on 60-90 days of `(forecast_high, actual_high)` pairs from Phase 2 training set. This corrects systematic warm/cold bias (e.g., Phoenix forecasts consistently 2°F high in spring).

**Stage B — Probability Calibration (Isotonic Regression)**

After bias correction, the CDF probabilities still may not be well-calibrated. Isotonic regression maps `predicted_probability → observed_frequency` monotonically.

Train on binned forecast probabilities vs actual resolution rates from historical Kalshi settlements.

### Files to create

| File | Purpose |
|---|---|
| `calibration.py` | `EMOSCalibrator` class (bias correction) + `IsotonicCalibrator` class (probability calibration). Per-city model training, persistence, and inference. |
| `train_calibration.py` | CLI script: fetch historical data, train all city models, save to `data/calibration_models/`. |

### Key Classes

```python
# calibration.py

class EMOSCalibrator:
    """Per-city linear bias correction: actual = a + b*forecast + c*spread."""

    def __init__(self, city: str):
        self.city = city
        self.a = 0.0  # intercept
        self.b = 1.0  # slope on forecast
        self.c = 0.0  # slope on ensemble spread

    def fit(self, df: pd.DataFrame):
        """Fit on DataFrame with columns: forecast_high_f, actual_high_f, ensemble_std_f."""
        # sklearn LinearRegression on [forecast, spread] -> actual

    def correct(self, forecast_f: float, spread_f: float) -> float:
        """Return bias-corrected forecast temperature."""
        return self.a + self.b * forecast_f + self.c * spread_f

    def save(self, path): ...
    def load(self, path): ...


class IsotonicCalibrator:
    """Maps raw CDF probability -> calibrated probability via isotonic regression."""

    def __init__(self, city: str, market_type: str):
        self.city = city
        self.market_type = market_type  # "high" or "low"
        self._model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, raw_probs: np.ndarray, actual_outcomes: np.ndarray):
        """raw_probs: model P(yes), actual_outcomes: 1 if market resolved YES, 0 otherwise."""
        self._model.fit(raw_probs, actual_outcomes)

    def calibrate(self, raw_prob: float) -> float:
        return float(self._model.predict([[raw_prob]])[0])

    def save(self, path): ...
    def load(self, path): ...
```

### Integration into Matcher

```python
# In matcher.py — updated flow per market:

# 1. Get ensemble σ (Phase 1)
# 2. Get bias-corrected forecast (Phase 3A — EMOS)
corrected_value = emos_model.correct(forecast_value, ensemble_sigma)
# 3. Compute raw CDF probability with corrected forecast + ensemble σ
raw_prob = compute_temperature_probability(corrected_value, ...)
# 4. Calibrate probability (Phase 3B — Isotonic)
our_prob = isotonic_model.calibrate(raw_prob)
# 5. Edge = our_prob - market_price
```

### Evaluation Metrics

- **Brier Score**: `mean((predicted - actual)^2)` — lower is better
- **Reliability Diagram**: plot predicted prob vs observed frequency in 10 bins
- **CRPS** (via `xskillscore` or manual): proper scoring rule for probabilistic forecasts
- **Resolution Rate**: % of edges that resolved profitably

### Training Cadence

- Retrain weekly (Sunday night) on rolling 90-day window
- Log Brier score trend to detect model staleness
- Alert if any city's Brier score degrades >10% from baseline

---

## Phase 4: HRRR Same-Day Adjustment via Herbie

**Goal:** For markets settling today, replace/blend Open-Meteo with HRRR 3km data that assimilates radar every 15 minutes. HRRR is the most accurate short-range model for CONUS.

### Why HRRR

- 3km horizontal resolution (vs Open-Meteo's ~11km ICON/GFS blend)
- Hourly updates, radar assimilation every 15 min
- Best free model for 0-18h temperature forecasts over CONUS
- HRRR is what NWS forecasters themselves use for same-day

### Dependencies

```
herbie-data>=2024.1.0   # pip install herbie-data
cfgrib                   # GRIB2 reading (auto-installed with herbie)
```

### Files to create

| File | Purpose |
|---|---|
| `fetch_hrrr.py` | Fetch latest HRRR run for specific lat/lon points. Extract 2m temperature at station locations. |

### Key Functions

```python
# fetch_hrrr.py

from herbie import Herbie

def fetch_hrrr_point_forecast(lat: float, lon: float, fxx: int = 18) -> dict:
    """Fetch latest HRRR run, extract 2m temp timeseries for a point.

    Args:
        lat, lon: station coordinates
        fxx: forecast hours (default 18 = max HRRR range)

    Returns:
        {"times": [...], "temperature_2m_f": [...], "run_time": "2026-03-28T15:00Z"}
    """
    H = Herbie(model="hrrr", product="sfc", fxx=range(0, fxx+1))
    # xarray nearest-neighbor extract at (lat, lon)
    # Variable: TMP:2 m above ground

def fetch_hrrr_multi(locations: list, fxx: int = 18) -> dict:
    """Batch fetch HRRR for all cities. Returns dict[city_name] -> forecast."""

def get_hrrr_high_low(hrrr_data: dict, target_date: str) -> tuple[float, float]:
    """Extract predicted daily high and low from HRRR timeseries."""
```

### Integration into Matcher

```python
# In matcher.py — for same-day markets only:

hours_to_settlement = (market_close_time - now).total_seconds() / 3600
if hours_to_settlement < 18:  # within HRRR range
    hrrr_forecast = hrrr_data.get(city)
    if hrrr_forecast:
        # Blend: weight HRRR more as settlement approaches
        hrrr_weight = max(0.3, 1.0 - hours_to_settlement / 18.0)
        om_weight = 1.0 - hrrr_weight
        forecast_value = om_weight * om_forecast + hrrr_weight * hrrr_forecast
```

### HRRR Caveats

- HRRR data is ~2-4 GB per run on AWS (full CONUS grid). Herbie handles subsetting.
- First fetch for a run may be slow (~30s) as it downloads from AWS/NOMADS.
- HRRR only covers CONUS — all 20 cities are covered.
- Max forecast horizon: 18h (subhourly runs) or 48h (00z/12z runs).

---

## Phase 5: GOES Cloud/Sun Adjustment via goes2go

**Goal:** For same-day high-temp markets, detect whether clouds are blocking solar heating and adjust the forecast downward if needed. Clouds are the #1 reason daytime highs miss forecasts.

### Why GOES

- GOES-16/18 ABI scans CONUS every 5 min, mesoscale every 60 sec
- Cloud fraction directly determines whether modeled solar heating is realized
- A city forecast for 78°F under clear skies might only hit 73°F if unexpected clouds persist through peak heating (10am-3pm local)

### Dependencies

```
goes2go>=2022.1.0   # pip install goes2go
```

### Files to create

| File | Purpose |
|---|---|
| `fetch_goes.py` | Fetch latest GOES ABI CONUS imagery. Extract cloud fraction for city bounding boxes. |

### Key Functions

```python
# fetch_goes.py

from goes2go import GOES

def fetch_cloud_fraction(lat: float, lon: float, radius_km: float = 25) -> dict:
    """Fetch latest GOES ABI data and compute cloud fraction near a point.

    Returns:
        {
            "cloud_fraction": 0.35,      # 0-1, fraction of pixels classified as cloud
            "clear_sky_fraction": 0.65,
            "scan_time": "2026-03-28T16:45Z",
            "channel": "C13"             # IR window channel
        }
    """

def fetch_cloud_trend(lat: float, lon: float, hours_back: int = 3) -> list[dict]:
    """Fetch cloud fraction trend over last N hours. Useful for detecting
    whether clouds are building or clearing during peak heating window."""

def solar_heating_adjustment(
    forecast_high_f: float,
    cloud_fraction: float,
    cloud_trend: list[dict],
    local_hour: int,
) -> float:
    """Adjust forecast high based on cloud conditions.

    Heuristic:
    - If cloud_fraction > 0.7 during 10am-2pm local: reduce forecast by 2-4°F
    - If clearing trend (decreasing clouds): smaller reduction
    - If building trend: larger reduction
    - After 3pm local: minimal adjustment (peak heating already occurred)
    """
```

### Integration into Matcher

```python
# In matcher.py — for same-day HIGH temp markets only, during heating window:

if market_type == "high" and 14 <= utc_hour <= 22:  # ~10am-6pm US time
    cloud_data = goes_data.get(city)
    if cloud_data and cloud_data["cloud_fraction"] > 0.5:
        adjustment = solar_heating_adjustment(
            forecast_value, cloud_data["cloud_fraction"],
            cloud_data.get("trend", []), local_hour
        )
        forecast_value += adjustment  # negative adjustment reduces expected high
```

### GOES Caveats

- goes2go downloads from AWS S3 (NOAA Open Data). No auth needed.
- Band 13 (10.3μm IR window) is best for cloud detection — works day and night.
- Band 2 (0.64μm visible) is higher resolution but daytime only.
- Data files are ~20-50 MB each. Cache aggressively.
- Need to map lat/lon to GOES fixed grid coordinates (goes2go handles this).

---

## Phase 6: Scoring & Evaluation Framework

**Goal:** Continuously measure whether the upgraded model actually produces profitable edges.

### Files to create

| File | Purpose |
|---|---|
| `evaluate.py` | Brier score, reliability diagram, CRPS, profit tracking. |
| `data/edge_log.csv` | Append-only log: `date,city,ticker,our_prob,market_price,edge,direction,actual_outcome,pnl` |

### Key Metrics

```python
# evaluate.py

def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of probability forecasts."""

def reliability_diagram(predictions, outcomes, n_bins=10) -> dict:
    """Returns {bin_centers, observed_freq, predicted_freq, counts}."""

def crps_score(ensemble_members: np.ndarray, actual: float) -> float:
    """Continuous Ranked Probability Score — proper scoring rule for ensembles."""

def track_edge_resolution(edge_log: pd.DataFrame) -> dict:
    """Compute: win_rate, avg_edge_when_win, avg_edge_when_lose, total_pnl, roi."""
```

---

## Dependency Summary

### New pip packages

```
herbie-data>=2024.1.0    # Phase 4: HRRR
goes2go>=2022.1.0        # Phase 5: GOES
xskillscore>=0.0.24      # Phase 6: CRPS scoring (optional, can do manually)
```

### Already available

```
scikit-learn   # Isotonic regression, linear regression (in requirements.txt)
numpy          # Array ops (in requirements.txt)
pandas         # DataFrames (in requirements.txt)
requests       # HTTP (in requirements.txt)
```

---

## File Tree After All Phases

```
weather_signals/
├── main.py                    # Orchestrator (updated: ensemble + HRRR + GOES flow)
├── config.json                # Config (updated: new flags)
├── stations.json              # NEW: city → station mapping
├── fetch_forecasts.py         # Updated: + ensemble multi-location
├── fetch_kalshi.py            # Unchanged
├── fetch_polymarket.py        # Unchanged
├── fetch_hrrr.py              # NEW: HRRR via Herbie
├── fetch_goes.py              # NEW: GOES cloud data via goes2go
├── station_truth.py           # NEW: NWS CLI + NCEI CDO fetchers
├── calibration.py             # NEW: EMOS + Isotonic per-city models
├── train_calibration.py       # NEW: CLI to train/retrain all models
├── matcher.py                 # Updated: ensemble σ, EMOS correction, isotonic calibration, HRRR blend, GOES adjust
├── analyze.py                 # Existing (integrate ensemble spread)
├── evaluate.py                # NEW: Brier, reliability, CRPS, PnL tracking
└── data/
    ├── station_actuals/       # NEW: daily CSV per city
    ├── forecast_archive/      # NEW: archived forecasts for backtesting
    ├── calibration_models/    # NEW: pickled models per city
    └── edge_log.csv           # NEW: append-only edge resolution log
```

---

## Implementation Order & Estimates

| Phase | What | Depends On | New Files | Changed Files |
|-------|------|------------|-----------|---------------|
| 1 | Ensemble σ | Nothing | — | fetch_forecasts.py, matcher.py, main.py, config.json |
| 2 | Station truth | Nothing | station_truth.py, stations.json | — |
| 3 | Calibration | Phase 1 + 2 | calibration.py, train_calibration.py | matcher.py |
| 4 | HRRR | Phase 1 | fetch_hrrr.py | matcher.py, main.py |
| 5 | GOES | Phase 4 | fetch_goes.py | matcher.py, main.py |
| 6 | Evaluation | Phase 2 | evaluate.py | main.py |

**Phases 1 and 2 can be built in parallel** — they have no dependencies on each other. Phase 3 requires both. Phases 4 and 5 are independent of 3 but benefit from having calibration in place to measure improvement.

---

## Config After Upgrade

```json
{
  "locations": [...],
  "forecast_hours": 72,
  "min_edge_threshold": 0.05,
  "uncertainty_std_f": 2.0,
  "scan_interval_minutes": 30,
  "enable_kalshi": true,
  "enable_polymarket": false,
  "enable_ensemble": true,
  "enable_hrrr": true,
  "enable_goes": true,
  "enable_calibration": true,
  "calibration_retrain_day": "sunday",
  "calibration_window_days": 90,
  "hrrr_blend_horizon_hours": 18,
  "goes_cloud_threshold": 0.5,
  "ncei_api_token": "",
  "edge_log_path": "data/edge_log.csv"
}
```
