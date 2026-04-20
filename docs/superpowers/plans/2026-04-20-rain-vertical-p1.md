# Rain Vertical P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel rain-market trading pipeline to the existing temperature scanner — Kalshi KXRAIN binary-rain markets, 10-city watchlist, separate calibration stack, own 20% bankroll slice, paper-only with a 30-day evaluation gate.

**Architecture:** A second vertical that reuses all existing infrastructure (Kalshi client, station truth, paper-trading ledger, deepseek worker) via category-aware routing. New modules for precipitation fetching, rain calibration, and rain matching sit alongside the temperature equivalents and are toggled by `enable_rain_vertical`. Historical backfill from NCEI PRCP + Open-Meteo Previous Runs gives day-1 chronological-holdout calibration parity with the temperature pipeline.

**Tech Stack:** Python 3.12, pandas, scikit-learn (LogisticRegression, IsotonicRegression), Open-Meteo API (deterministic + ensemble + previous runs), Herbie (HRRR APCP), NCEI CDO + NWS CLI (truth), pytest.

**Spec:** [docs/superpowers/specs/2026-04-20-rain-vertical-p1-design.md](../specs/2026-04-20-rain-vertical-p1-design.md)

---

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `src/fetch_precipitation.py` | Open-Meteo precip (deterministic + ensemble) and HRRR APCP point extraction |
| `src/rain_calibration.py` | `LogisticRainCalibrator` + `IsotonicRainCalibrator` + rain-model manager helpers |
| `src/rain_matcher.py` | Parse KXRAIN outcomes, compute probability & edge |
| `train_rain_calibration.py` | CLI trainer — logistic + isotonic per city |
| `evaluate_rain_calibration.py` | CLI evaluator — chronological holdout, Brier/log-loss vs climatology |
| `strategy/rain_policy_v1.json` | Rain policy |
| `tests/test_fetch_precipitation.py` | Precip fetcher tests |
| `tests/test_rain_calibration.py` | Calibration class tests |
| `tests/test_rain_matcher.py` | KXRAIN parse + probability tests |
| `tests/test_rain_policy.py` | Category-aware policy filter tests |

### Modified Files

| File | Change |
|---|---|
| `src/station_truth.py` | Add `PRCP` to NCEI CDO datatypeid list; add `archive_precipitation_snapshot` + `build_rain_training_set` |
| `backfill_training_data.py` | Extend to backfill precipitation forecasts + actuals |
| `src/paper_trading.py` | Add `market_category` column; `category_breakdown` + daily-PnL correlation in `summary.json` |
| `src/strategy_policy.py` | Category-aware opportunity filter; accept `market_category` in policy schema |
| `main.py` | Wire rain matcher pass behind `enable_rain_vertical` flag |
| `config.example.json` | Add rain-vertical config keys (default off) |
| `data/CLAUDE.md`, `src/CLAUDE.md`, `strategy/CLAUDE.md`, `.claude/rules/` | Docs |
| `tests/test_paper_trading.py`, `tests/test_strategy_policy.py` | Add rain-category coverage |

### Generated Data (not committed to source control layout, but expected outputs)

- `data/rain_market_inventory.csv` — one-off discovery dump
- `data/precip_archive/{city_slug}.csv` — per-city precip forecast snapshots
- `data/calibration_models/{city_slug}_rain_binary_logistic.pkl`
- `data/calibration_models/{city_slug}_rain_binary_isotonic.pkl`

---

## Task 1: Extend NCEI CDO to fetch PRCP

**Rationale:** The `station_actuals/*.csv` schema already has a `precip_in` column (empty for ~355/363 rows) because the NCEI fetcher requests only `TMAX`/`TMIN`. Adding `PRCP` to the datatypeid list fills the column with no schema migration.

**Files:**
- Modify: `src/station_truth.py` (NCEI fetch function — search for `datatypeid` and `TMAX`)
- Test: `tests/test_training_regressions.py` (extend existing coverage)

- [ ] **Step 1: Locate the NCEI CDO fetch function**

Run: `grep -n "datatypeid\|TMAX" src/station_truth.py`

Expected: line numbers identifying where NCEI payload params are built.

- [ ] **Step 2: Write a failing test for PRCP ingestion**

Add to `tests/test_training_regressions.py`:

```python
def test_ncei_cdo_request_includes_prcp(monkeypatch):
    """NCEI CDO fetch must request PRCP alongside TMAX/TMIN."""
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params

        class _R:
            status_code = 200
            def json(self):
                return {"results": []}
            def raise_for_status(self):
                pass
        return _R()

    import requests
    from src import station_truth
    monkeypatch.setattr(requests, "get", fake_get)

    station_truth.fetch_historical_daily(
        ghcnd_id="USW00094728",
        start="2025-04-01",
        end="2025-04-05",
        token="fake-token",
    )

    datatypes = captured["params"].get("datatypeid")
    if isinstance(datatypes, list):
        assert "PRCP" in datatypes
    else:
        assert "PRCP" in str(datatypes)
    assert "TMAX" in str(datatypes) and "TMIN" in str(datatypes)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_training_regressions.py::test_ncei_cdo_request_includes_prcp -v`

Expected: FAIL with `assert "PRCP" in ...`.

- [ ] **Step 4: Add `PRCP` to the datatypeid list and parse precip rows**

In `src/station_truth.py`, locate the `fetch_historical_daily` function (or equivalent NCEI fetcher). Extend the datatypeid list:

```python
_NCEI_DATATYPES = ["TMAX", "TMIN", "PRCP"]
```

Update the results-parsing loop so `PRCP` rows populate `precip_in`. NCEI reports `PRCP` in tenths of mm; convert via `precip_in = prcp_tenths_mm / 254.0`:

```python
for row in results:
    dt = row.get("date", "")[:10]
    dtype = row.get("datatype")
    value = row.get("value")
    if dtype == "TMAX":
        daily[dt]["tmax_f"] = _tenths_c_to_f(value)
    elif dtype == "TMIN":
        daily[dt]["tmin_f"] = _tenths_c_to_f(value)
    elif dtype == "PRCP":
        daily[dt]["precip_in"] = float(value) / 254.0 if value is not None else None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_training_regressions.py::test_ncei_cdo_request_includes_prcp -v`

Expected: PASS.

- [ ] **Step 6: Run full test suite to confirm no regression**

Run: `.\.venv\Scripts\python.exe -m pytest tests -x`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/station_truth.py tests/test_training_regressions.py
git commit -m "feat(station_truth): fetch PRCP alongside TMAX/TMIN from NCEI CDO"
```

---

## Task 2: One-shot backfill of historical PRCP actuals

**Rationale:** Fills the empty `precip_in` column in existing `station_actuals/*.csv` files for the 10 watchlist cities. This is a one-off run, not new code — but it's gated on Task 1 and must happen before calibration training.

**Files:**
- Operational: `backfill_training_data.py` (existing CLI)
- Output: `data/station_actuals/*.csv` (updated in place)

- [ ] **Step 1: Verify Task 1 shipped and NCEI token is configured**

Run: `.\.venv\Scripts\python.exe -c "from src.config import load_app_config, resolve_config_path; c = load_app_config(str(resolve_config_path())); print('token_set' if c.get('ncei_api_token') else 'MISSING')"`

Expected: `token_set`. If `MISSING`, stop and ask the user to set `ncei_api_token` in `config.json`.

- [ ] **Step 2: Run the backfill for all 10 watchlist cities, 365 days**

Run:

```powershell
.\.venv\Scripts\python.exe backfill_training_data.py --days 365
```

Expected: log lines confirming station-actuals backfill for each city. Exit code 0.

- [ ] **Step 3: Spot-check precip fill rate**

Run: `.\.venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/station_actuals/new_york.csv'); print(f'rows={len(df)} precip_filled={df[\"precip_in\"].notna().sum()}')"`

Expected: `precip_filled` > 300 (≥ 80% of a year's days).

- [ ] **Step 4: Commit the refreshed actuals**

```bash
git add data/station_actuals/
git commit -m "data(station_actuals): backfill PRCP for 10 watchlist cities (365d)"
```

---

## Task 3: Extend backfill_training_data.py for precip forecasts

**Rationale:** Populate `data/precip_archive/*.csv` retroactively using Open-Meteo Previous Runs API, which serves historical `precipitation_sum` and `precipitation_probability_max` at the daily level.

**Files:**
- Modify: `backfill_training_data.py`
- Modify: `src/station_truth.py` (add `archive_previous_run_precipitation` helper)
- Modify: `src/fetch_forecasts.py` (extend `fetch_previous_run_forecast` to also return precip)
- Test: `tests/test_training_regressions.py`

- [ ] **Step 1: Write failing test for precip fields in previous-run response parsing**

Add to `tests/test_training_regressions.py`:

```python
def test_fetch_previous_run_forecast_returns_precipitation(monkeypatch):
    """Previous-run forecast must expose precipitation_sum and probability."""
    import requests
    from src import fetch_forecasts

    def fake_get(url, params=None, timeout=None):
        class _R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {
                    "daily": {
                        "time": ["2025-04-01"],
                        "temperature_2m_max": [60.0],
                        "temperature_2m_min": [45.0],
                        "precipitation_sum": [0.12],
                        "precipitation_probability_max": [78],
                    }
                }
        return _R()

    monkeypatch.setattr(requests, "get", fake_get)

    result = fetch_forecasts.fetch_previous_run_forecast(
        lat=40.7, lon=-74.0,
        target_date="2025-04-01",
        lead_days=1,
    )
    assert result is not None
    assert result["precipitation_sum_in"] == pytest.approx(0.12 / 25.4, rel=1e-3)
    assert result["precipitation_probability_max"] == 78
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_training_regressions.py::test_fetch_previous_run_forecast_returns_precipitation -v`

Expected: FAIL (fields not in return dict).

- [ ] **Step 3: Extend `fetch_previous_run_forecast` in `src/fetch_forecasts.py`**

Locate the function. Add to the query string: `precipitation_sum,precipitation_probability_max`. Parse them into the return dict:

```python
daily = payload.get("daily", {})
precip_mm = _first_or_none(daily.get("precipitation_sum"))
precip_prob = _first_or_none(daily.get("precipitation_probability_max"))
return {
    "date": target_date,
    "forecast_high_f": _c_to_f(_first_or_none(daily.get("temperature_2m_max"))),
    "forecast_low_f": _c_to_f(_first_or_none(daily.get("temperature_2m_min"))),
    "precipitation_sum_in": precip_mm / 25.4 if precip_mm is not None else None,
    "precipitation_probability_max": precip_prob,
    "lead_days": lead_days,
    "as_of_utc": run_datetime_iso,
    "forecast_source": "open_meteo_previous_runs",
}
```

(If helper names differ, adapt — the key additions are the two new fields and unit conversion.)

- [ ] **Step 4: Run test to confirm it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_training_regressions.py::test_fetch_previous_run_forecast_returns_precipitation -v`

Expected: PASS.

- [ ] **Step 5: Write a test for `archive_previous_run_precipitation` helper**

Add to `tests/test_training_regressions.py`:

```python
def test_archive_previous_run_precipitation_writes_expected_row(tmp_path):
    from src import station_truth
    station_truth.PROJECT_ROOT = tmp_path  # harmless; helper accepts base_dir
    precip_dir = tmp_path / "precip_archive"

    snapshot = {
        "date": "2025-04-01",
        "precipitation_sum_in": 0.12,
        "precipitation_probability_max": 78,
        "lead_days": 1,
        "as_of_utc": "2025-03-31T12:00:00+00:00",
        "forecast_source": "open_meteo_previous_runs",
    }

    station_truth.archive_previous_run_precipitation(
        city="New York",
        snapshot=snapshot,
        base_dir=precip_dir,
    )
    import pandas as pd
    df = pd.read_csv(precip_dir / "new_york.csv")
    assert df.iloc[0]["date"] == "2025-04-01"
    assert df.iloc[0]["forecast_amount_in"] == pytest.approx(0.12, rel=1e-3)
    assert df.iloc[0]["forecast_prob_any_rain"] == pytest.approx(0.78, rel=1e-3)
```

- [ ] **Step 6: Run test to confirm it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_training_regressions.py::test_archive_previous_run_precipitation_writes_expected_row -v`

Expected: FAIL — helper not defined.

- [ ] **Step 7: Implement `archive_previous_run_precipitation` in `src/station_truth.py`**

Add near `archive_forecast_snapshot`:

```python
PRECIP_ARCHIVE_DIR = DATA_DIR / "precip_archive"

_PRECIP_ARCHIVE_COLUMNS = [
    "as_of_utc", "date", "forecast_prob_any_rain", "forecast_amount_in",
    "ensemble_wet_fraction", "ensemble_amount_std_in",
    "forecast_model", "forecast_lead_days", "forecast_source",
]


def archive_previous_run_precipitation(
    city: str,
    snapshot: dict,
    base_dir: Path | str | None = None,
) -> None:
    directory = Path(base_dir) if base_dir is not None else PRECIP_ARCHIVE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slugify_city(city)}.csv"

    prob = snapshot.get("precipitation_probability_max")
    prob_fraction = float(prob) / 100.0 if prob is not None else None
    row = {
        "as_of_utc": snapshot.get("as_of_utc"),
        "date": snapshot.get("date"),
        "forecast_prob_any_rain": prob_fraction,
        "forecast_amount_in": snapshot.get("precipitation_sum_in"),
        "ensemble_wet_fraction": None,
        "ensemble_amount_std_in": None,
        "forecast_model": snapshot.get("forecast_model", "best_match"),
        "forecast_lead_days": snapshot.get("lead_days"),
        "forecast_source": snapshot.get("forecast_source", "open_meteo_previous_runs"),
    }
    frame = pd.DataFrame([row], columns=_PRECIP_ARCHIVE_COLUMNS)
    mode, header = ("a", False) if path.exists() else ("w", True)
    frame.to_csv(path, mode=mode, header=header, index=False)
```

- [ ] **Step 8: Run test to confirm it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_training_regressions.py::test_archive_previous_run_precipitation_writes_expected_row -v`

Expected: PASS.

- [ ] **Step 9: Wire the backfill CLI**

In `backfill_training_data.py`, extend the city-loop to also archive precipitation. Add after the temperature archive call:

```python
from src.station_truth import archive_previous_run_precipitation

if snapshot.get("precipitation_sum_in") is not None or snapshot.get("precipitation_probability_max") is not None:
    archive_previous_run_precipitation(city=city, snapshot=snapshot)
```

- [ ] **Step 10: Commit**

```bash
git add src/fetch_forecasts.py src/station_truth.py backfill_training_data.py tests/test_training_regressions.py
git commit -m "feat(backfill): extend previous-run forecast and backfill CLI with precipitation"
```

- [ ] **Step 11: Run the precipitation backfill for 365 days**

Run: `.\.venv\Scripts\python.exe backfill_training_data.py --days 365`

Expected: logs show per-city precipitation archive writes. Verify:

```bash
ls data/precip_archive/
```

Expected: ~10 CSVs, each with ~300+ rows. If fewer than 10 cities appear, investigate before continuing.

- [ ] **Step 12: Commit the archived precip**

```bash
git add data/precip_archive/
git commit -m "data(precip_archive): backfill 365d of Open-Meteo previous-run precip"
```

---

## Task 4: Create `src/fetch_precipitation.py` (live forecast fetcher)

**Rationale:** Parallel to `fetch_forecasts.py` — fetches Open-Meteo deterministic and ensemble precipitation for live scans, and HRRR APCP at station points for same-day blends.

**Files:**
- Create: `src/fetch_precipitation.py`
- Create: `tests/test_fetch_precipitation.py`

- [ ] **Step 1: Write failing test for deterministic precip fetcher**

Create `tests/test_fetch_precipitation.py`:

```python
import pytest
from unittest.mock import patch


def test_fetch_precipitation_returns_daily_probs_and_amounts():
    from src.fetch_precipitation import fetch_precipitation_multi

    fake_payload = {
        "daily": {
            "time": ["2026-04-21", "2026-04-22"],
            "precipitation_sum": [0.0, 5.1],
            "precipitation_probability_max": [5, 82],
        }
    }

    with patch("src.fetch_precipitation.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = fake_payload

        result = fetch_precipitation_multi(
            [{"name": "New York", "lat": 40.7, "lon": -74.0}],
            forecast_hours=72,
        )

    ny = result["New York"]
    assert ny["daily"][0]["date"] == "2026-04-21"
    assert ny["daily"][0]["forecast_prob_any_rain"] == pytest.approx(0.05)
    assert ny["daily"][0]["forecast_amount_in"] == pytest.approx(0.0, abs=1e-6)
    assert ny["daily"][1]["forecast_prob_any_rain"] == pytest.approx(0.82)
    assert ny["daily"][1]["forecast_amount_in"] == pytest.approx(5.1 / 25.4, rel=1e-3)
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_fetch_precipitation.py -v`

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `src/fetch_precipitation.py`**

```python
"""Open-Meteo precipitation + HRRR APCP fetcher, parallel to fetch_forecasts."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import requests

logger = logging.getLogger("weather.fetch_precipitation")

_DETERMINISTIC_URL = "https://api.open-meteo.com/v1/forecast"
_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"


def _mm_to_in(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / 25.4
    except (TypeError, ValueError):
        return None


def _percent_to_fraction(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return None


def fetch_precipitation_multi(
    locations: Iterable[dict],
    forecast_hours: int = 72,
    models: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Fetch deterministic precipitation for each location.

    Returns: {city_name: {"daily": [{"date", "forecast_prob_any_rain",
    "forecast_amount_in"}, ...]}}
    """
    results: dict[str, dict] = {}
    for loc in locations:
        name = loc.get("name")
        if not name:
            continue
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "daily": "precipitation_sum,precipitation_probability_max",
            "forecast_days": max(1, int(forecast_hours / 24)),
            "timezone": "UTC",
        }
        if models:
            params["models"] = ",".join(models)
        try:
            r = requests.get(_DETERMINISTIC_URL, params=params, timeout=20)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            logger.warning("Precip fetch failed for %s: %s", name, exc)
            continue
        daily = payload.get("daily", {})
        dates = daily.get("time", []) or []
        amounts = daily.get("precipitation_sum", []) or []
        probs = daily.get("precipitation_probability_max", []) or []
        daily_rows = []
        for i, d in enumerate(dates):
            daily_rows.append({
                "date": d,
                "forecast_prob_any_rain": _percent_to_fraction(probs[i] if i < len(probs) else None),
                "forecast_amount_in": _mm_to_in(amounts[i] if i < len(amounts) else None),
            })
        results[name] = {"daily": daily_rows}
    return results


def fetch_precipitation_ensemble_multi(
    locations: Iterable[dict],
    forecast_hours: int = 72,
) -> dict[str, dict]:
    """Fetch ensemble-member precipitation per location.

    Returns wet-day fraction and amount std across members for each date.
    """
    results: dict[str, dict] = {}
    for loc in locations:
        name = loc.get("name")
        if not name:
            continue
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "hourly": "precipitation",
            "forecast_days": max(1, int(forecast_hours / 24)),
            "timezone": "UTC",
            "models": "icon_seamless",
        }
        try:
            r = requests.get(_ENSEMBLE_URL, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            logger.warning("Ensemble precip fetch failed for %s: %s", name, exc)
            continue
        results[name] = _summarize_ensemble_precip(payload)
    return results


def _summarize_ensemble_precip(payload: dict) -> dict:
    """Compute per-date wet-fraction and amount std across ensemble members."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    # Each member comes back as a key like "precipitation_member01"
    member_keys = [k for k in hourly.keys() if k.startswith("precipitation")]
    if not times or not member_keys:
        return {"daily": []}

    # Bucket hourly rows by YYYY-MM-DD
    from collections import defaultdict
    date_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(times):
        date_to_indices[str(t)[:10]].append(i)

    daily_rows = []
    for d, indices in date_to_indices.items():
        member_totals = []
        for key in member_keys:
            series = hourly.get(key) or []
            total_mm = sum((series[i] or 0.0) for i in indices if i < len(series))
            member_totals.append(total_mm)
        if not member_totals:
            continue
        wet_count = sum(1 for m in member_totals if m >= (0.01 * 25.4))  # 0.01 in → mm
        wet_fraction = wet_count / len(member_totals)
        mean_mm = sum(member_totals) / len(member_totals)
        var_mm = sum((m - mean_mm) ** 2 for m in member_totals) / max(1, len(member_totals))
        std_mm = var_mm ** 0.5
        daily_rows.append({
            "date": d,
            "ensemble_wet_fraction": wet_fraction,
            "ensemble_amount_mean_in": _mm_to_in(mean_mm),
            "ensemble_amount_std_in": _mm_to_in(std_mm),
            "member_count": len(member_totals),
        })
    return {"daily": daily_rows}
```

- [ ] **Step 4: Run test to confirm it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_fetch_precipitation.py -v`

Expected: PASS.

- [ ] **Step 5: Write a test for the ensemble summarizer**

Add to `tests/test_fetch_precipitation.py`:

```python
def test_ensemble_precip_summarizer_computes_wet_fraction():
    from src.fetch_precipitation import _summarize_ensemble_precip

    payload = {
        "hourly": {
            "time": ["2026-04-21T00:00", "2026-04-21T12:00", "2026-04-22T00:00"],
            "precipitation_member01": [0.5, 0.0, 0.0],
            "precipitation_member02": [0.0, 0.0, 2.0],
            "precipitation_member03": [0.0, 0.0, 0.0],
            "precipitation_member04": [1.0, 0.0, 1.5],
        }
    }

    result = _summarize_ensemble_precip(payload)
    apr21 = next(d for d in result["daily"] if d["date"] == "2026-04-21")
    apr22 = next(d for d in result["daily"] if d["date"] == "2026-04-22")

    assert apr21["ensemble_wet_fraction"] == pytest.approx(0.5)   # members 01 + 04 wet
    assert apr22["ensemble_wet_fraction"] == pytest.approx(0.5)   # members 02 + 04 wet
    assert apr21["member_count"] == 4
```

- [ ] **Step 6: Run test to confirm it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_fetch_precipitation.py -v`

Expected: both tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/fetch_precipitation.py tests/test_fetch_precipitation.py
git commit -m "feat(fetch_precipitation): deterministic + ensemble precip fetchers"
```

---

## Task 5: Add HRRR APCP extraction

**Rationale:** Same-day blend requires HRRR accumulated-precipitation at station points. Mirrors the existing 2-m temp extraction in `fetch_hrrr.py`.

**Files:**
- Modify: `src/fetch_precipitation.py` (new function `fetch_hrrr_precip_multi`)
- Modify: `tests/test_fetch_precipitation.py`

- [ ] **Step 1: Write failing test for HRRR precip extraction**

Add to `tests/test_fetch_precipitation.py`:

```python
def test_hrrr_precip_returns_daily_accumulation(monkeypatch):
    """HRRR APCP extraction returns per-date cumulative inches per city."""
    from src import fetch_precipitation

    # Simulate extracted member timeseries
    fake_series = [
        {"valid_time": "2026-04-21T00:00:00", "apcp_kg_m2": 0.0},
        {"valid_time": "2026-04-21T06:00:00", "apcp_kg_m2": 2.54},   # 0.1 in
        {"valid_time": "2026-04-21T12:00:00", "apcp_kg_m2": 2.54},   # unchanged (no new rain)
        {"valid_time": "2026-04-21T18:00:00", "apcp_kg_m2": 7.62},   # +0.2 in
    ]

    def fake_extract(loc, fxx, **kwargs):
        return fake_series

    monkeypatch.setattr(fetch_precipitation, "_extract_hrrr_apcp_series", fake_extract)

    result = fetch_precipitation.fetch_hrrr_precip_multi(
        [{"name": "New York", "lat": 40.7, "lon": -74.0}],
        fxx=18,
    )

    ny = result["New York"]
    assert ny["2026-04-21"]["total_in"] == pytest.approx(0.3, rel=1e-3)
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_fetch_precipitation.py::test_hrrr_precip_returns_daily_accumulation -v`

Expected: FAIL — function not defined.

- [ ] **Step 3: Implement `fetch_hrrr_precip_multi` in `src/fetch_precipitation.py`**

Add to `src/fetch_precipitation.py`:

```python
def _extract_hrrr_apcp_series(location: dict, fxx: int, **kwargs) -> list[dict]:
    """Return [{valid_time, apcp_kg_m2}, ...] from the latest HRRR run.

    Uses Herbie the same way fetch_hrrr does, but requests the APCP variable.
    Mirrors the fetch_hrrr import-failure handling (raises RuntimeError if
    Herbie is unavailable so the caller can degrade gracefully).
    """
    try:
        from herbie import Herbie
    except ImportError as exc:
        raise RuntimeError("Herbie not installed — HRRR precip unavailable") from exc

    series: list[dict] = []
    for hour in range(0, fxx + 1):
        try:
            H = Herbie(model="hrrr", product="sfc", fxx=hour)
            ds = H.xarray(":APCP:surface")
            val_kg_m2 = float(ds.sel(
                latitude=location["lat"], longitude=location["lon"], method="nearest"
            ).values)
        except Exception as exc:
            logger.debug("HRRR APCP extract failed at f%02d: %s", hour, exc)
            continue
        series.append({
            "valid_time": str(ds.valid_time.values),
            "apcp_kg_m2": val_kg_m2,
        })
    return series


def fetch_hrrr_precip_multi(
    locations: Iterable[dict],
    fxx: int = 18,
    **kwargs,
) -> dict[str, dict]:
    """Fetch HRRR accumulated precipitation for each location.

    Returns: {city_name: {date: {"total_in": float}, ...}}

    HRRR APCP is reset at the start of each run, so "total accumulation" for
    a given date is the max APCP value observed on that date minus the first.
    One kg/m^2 of liquid water equals ~0.0394 in (1 mm = 0.0394 in).
    """
    results: dict[str, dict] = {}
    for loc in locations:
        name = loc.get("name")
        if not name:
            continue
        try:
            series = _extract_hrrr_apcp_series(loc, fxx=fxx, **kwargs)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("HRRR precip failed for %s: %s", name, exc)
            continue

        from collections import defaultdict
        by_date: dict[str, list[float]] = defaultdict(list)
        for row in series:
            date_str = str(row["valid_time"])[:10]
            by_date[date_str].append(row["apcp_kg_m2"])

        daily_totals: dict[str, dict] = {}
        for date_str, kg_values in by_date.items():
            if not kg_values:
                continue
            total_mm = max(kg_values) - min(kg_values)  # 1 kg/m^2 ≈ 1 mm
            daily_totals[date_str] = {"total_in": _mm_to_in(total_mm)}
        results[name] = daily_totals
    return results
```

- [ ] **Step 4: Run test to confirm it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_fetch_precipitation.py::test_hrrr_precip_returns_daily_accumulation -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fetch_precipitation.py tests/test_fetch_precipitation.py
git commit -m "feat(fetch_precipitation): add HRRR APCP extraction for same-day blend"
```

---

## Task 6: Discovery pass — dump Kalshi rain market inventory

**Rationale:** Before writing the matcher, we need to see the actual KXRAIN ticker/outcome format Kalshi is using today. This is an operational run, not new code.

**Files:**
- Output: `data/rain_market_inventory.csv` (one-off)

- [ ] **Step 1: Write a short script to dump the inventory**

Create `scripts/dump_rain_market_inventory.py`:

```python
"""One-off: dump Kalshi KXRAIN market inventory to CSV."""
from __future__ import annotations

import csv
from pathlib import Path

from src.fetch_kalshi import fetch_weather_markets


def main() -> None:
    markets = fetch_weather_markets(pages=5)
    rain = [m for m in markets if str(m.get("ticker", "")).startswith("KXRAIN")]
    out = Path("data/rain_market_inventory.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ticker", "market_question", "city", "market_date",
              "outcome", "close_time", "yes_ask", "volume_24h"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for m in rain:
            w.writerow({k: m.get(k) for k in fields})
    print(f"wrote {len(rain)} rows to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the discovery script**

Run: `.\.venv\Scripts\python.exe scripts/dump_rain_market_inventory.py`

Expected: `wrote N rows to data/rain_market_inventory.csv` with N ≥ 1. If 0, the existing Kalshi fetcher isn't returning KXRAIN markets — investigate before proceeding.

- [ ] **Step 3: Inspect the outcome column shape**

Run: `.\.venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/rain_market_inventory.csv'); print(df['outcome'].value_counts().head(20))"`

Expected: a handful of distinct outcome strings (e.g., `>= 0.01 in`, `>= 0.25 in`, `Any Rain`). Record what you see — the matcher parser in Task 7 needs to handle these shapes.

- [ ] **Step 4: Commit**

```bash
git add scripts/dump_rain_market_inventory.py data/rain_market_inventory.csv
git commit -m "chore(rain): dump one-off KXRAIN market inventory for parser design"
```

---

## Task 7: Create `src/rain_matcher.py`

**Rationale:** Parse KXRAIN outcomes, compute YES probability for binary any-rain markets from the calibrated rain probability, return opportunities matching `matcher.py`'s shape for downstream compatibility.

**Files:**
- Create: `src/rain_matcher.py`
- Create: `tests/test_rain_matcher.py`

- [ ] **Step 1: Write failing test for outcome parsing**

Create `tests/test_rain_matcher.py`:

```python
import pytest


def test_parse_rain_outcome_binary():
    from src.rain_matcher import parse_rain_outcome

    # Expect the parser to accept common KXRAIN outcome strings and return
    # {"threshold_in": float, "market_type": "rain_binary"} for binary markets.
    examples = [
        ("Yes", {"threshold_in": 0.01, "market_type": "rain_binary"}),
        ("No", {"threshold_in": 0.01, "market_type": "rain_binary"}),
        (">= 0.01 in", {"threshold_in": 0.01, "market_type": "rain_binary"}),
        ("any rain", {"threshold_in": 0.01, "market_type": "rain_binary"}),
    ]
    for raw, expected in examples:
        parsed = parse_rain_outcome(raw)
        assert parsed == expected, f"{raw} -> {parsed}"


def test_parse_rain_outcome_non_binary_returns_none():
    from src.rain_matcher import parse_rain_outcome

    # Non-binary thresholds are out of scope for P1 and should return None.
    assert parse_rain_outcome(">= 0.5 in") is None
    assert parse_rain_outcome("0.25 to 0.5 in") is None
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rain_matcher.py -v`

Expected: FAIL — module missing.

- [ ] **Step 3: Implement the parser and probability math**

Create `src/rain_matcher.py`:

```python
"""Rain market matcher for Kalshi KXRAIN binary any-rain contracts."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("weather.rain_matcher")

_BINARY_THRESHOLD_IN = 0.01
_BINARY_PHRASES = {"yes", "no", "any rain", "rain"}
_BINARY_PATTERN = re.compile(r"^\s*>?=?\s*0\.01\s*in\s*$", re.IGNORECASE)
_NON_BINARY_PATTERN = re.compile(r"\d+\.\d+", re.IGNORECASE)


def parse_rain_outcome(outcome: str) -> Optional[dict]:
    """Parse a Kalshi KXRAIN outcome string.

    Returns {"threshold_in": 0.01, "market_type": "rain_binary"} for
    binary any-rain markets. Returns None for non-binary thresholds
    (out of scope for P1).
    """
    if outcome is None:
        return None
    normalized = str(outcome).strip().lower()
    if normalized in _BINARY_PHRASES:
        return {"threshold_in": _BINARY_THRESHOLD_IN, "market_type": "rain_binary"}
    if _BINARY_PATTERN.match(normalized):
        return {"threshold_in": _BINARY_THRESHOLD_IN, "market_type": "rain_binary"}
    # anything with a non-0.01 decimal threshold is non-binary
    matches = _NON_BINARY_PATTERN.findall(normalized)
    if matches and not all(abs(float(m) - 0.01) < 1e-9 for m in matches):
        return None
    return None


def compute_rain_yes_probability(
    calibrated_prob_any_rain: float,
    position_side: str = "yes",
) -> float:
    """Map a calibrated any-rain probability to YES-contract probability.

    For KXRAIN binary markets, the YES outcome resolves if precip_in >= 0.01,
    which is exactly the calibrated probability. For NO contracts, return
    1 - p.
    """
    p = float(calibrated_prob_any_rain)
    p = max(0.001, min(0.999, p))
    if position_side == "no":
        return 1.0 - p
    return p


def match_kalshi_rain(
    precip_forecasts: dict,
    markets: list[dict],
    calibration_manager=None,
    hrrr_data: Optional[dict] = None,
    hrrr_blend_horizon_hours: float = 12.0,
    min_edge: float = 0.15,
    now_utc: Optional[datetime] = None,
) -> list[dict]:
    """Compute opportunities for binary KXRAIN markets.

    Only processes markets whose outcome parses as rain_binary.
    Output shape mirrors matcher.match_kalshi_markets().
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    opportunities: list[dict] = []

    for market in markets:
        ticker = str(market.get("ticker") or "")
        if not ticker.startswith("KXRAIN"):
            continue
        parsed = parse_rain_outcome(market.get("outcome"))
        if parsed is None:
            continue
        city = market.get("city")
        if not city or city not in precip_forecasts:
            continue
        market_date = market.get("market_date")
        if not market_date:
            continue

        daily = precip_forecasts[city].get("daily") or []
        forecast_row = next((d for d in daily if d.get("date") == market_date), None)
        if forecast_row is None:
            continue

        raw_prob = forecast_row.get("forecast_prob_any_rain")
        if raw_prob is None:
            continue

        forecast_blend_source = "open-meteo"
        prob_for_probability_calibration = raw_prob

        # HRRR same-day blend (within 12 h by default)
        close_time_raw = market.get("close_time")
        hours_to_settlement = None
        if close_time_raw:
            try:
                close_dt = datetime.fromisoformat(str(close_time_raw).replace("Z", "+00:00"))
                hours_to_settlement = (close_dt - now_utc).total_seconds() / 3600.0
            except ValueError:
                hours_to_settlement = None
        if (
            hrrr_data
            and city in hrrr_data
            and hours_to_settlement is not None
            and 0 <= hours_to_settlement <= hrrr_blend_horizon_hours
        ):
            hrrr_day = hrrr_data[city].get(market_date)
            if hrrr_day and hrrr_day.get("total_in") is not None:
                hrrr_wet = 1.0 if hrrr_day["total_in"] >= 0.01 else 0.0
                # Ramp: weight HRRR more as settlement nears, cap at 0.7
                w_hrrr = min(0.7, max(0.0, 1.0 - hours_to_settlement / hrrr_blend_horizon_hours))
                prob_for_probability_calibration = (
                    (1.0 - w_hrrr) * raw_prob + w_hrrr * hrrr_wet
                )
                forecast_blend_source = "open-meteo+hrrr"

        # Calibration (logistic bias + isotonic); fall through to raw if unavailable
        calibrated_prob = prob_for_probability_calibration
        forecast_calibration_source = "raw"
        probability_calibration_source = "raw"
        if calibration_manager is not None:
            cal = calibration_manager.calibrate_rain_probability(
                city=city,
                raw_prob=prob_for_probability_calibration,
            )
            if cal is not None:
                calibrated_prob = cal["calibrated_prob"]
                forecast_calibration_source = cal.get("forecast_calibration_source", "logistic")
                probability_calibration_source = cal.get("probability_calibration_source", "isotonic")

        yes_outcome = str(market.get("outcome", "")).strip().lower() in {"yes", "any rain", "rain"}
        position_side = "yes"
        our_prob = compute_rain_yes_probability(calibrated_prob, position_side="yes")
        market_price = float(market.get("yes_ask") or market.get("market_price") or 0.0)
        edge = our_prob - market_price
        abs_edge = abs(edge)
        if abs_edge < min_edge:
            continue

        opportunities.append({
            "source": "kalshi",
            "ticker": ticker,
            "city": city,
            "market_type": "rain_binary",
            "market_category": "rain",
            "market_date": market_date,
            "outcome": market.get("outcome"),
            "position_side": position_side,
            "our_probability": our_prob,
            "raw_probability": raw_prob,
            "market_price": market_price,
            "edge": edge,
            "abs_edge": abs_edge,
            "forecast_value_f": None,
            "forecast_blend_source": forecast_blend_source,
            "forecast_calibration_source": forecast_calibration_source,
            "probability_calibration_source": probability_calibration_source,
            "hours_to_settlement": hours_to_settlement,
            "volume24hr": market.get("volume24hr") or market.get("volume_24h"),
            "yes_outcome": yes_outcome,
        })
    return opportunities
```

- [ ] **Step 4: Run parser tests to confirm they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rain_matcher.py -v`

Expected: PASS.

- [ ] **Step 5: Add probability-math tests**

Add to `tests/test_rain_matcher.py`:

```python
def test_compute_rain_yes_probability_clips():
    from src.rain_matcher import compute_rain_yes_probability
    assert compute_rain_yes_probability(0.0) == 0.001
    assert compute_rain_yes_probability(1.0) == 0.999
    assert compute_rain_yes_probability(0.5) == 0.5


def test_match_kalshi_rain_emits_edge_opportunity():
    from src.rain_matcher import match_kalshi_rain

    precip = {
        "New York": {
            "daily": [
                {"date": "2026-04-21", "forecast_prob_any_rain": 0.82, "forecast_amount_in": 0.4},
            ]
        }
    }
    markets = [{
        "ticker": "KXRAINNYC-26APR21",
        "outcome": "Yes",
        "city": "New York",
        "market_date": "2026-04-21",
        "yes_ask": 0.55,
        "volume_24h": 1500,
        "close_time": "2026-04-21T23:59:00Z",
    }]
    from datetime import datetime, timezone
    now = datetime(2026, 4, 19, 12, 0, tzinfo=timezone.utc)

    opps = match_kalshi_rain(precip, markets, now_utc=now, min_edge=0.15)
    assert len(opps) == 1
    opp = opps[0]
    assert opp["market_category"] == "rain"
    assert opp["our_probability"] == pytest.approx(0.82)
    assert opp["abs_edge"] == pytest.approx(0.27, rel=1e-2)
```

- [ ] **Step 6: Run all matcher tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rain_matcher.py -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rain_matcher.py tests/test_rain_matcher.py
git commit -m "feat(rain_matcher): KXRAIN parsing + probability + HRRR blend"
```

---

## Task 8: Implement `src/rain_calibration.py`

**Rationale:** Two-stage calibration matching the temperature pipeline pattern (logistic bias correction + isotonic probability recalibration), designed with a generic binary-outcome interface so P2 can reuse it for temperature tails.

**Files:**
- Create: `src/rain_calibration.py`
- Create: `tests/test_rain_calibration.py`

- [ ] **Step 1: Write failing tests for `LogisticRainCalibrator`**

Create `tests/test_rain_calibration.py`:

```python
import numpy as np
import pytest


def test_logistic_rain_calibrator_fit_and_predict():
    from src.rain_calibration import LogisticRainCalibrator

    rng = np.random.default_rng(7)
    n = 300
    raw_probs = rng.uniform(0.05, 0.95, n)
    # True wet-day rate is somewhat biased vs raw_probs:
    # wet_rate = sigmoid(2 * raw_prob - 1)  → known systematic shift
    true = 1 / (1 + np.exp(-(2.0 * raw_probs - 1.0)))
    outcomes = (rng.uniform(0, 1, n) < true).astype(int)

    cal = LogisticRainCalibrator(city="New York")
    cal.fit(raw_probs, outcomes)
    # Predictions should be roughly ordered consistently with raw_probs
    preds = np.array([cal.predict(p) for p in [0.1, 0.5, 0.9]])
    assert preds[0] < preds[1] < preds[2]
    # And clipped to [MIN_PROB, MAX_PROB]
    assert all(0.001 <= p <= 0.999 for p in preds)


def test_logistic_rain_calibrator_save_load(tmp_path):
    from src.rain_calibration import LogisticRainCalibrator

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.05, 0.95, 100)
    out = (rng.uniform(0, 1, 100) < raw).astype(int)

    cal = LogisticRainCalibrator(city="New York")
    cal.fit(raw, out)
    path = tmp_path / "ny_rain_logistic.pkl"
    cal.save(path)

    loaded = LogisticRainCalibrator.load(path)
    assert loaded.city == "New York"
    assert abs(loaded.predict(0.4) - cal.predict(0.4)) < 1e-9
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rain_calibration.py -v`

Expected: FAIL — module missing.

- [ ] **Step 3: Implement `LogisticRainCalibrator` + `IsotonicRainCalibrator`**

Create `src/rain_calibration.py`:

```python
"""Rain calibration: logistic bias correction + isotonic probability recalibration."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.station_truth import CALIBRATION_MODELS_DIR, _slugify_city

logger = logging.getLogger("weather.rain_calibration")

_MIN_PROB = 0.001
_MAX_PROB = 0.999


def _clip(p: float) -> float:
    return max(_MIN_PROB, min(_MAX_PROB, float(p)))


class LogisticRainCalibrator:
    """Per-city logistic bias correction for binary rain outcomes."""

    def __init__(self, city: str):
        self.city = city
        self._model: Optional[LogisticRegression] = None

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> None:
        x = np.asarray(raw_probs, dtype=float).reshape(-1, 1)
        y = np.asarray(outcomes, dtype=int).reshape(-1)
        if len(np.unique(y)) < 2:
            # Degenerate — no contrast; refuse to fit, caller falls back to raw.
            self._model = None
            return
        self._model = LogisticRegression().fit(x, y)

    def predict(self, raw_prob: float) -> float:
        if self._model is None:
            return _clip(raw_prob)
        prob = float(self._model.predict_proba(np.array([[float(raw_prob)]]))[0, 1])
        return _clip(prob)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"city": self.city, "model": self._model}, f)

    @classmethod
    def load(cls, path: Path | str) -> "LogisticRainCalibrator":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        obj = cls(city=data["city"])
        obj._model = data["model"]
        return obj


class IsotonicRainCalibrator:
    """Per-city isotonic probability recalibrator."""

    def __init__(self, city: str):
        self.city = city
        self._model: Optional[IsotonicRegression] = None

    def fit(self, predicted_probs: np.ndarray, outcomes: np.ndarray) -> None:
        x = np.asarray(predicted_probs, dtype=float).reshape(-1)
        y = np.asarray(outcomes, dtype=int).reshape(-1)
        if len(np.unique(y)) < 2:
            self._model = None
            return
        self._model = IsotonicRegression(out_of_bounds="clip").fit(x, y)

    def predict(self, predicted_prob: float) -> float:
        if self._model is None:
            return _clip(predicted_prob)
        return _clip(float(self._model.predict([float(predicted_prob)])[0]))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"city": self.city, "model": self._model}, f)

    @classmethod
    def load(cls, path: Path | str) -> "IsotonicRainCalibrator":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        obj = cls(city=data["city"])
        obj._model = data["model"]
        return obj


def _rain_model_path(city: str, kind: str, model_dir: Optional[Path | str] = None) -> Path:
    directory = Path(model_dir) if model_dir is not None else CALIBRATION_MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_slugify_city(city)}_rain_binary_{kind}.pkl"


class RainCalibrationManager:
    """Load and apply rain calibration models at scan time."""

    def __init__(self, model_dir: Optional[Path | str] = None):
        self._model_dir = Path(model_dir) if model_dir else CALIBRATION_MODELS_DIR
        self._logistic_cache: dict[str, LogisticRainCalibrator | None] = {}
        self._isotonic_cache: dict[str, IsotonicRainCalibrator | None] = {}

    def _logistic(self, city: str) -> Optional[LogisticRainCalibrator]:
        if city not in self._logistic_cache:
            path = _rain_model_path(city, "logistic", self._model_dir)
            self._logistic_cache[city] = LogisticRainCalibrator.load(path) if path.exists() else None
        return self._logistic_cache[city]

    def _isotonic(self, city: str) -> Optional[IsotonicRainCalibrator]:
        if city not in self._isotonic_cache:
            path = _rain_model_path(city, "isotonic", self._model_dir)
            self._isotonic_cache[city] = IsotonicRainCalibrator.load(path) if path.exists() else None
        return self._isotonic_cache[city]

    def calibrate_rain_probability(self, city: str, raw_prob: float) -> Optional[dict]:
        """Apply logistic → isotonic chain. Returns None if no model exists."""
        logistic = self._logistic(city)
        isotonic = self._isotonic(city)
        if logistic is None and isotonic is None:
            return None
        p = _clip(raw_prob)
        forecast_src = "raw"
        prob_src = "raw"
        if logistic is not None:
            p = logistic.predict(p)
            forecast_src = "logistic"
        if isotonic is not None:
            p = isotonic.predict(p)
            prob_src = "isotonic"
        return {
            "calibrated_prob": p,
            "forecast_calibration_source": forecast_src,
            "probability_calibration_source": prob_src,
        }
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rain_calibration.py -v`

Expected: PASS.

- [ ] **Step 5: Add isotonic + manager tests**

Add to `tests/test_rain_calibration.py`:

```python
def test_isotonic_rain_calibrator_preserves_monotonicity(tmp_path):
    from src.rain_calibration import IsotonicRainCalibrator
    rng = np.random.default_rng(3)
    preds = np.linspace(0.02, 0.98, 200)
    outcomes = (rng.uniform(0, 1, 200) < preds).astype(int)
    cal = IsotonicRainCalibrator(city="Chicago")
    cal.fit(preds, outcomes)
    out = [cal.predict(p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
    assert all(b >= a - 1e-9 for a, b in zip(out, out[1:]))


def test_rain_calibration_manager_round_trip(tmp_path):
    from src.rain_calibration import (
        LogisticRainCalibrator, IsotonicRainCalibrator, RainCalibrationManager,
    )
    rng = np.random.default_rng(11)
    probs = rng.uniform(0.05, 0.95, 300)
    outcomes = (rng.uniform(0, 1, 300) < probs).astype(int)

    LogisticRainCalibrator("New York").__class__  # import check
    logistic = LogisticRainCalibrator(city="New York"); logistic.fit(probs, outcomes)
    isotonic = IsotonicRainCalibrator(city="New York"); isotonic.fit(probs, outcomes)
    logistic.save(tmp_path / "new_york_rain_binary_logistic.pkl")
    isotonic.save(tmp_path / "new_york_rain_binary_isotonic.pkl")

    mgr = RainCalibrationManager(model_dir=tmp_path)
    result = mgr.calibrate_rain_probability(city="New York", raw_prob=0.42)
    assert result is not None
    assert 0.001 <= result["calibrated_prob"] <= 0.999
    assert result["forecast_calibration_source"] == "logistic"
    assert result["probability_calibration_source"] == "isotonic"
```

- [ ] **Step 6: Run all tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rain_calibration.py -v`

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/rain_calibration.py tests/test_rain_calibration.py
git commit -m "feat(rain_calibration): logistic bias + isotonic recalibration + manager"
```

---

## Task 9: Training + evaluation CLIs

**Rationale:** `train_rain_calibration.py` reads the precip archive + station actuals, trains per-city models, persists to `data/calibration_models/`. `evaluate_rain_calibration.py` runs chronological holdout and reports Brier / log-loss vs climatology.

**Files:**
- Create: `train_rain_calibration.py`
- Create: `evaluate_rain_calibration.py`
- Modify: `src/station_truth.py` (add `build_rain_training_set`)
- Test: `tests/test_rain_calibration.py`

- [ ] **Step 1: Write a failing test for `build_rain_training_set`**

Add to `tests/test_rain_calibration.py`:

```python
def test_build_rain_training_set_joins_forecasts_and_actuals(tmp_path):
    from src import station_truth
    import pandas as pd

    actuals_dir = tmp_path / "station_actuals"
    actuals_dir.mkdir()
    pd.DataFrame({
        "date": ["2026-04-01", "2026-04-02"],
        "tmax_f": [70, 72], "tmin_f": [55, 58],
        "precip_in": [0.0, 0.5],
        "precip_trace": [False, False],
        "cli_station": ["NYC", "NYC"], "source_url": ["", ""],
        "city": ["New York", "New York"], "source": ["cdo", "cdo"], "archive_version": ["", ""],
    }).to_csv(actuals_dir / "new_york.csv", index=False)

    precip_dir = tmp_path / "precip_archive"
    precip_dir.mkdir()
    pd.DataFrame({
        "as_of_utc": ["2026-03-31T12:00:00+00:00", "2026-04-01T12:00:00+00:00"],
        "date": ["2026-04-01", "2026-04-02"],
        "forecast_prob_any_rain": [0.15, 0.80],
        "forecast_amount_in": [0.0, 0.3],
        "ensemble_wet_fraction": [0.1, 0.75],
        "ensemble_amount_std_in": [0.01, 0.2],
        "forecast_model": ["best_match", "best_match"],
        "forecast_lead_days": [1, 1],
        "forecast_source": ["open_meteo_previous_runs"] * 2,
    }).to_csv(precip_dir / "new_york.csv", index=False)

    training = station_truth.build_rain_training_set(
        city="New York",
        actuals_dir=actuals_dir,
        precip_dir=precip_dir,
    )
    assert list(training["date"]) == ["2026-04-01", "2026-04-02"]
    assert list(training["actual_wet_0_1"]) == [0, 1]
    assert list(training["raw_prob"]) == [0.15, 0.80]
```

- [ ] **Step 2: Run it to confirm failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rain_calibration.py::test_build_rain_training_set_joins_forecasts_and_actuals -v`

Expected: FAIL — function missing.

- [ ] **Step 3: Implement `build_rain_training_set` in `src/station_truth.py`**

Add near `build_training_set`:

```python
def build_rain_training_set(
    city: str,
    actuals_dir: Path | str | None = None,
    precip_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Join archived precipitation forecasts with station precip actuals.

    Returns a DataFrame with columns:
    date, raw_prob, forecast_amount_in, ensemble_wet_fraction,
    ensemble_amount_std_in, actual_precip_in, actual_wet_0_1,
    forecast_lead_days, as_of_utc.
    """
    actuals_dir = Path(actuals_dir) if actuals_dir is not None else STATION_ACTUALS_DIR
    precip_dir = Path(precip_dir) if precip_dir is not None else PRECIP_ARCHIVE_DIR
    slug = _slugify_city(city)
    actuals_path = actuals_dir / f"{slug}.csv"
    precip_path = precip_dir / f"{slug}.csv"
    empty_cols = [
        "date", "raw_prob", "forecast_amount_in", "ensemble_wet_fraction",
        "ensemble_amount_std_in", "actual_precip_in", "actual_wet_0_1",
        "forecast_lead_days", "as_of_utc",
    ]
    if not actuals_path.exists() or not precip_path.exists():
        return pd.DataFrame(columns=empty_cols)

    actuals = pd.read_csv(actuals_path)
    precip = pd.read_csv(precip_path)
    if precip.empty or actuals.empty:
        return pd.DataFrame(columns=empty_cols)

    # Prefer the earliest lead-1 forecast per date (mirrors temp pipeline discipline)
    precip["date"] = pd.to_datetime(precip["date"]).dt.strftime("%Y-%m-%d")
    precip = precip.sort_values(["date", "forecast_lead_days", "as_of_utc"])
    precip = precip.drop_duplicates("date", keep="first")

    actuals["date"] = pd.to_datetime(actuals["date"]).dt.strftime("%Y-%m-%d")
    joined = precip.merge(actuals[["date", "precip_in"]], on="date", how="inner")
    joined["actual_precip_in"] = pd.to_numeric(joined["precip_in"], errors="coerce")
    joined["actual_wet_0_1"] = (joined["actual_precip_in"] >= 0.01).astype("Int64")

    return joined.rename(
        columns={"forecast_prob_any_rain": "raw_prob"}
    )[[
        "date", "raw_prob", "forecast_amount_in", "ensemble_wet_fraction",
        "ensemble_amount_std_in", "actual_precip_in", "actual_wet_0_1",
        "forecast_lead_days", "as_of_utc",
    ]].reset_index(drop=True)
```

- [ ] **Step 4: Run the test**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_rain_calibration.py::test_build_rain_training_set_joins_forecasts_and_actuals -v`

Expected: PASS.

- [ ] **Step 5: Implement `train_rain_calibration.py`**

Create `train_rain_calibration.py`:

```python
"""Train per-city rain calibration models (logistic + isotonic)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.logging_setup import configure_logging
from src.rain_calibration import (
    LogisticRainCalibrator, IsotonicRainCalibrator, _rain_model_path,
)
from src.station_truth import build_rain_training_set, load_station_map, ensure_data_directories

log = configure_logging("weather.train_rain", log_filename="train_rain_calibration.log", level=logging.INFO)


def train_city(city: str, window_days: int = 90) -> dict:
    training = build_rain_training_set(city=city)
    training = training.dropna(subset=["raw_prob", "actual_wet_0_1"])
    if len(training) < 30:
        log.warning("%s: only %d usable rows; skipping", city, len(training))
        return {"city": city, "trained": False, "rows": len(training)}

    # Use the most recent window_days rows (chronological order preserved by build_)
    recent = training.tail(window_days).reset_index(drop=True)
    raw = recent["raw_prob"].astype(float).values
    outcomes = recent["actual_wet_0_1"].astype(int).values

    logistic = LogisticRainCalibrator(city=city)
    logistic.fit(raw, outcomes)
    logistic.save(_rain_model_path(city, "logistic"))

    # Isotonic is fit on logistic-corrected predictions
    logistic_preds = [logistic.predict(p) for p in raw]
    isotonic = IsotonicRainCalibrator(city=city)
    isotonic.fit(logistic_preds, outcomes)
    isotonic.save(_rain_model_path(city, "isotonic"))

    log.info("%s: trained logistic+isotonic on %d rows", city, len(recent))
    return {"city": city, "trained": True, "rows": len(recent)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", help="Single city (default: all in stations.json)")
    parser.add_argument("--window-days", type=int, default=90)
    args = parser.parse_args()

    ensure_data_directories()
    cities = [args.city] if args.city else list(load_station_map().keys())
    for city in cities:
        train_city(city, window_days=args.window_days)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Implement `evaluate_rain_calibration.py`**

Create `evaluate_rain_calibration.py`:

```python
"""Chronological-holdout evaluation of rain calibration."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.rain_calibration import (
    LogisticRainCalibrator, IsotonicRainCalibrator, _rain_model_path,
)
from src.station_truth import build_rain_training_set, load_station_map


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def _log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-np.mean(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs)))


def evaluate_city(city: str, days: int, holdout_days: int) -> dict:
    training = build_rain_training_set(city=city).dropna(
        subset=["raw_prob", "actual_wet_0_1"]
    )
    training = training.tail(days).reset_index(drop=True)
    if len(training) < holdout_days + 20:
        return {"city": city, "usable_rows": len(training), "status": "insufficient_data"}

    train_slice = training.iloc[:-holdout_days]
    holdout = training.iloc[-holdout_days:]

    logistic = LogisticRainCalibrator(city=city)
    logistic.fit(train_slice["raw_prob"].values, train_slice["actual_wet_0_1"].astype(int).values)
    logistic_preds_train = np.array([logistic.predict(p) for p in train_slice["raw_prob"].values])
    isotonic = IsotonicRainCalibrator(city=city)
    isotonic.fit(logistic_preds_train, train_slice["actual_wet_0_1"].astype(int).values)

    y = holdout["actual_wet_0_1"].astype(int).values
    raw = holdout["raw_prob"].astype(float).values
    cal = np.array([isotonic.predict(logistic.predict(p)) for p in raw])
    climatology = float(train_slice["actual_wet_0_1"].astype(int).mean()) * np.ones_like(y, dtype=float)

    return {
        "city": city,
        "status": "ok",
        "holdout_n": int(len(y)),
        "train_n": int(len(train_slice)),
        "brier_raw": _brier(raw, y),
        "brier_calibrated": _brier(cal, y),
        "brier_climatology": _brier(climatology, y),
        "log_loss_raw": _log_loss(raw, y),
        "log_loss_calibrated": _log_loss(cal, y),
        "log_loss_climatology": _log_loss(climatology, y),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--holdout-days", type=int, default=30)
    args = parser.parse_args()

    cities = list(load_station_map().keys())
    per_city = [evaluate_city(c, days=args.days, holdout_days=args.holdout_days) for c in cities]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": {"days": args.days, "holdout_days": args.holdout_days},
        "cities": per_city,
    }
    out = Path("data/evaluation_reports") / f"rain_eval_{datetime.now(timezone.utc).date().isoformat()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the test suite and confirm no regressions**

Run: `.\.venv\Scripts\python.exe -m pytest tests -x`

Expected: all tests PASS.

- [ ] **Step 8: Train initial models against the backfilled archive**

Run: `.\.venv\Scripts\python.exe train_rain_calibration.py`

Expected: logs "trained logistic+isotonic on N rows" for each watchlist city. `data/calibration_models/*_rain_binary_*.pkl` files exist.

- [ ] **Step 9: Run the first evaluation scorecard**

Run: `.\.venv\Scripts\python.exe evaluate_rain_calibration.py --days 400 --holdout-days 30`

Expected: report written to `data/evaluation_reports/rain_eval_2026-04-20.json`. Inspect: `brier_calibrated` should be less than `brier_climatology` for most cities. If not, review training-set size and retry after more archive has accumulated.

- [ ] **Step 10: Commit**

```bash
git add src/station_truth.py train_rain_calibration.py evaluate_rain_calibration.py tests/test_rain_calibration.py data/calibration_models/ data/evaluation_reports/
git commit -m "feat(rain): trainer + chronological-holdout evaluator, first model batch"
```

---

## Task 10: Rain policy JSON + category-aware filter

**Rationale:** Dedicated rain policy with conservative starting thresholds. Extend `strategy_policy.py` to accept a `market_category` scope and route opportunities by category.

**Files:**
- Create: `strategy/rain_policy_v1.json`
- Modify: `src/strategy_policy.py`
- Modify: `tests/test_strategy_policy.py`
- Create (or modify existing): `tests/test_rain_policy.py`

- [ ] **Step 1: Write the policy JSON**

Create `strategy/rain_policy_v1.json`:

```json
{
  "policy_version": 1,
  "status": "active",
  "market_category": "rain",
  "generated_at_utc": "2026-04-20T12:00:00Z",
  "generated_by": "rain_vertical_p1_plan",
  "objective": "Paper-only evaluation of binary any-rain markets on 10-city watchlist. 30-day gate before any real-money consideration.",
  "selection": {
    "sources": ["kalshi"],
    "min_abs_edge": 0.15,
    "min_volume24hr": 500,
    "max_candidates_per_scan": 2,
    "max_hours_to_settlement": 24,
    "allowed_market_types": ["rain_binary"],
    "allowed_position_sides": ["yes"],
    "allowed_cities": [
      "New York", "Chicago", "Boston", "Philadelphia", "Miami",
      "Atlanta", "Houston", "Seattle", "New Orleans", "Washington DC"
    ],
    "blocked_cities": []
  },
  "execution": {
    "max_contracts_per_trade": 1,
    "max_new_orders_per_day": 2,
    "max_order_cost_dollars": 10.0,
    "time_in_force": "fill_or_kill"
  },
  "rationale": {
    "v1_notes": "Initial launch of rain vertical. Thresholds intentionally conservative, mirroring temperature v4's BUY-only posture while calibration evidence accumulates. min_volume24hr set lower than temperature (500 vs 2000) because rain markets are expected to be smaller."
  }
}
```

- [ ] **Step 2: Write failing test for category-aware filter**

Add to `tests/test_strategy_policy.py`:

```python
def test_filter_respects_market_category_scope(tmp_path):
    from src.strategy_policy import filter_opportunities_for_policy

    rain_policy = {
        "market_category": "rain",
        "selection": {
            "sources": ["kalshi"],
            "min_abs_edge": 0.10, "min_volume24hr": 0,
            "max_candidates_per_scan": 5,
            "max_hours_to_settlement": 48,
            "allowed_market_types": ["rain_binary"],
            "allowed_position_sides": ["yes"],
            "allowed_cities": ["New York"],
            "blocked_cities": [],
        },
    }
    opps = [
        {"source": "kalshi", "ticker": "KXRAINNYC-26APR21", "market_category": "rain",
         "city": "New York", "market_type": "rain_binary", "position_side": "yes",
         "abs_edge": 0.2, "volume24hr": 1000, "hours_to_settlement": 12},
        {"source": "kalshi", "ticker": "KXHIGHTNYC-26APR21", "market_category": "temperature",
         "city": "New York", "market_type": "high", "position_side": "yes",
         "abs_edge": 0.2, "volume24hr": 1000, "hours_to_settlement": 12},
    ]
    result = filter_opportunities_for_policy(opps, rain_policy)
    assert len(result) == 1
    assert result[0]["market_category"] == "rain"
```

- [ ] **Step 3: Run test to confirm failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_strategy_policy.py::test_filter_respects_market_category_scope -v`

Expected: FAIL — filter does not currently respect category.

- [ ] **Step 4: Extend `src/strategy_policy.py` with category filtering**

Locate `filter_opportunities_for_policy`. Before the existing filter loop, add:

```python
policy_category = (policy or {}).get("market_category")
if policy_category:
    opportunities = [
        o for o in opportunities
        if (o.get("market_category") or "temperature") == policy_category
    ]
```

Legacy opportunities without `market_category` default to `temperature` so existing policies continue to filter correctly.

- [ ] **Step 5: Run the failing test to confirm it now passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_strategy_policy.py::test_filter_respects_market_category_scope -v`

Expected: PASS.

- [ ] **Step 6: Add regression test for temperature policy unchanged behavior**

Add to `tests/test_strategy_policy.py`:

```python
def test_temperature_policy_without_category_still_accepts_legacy_opps():
    from src.strategy_policy import filter_opportunities_for_policy

    policy = {
        "selection": {
            "sources": ["kalshi"], "min_abs_edge": 0.0, "min_volume24hr": 0,
            "max_candidates_per_scan": 10, "max_hours_to_settlement": 48,
            "allowed_market_types": ["high", "low"],
            "allowed_position_sides": ["yes", "no"],
            "allowed_cities": [], "blocked_cities": [],
        },
    }
    opps = [
        {"source": "kalshi", "ticker": "KXHIGHT", "city": "New York",
         "market_type": "high", "position_side": "yes",
         "abs_edge": 0.1, "volume24hr": 100, "hours_to_settlement": 12},
    ]
    assert len(filter_opportunities_for_policy(opps, policy)) == 1
```

- [ ] **Step 7: Run full policy test file**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_strategy_policy.py -v`

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add strategy/rain_policy_v1.json src/strategy_policy.py tests/test_strategy_policy.py
git commit -m "feat(strategy): rain_policy_v1.json + category-aware opportunity filter"
```

---

## Task 11: Paper-trade ledger market_category column + category_breakdown summary

**Rationale:** The ledger needs to record which vertical each trade belongs to, and `summary.json` needs per-category PnL plus daily-PnL correlation for the promotion-gate monitoring.

**Files:**
- Modify: `src/paper_trading.py`
- Modify: `tests/test_paper_trading.py`

- [ ] **Step 1: Write failing test — new row gets market_category from opportunity**

Add to `tests/test_paper_trading.py`:

```python
def test_log_paper_trades_records_market_category(tmp_path):
    from src.paper_trading import log_paper_trades
    opps = [{
        "source": "kalshi", "ticker": "KXRAINNYC-26APR21",
        "market_category": "rain", "city": "New York", "market_type": "rain_binary",
        "market_date": "2026-04-21", "outcome": "Yes", "position_side": "yes",
        "our_probability": 0.8, "market_price": 0.55, "edge": 0.25,
        "abs_edge": 0.25, "forecast_blend_source": "open-meteo",
        "forecast_calibration_source": "logistic",
        "probability_calibration_source": "isotonic",
        "hours_to_settlement": 10, "raw_probability": 0.7,
        "volume24hr": 1500, "yes_outcome": True,
    }]
    ledger = tmp_path / "ledger.csv"
    log_paper_trades(opps, scan_timestamp="2026-04-20T12:00:00+00:00",
                     ledger_path=ledger, contracts=1)
    import pandas as pd
    df = pd.read_csv(ledger)
    assert "market_category" in df.columns
    assert df.iloc[0]["market_category"] == "rain"
```

- [ ] **Step 2: Write failing test — legacy ledgers migrate with `temperature`**

```python
def test_legacy_ledger_migrates_market_category_to_temperature(tmp_path):
    import pandas as pd
    from src.paper_trading import _ensure_ledger_schema

    legacy = tmp_path / "legacy.csv"
    pd.DataFrame({
        "trade_id": ["t1"], "scan_id": ["s1"], "source": ["kalshi"],
        "ticker": ["KXHIGHT"], "city": ["Boston"], "market_type": ["high"],
        "market_date": ["2026-04-20"], "status": ["open"],
    }).to_csv(legacy, index=False)

    df = _ensure_ledger_schema(pd.read_csv(legacy))
    assert "market_category" in df.columns
    assert df.iloc[0]["market_category"] == "temperature"
```

- [ ] **Step 3: Run the tests to confirm failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_paper_trading.py::test_log_paper_trades_records_market_category tests/test_paper_trading.py::test_legacy_ledger_migrates_market_category_to_temperature -v`

Expected: both FAIL.

- [ ] **Step 4: Add `market_category` to ledger schema and migration**

In `src/paper_trading.py`:

1. Add `"market_category"` to both `OBJECT_COLUMNS` and `LEDGER_COLUMNS` (after `"market_type"`).
2. In the row-construction code (find where opportunity fields are copied into the new row), add:

```python
"market_category": _coerce_text(opp.get("market_category"), default="temperature"),
```

3. Add or extend `_ensure_ledger_schema` so legacy ledgers get `market_category` backfilled:

```python
def _ensure_ledger_schema(df: pd.DataFrame) -> pd.DataFrame:
    if "market_category" not in df.columns:
        df["market_category"] = "temperature"
    else:
        df["market_category"] = df["market_category"].fillna("temperature")
    # Existing legacy-column migrations stay as-is
    return df
```

Call `_ensure_ledger_schema` wherever the ledger is loaded from disk (e.g., in `settle_paper_trades` and `log_paper_trades`).

- [ ] **Step 5: Run the tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_paper_trading.py::test_log_paper_trades_records_market_category tests/test_paper_trading.py::test_legacy_ledger_migrates_market_category_to_temperature -v`

Expected: both PASS.

- [ ] **Step 6: Write a failing test for `category_breakdown` in summary.json**

Add to `tests/test_paper_trading.py`:

```python
def test_settle_paper_trades_reports_category_breakdown(tmp_path, monkeypatch):
    """Summary must report per-category PnL/ROI/trade counts and a
    daily-PnL Pearson correlation when both categories have settled rows."""
    import pandas as pd
    from src.paper_trading import settle_paper_trades

    ledger = tmp_path / "ledger.csv"
    rows = []
    # Temperature: 5 settled trades across 5 distinct days
    for i in range(5):
        rows.append({
            "trade_id": f"t{i}", "scan_id": "s", "status": "settled",
            "source": "kalshi", "ticker": f"KXHIGHT-{i}", "market_category": "temperature",
            "city": "New York", "market_type": "high",
            "market_date": f"2026-04-{10+i:02d}", "outcome": "Yes", "position_side": "yes",
            "contracts": 1, "entry_price": 0.5, "entry_cost": 0.5,
            "payout": 1.0 if i % 2 == 0 else 0.0,
            "pnl": 0.5 if i % 2 == 0 else -0.5,
            "roi": 1.0 if i % 2 == 0 else -1.0,
            "total_fees": 0, "entry_fee": 0, "settlement_fee": 0,
            "yes_outcome": i % 2 == 0, "settled_at_utc": f"2026-04-{10+i:02d}T23:00:00+00:00",
        })
    # Rain: 5 settled trades on the same 5 days
    for i in range(5):
        rows.append({
            "trade_id": f"r{i}", "scan_id": "s", "status": "settled",
            "source": "kalshi", "ticker": f"KXRAIN-{i}", "market_category": "rain",
            "city": "New York", "market_type": "rain_binary",
            "market_date": f"2026-04-{10+i:02d}", "outcome": "Yes", "position_side": "yes",
            "contracts": 1, "entry_price": 0.5, "entry_cost": 0.5,
            "payout": 1.0 if i == 0 else 0.0,
            "pnl": 0.5 if i == 0 else -0.5,
            "roi": 1.0 if i == 0 else -1.0,
            "total_fees": 0, "entry_fee": 0, "settlement_fee": 0,
            "yes_outcome": i == 0, "settled_at_utc": f"2026-04-{10+i:02d}T23:00:00+00:00",
        })
    pd.DataFrame(rows).to_csv(ledger, index=False)

    summary_path = tmp_path / "summary.json"
    summary = settle_paper_trades(ledger_path=ledger, summary_path=summary_path)

    cb = summary["category_breakdown"]
    assert cb["temperature"]["trade_count"] == 5
    assert cb["rain"]["trade_count"] == 5
    assert -1.0 <= cb["correlation_30d"] <= 1.0
    assert cb["correlation_sample_size"] == 5
```

- [ ] **Step 7: Run the failing test**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_paper_trading.py::test_settle_paper_trades_reports_category_breakdown -v`

Expected: FAIL — `category_breakdown` absent.

- [ ] **Step 8: Add `category_breakdown` computation in settlement**

In `src/paper_trading.py`, find the summary-construction code in `settle_paper_trades`. Add:

```python
def _compute_category_breakdown(settled_df: pd.DataFrame) -> dict:
    if settled_df.empty:
        return {
            "temperature": {"trade_count": 0, "pnl": 0.0, "roi": 0.0, "fees": 0.0},
            "rain": {"trade_count": 0, "pnl": 0.0, "roi": 0.0, "fees": 0.0},
            "correlation_30d": None,
            "correlation_sample_size": 0,
        }
    result = {}
    for category in ("temperature", "rain"):
        sub = settled_df[settled_df["market_category"] == category]
        if sub.empty:
            result[category] = {"trade_count": 0, "pnl": 0.0, "roi": 0.0, "fees": 0.0}
            continue
        entry_cost = sub["entry_cost"].sum()
        pnl = sub["pnl"].sum()
        result[category] = {
            "trade_count": int(len(sub)),
            "pnl": float(pnl),
            "roi": float(pnl / entry_cost) if entry_cost else 0.0,
            "fees": float(sub["total_fees"].sum()),
        }
    # Daily-PnL Pearson correlation (trailing 30 days by market_date)
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=30)).isoformat()
    recent = settled_df[settled_df["market_date"] >= cutoff]
    temp_daily = (
        recent[recent["market_category"] == "temperature"]
        .groupby("market_date")["pnl"].sum()
    )
    rain_daily = (
        recent[recent["market_category"] == "rain"]
        .groupby("market_date")["pnl"].sum()
    )
    merged = temp_daily.to_frame("t").join(rain_daily.to_frame("r"), how="inner")
    if len(merged) >= 2 and merged["t"].std() > 0 and merged["r"].std() > 0:
        corr = float(merged["t"].corr(merged["r"]))
    else:
        corr = None
    result["correlation_30d"] = corr
    result["correlation_sample_size"] = int(len(merged))
    return result
```

Include it in the summary dict returned by `settle_paper_trades`:

```python
summary["category_breakdown"] = _compute_category_breakdown(settled_df)
```

- [ ] **Step 9: Run the test**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_paper_trading.py::test_settle_paper_trades_reports_category_breakdown -v`

Expected: PASS.

- [ ] **Step 10: Run the full paper-trading test file**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_paper_trading.py -v`

Expected: all PASS (legacy tests unaffected).

- [ ] **Step 11: Commit**

```bash
git add src/paper_trading.py tests/test_paper_trading.py
git commit -m "feat(paper_trading): market_category column + category_breakdown correlation"
```

---

## Task 12: Wire rain matcher into `main.py`

**Rationale:** Feature-flagged integration — `enable_rain_vertical` gates the new code path. Reuse existing fetch / archive / policy infrastructure.

**Files:**
- Modify: `main.py`
- Modify: `config.example.json`
- Modify: `tests/test_main_deepseek_integration.py` (or create minimal `tests/test_main_rain_integration.py`)

- [ ] **Step 1: Write failing integration test — rain opportunities flow through**

Create `tests/test_main_rain_integration.py`:

```python
from unittest.mock import patch


def test_run_scan_includes_rain_opportunities_when_enabled(tmp_path):
    """With enable_rain_vertical=true, run_scan returns rain opportunities
    alongside any temperature opportunities."""
    from main import run_scan

    config = {
        "locations": [{"name": "New York", "lat": 40.7, "lon": -74.0}],
        "forecast_hours": 48,
        "min_edge_threshold": 0.1,
        "uncertainty_std_f": 2.0,
        "enable_ensemble": False,
        "enable_calibration": False,
        "enable_hrrr": False,
        "enable_kalshi": True,
        "enable_polymarket": False,
        "enable_rain_vertical": True,
        "rain_watchlist": ["New York"],
        "rain_min_edge_threshold": 0.1,
        "rain_hrrr_blend_horizon_hours": 12,
        "rain_strategy_policy_path": "strategy/rain_policy_v1.json",
        "strategy_policy_path": "strategy/strategy_policy.json",
        "enable_deepseek_worker": False,
        "enable_paper_trading": False,
        "opportunity_archive_enabled": False,
    }

    precip = {"New York": {"daily": [
        {"date": "2026-04-21", "forecast_prob_any_rain": 0.82, "forecast_amount_in": 0.4}
    ]}}
    rain_markets = [{
        "ticker": "KXRAINNYC-26APR21", "outcome": "Yes", "city": "New York",
        "market_date": "2026-04-21", "yes_ask": 0.55, "volume_24h": 1500,
        "close_time": "2026-04-21T23:59:00+00:00",
    }]

    with patch("src.fetch_forecasts.fetch_multi_location", return_value={"New York": {"daily": {"time": ["2026-04-21"], "temperature_2m_max": [70], "temperature_2m_min": [50]}}}), \
         patch("src.fetch_kalshi.fetch_weather_markets", return_value=rain_markets), \
         patch("src.fetch_precipitation.fetch_precipitation_multi", return_value=precip), \
         patch("src.station_truth.archive_forecast_snapshot"):
        result = run_scan(config=config)

    rain_opps = [o for o in result["opportunities"] if o.get("market_category") == "rain"]
    assert len(rain_opps) == 1
    assert rain_opps[0]["ticker"] == "KXRAINNYC-26APR21"
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_main_rain_integration.py -v`

Expected: FAIL — `enable_rain_vertical` not yet handled.

- [ ] **Step 3: Wire the rain pipeline into `run_scan` in `main.py`**

After the existing Kalshi match block (near the `all_opportunities.extend(kalshi_opps)` line), add:

```python
if config.get("enable_rain_vertical", False) and kalshi_markets:
    try:
        from src.fetch_precipitation import (
            fetch_precipitation_multi, fetch_precipitation_ensemble_multi,
        )
        from src.rain_matcher import match_kalshi_rain
        from src.rain_calibration import RainCalibrationManager

        rain_watchlist = config.get("rain_watchlist") or []
        rain_locations = [loc for loc in locations if loc.get("name") in rain_watchlist]
        if rain_locations:
            log.info("Fetching rain forecasts for %d city(ies)...", len(rain_locations))
            precip_forecasts = fetch_precipitation_multi(rain_locations, forecast_hours)
            try:
                ensemble_precip = fetch_precipitation_ensemble_multi(rain_locations, forecast_hours)
                for city, data in ensemble_precip.items():
                    daily_lookup = {d["date"]: d for d in (precip_forecasts.get(city, {}).get("daily") or [])}
                    for ens_day in data.get("daily", []):
                        d = daily_lookup.get(ens_day["date"])
                        if d:
                            d["ensemble_wet_fraction"] = ens_day.get("ensemble_wet_fraction")
                            d["ensemble_amount_std_in"] = ens_day.get("ensemble_amount_std_in")
            except Exception as exc:
                log.warning("Ensemble precip fetch failed: %s", exc)

            rain_cal_mgr = RainCalibrationManager(model_dir=config.get("calibration_model_dir"))
            rain_opps = match_kalshi_rain(
                precip_forecasts,
                kalshi_markets,
                calibration_manager=rain_cal_mgr,
                hrrr_data=None,
                hrrr_blend_horizon_hours=float(config.get("rain_hrrr_blend_horizon_hours", 12)),
                min_edge=float(config.get("rain_min_edge_threshold", 0.15)),
                now_utc=now_utc,
            )
            all_opportunities.extend(rain_opps)
            log.info("Rain: %d opportunities with edge >= %.0f%%",
                     len(rain_opps), float(config.get("rain_min_edge_threshold", 0.15)) * 100)
    except Exception as exc:
        log.warning("Rain vertical failed: %s", exc)
```

Also, before the paper-trade log step, load both policies and filter by category:

```python
from src.strategy_policy import filter_opportunities_for_policy, load_strategy_policy

temp_policy, _ = load_strategy_policy(config.get("strategy_policy_path"))
rain_policy, _ = (
    load_strategy_policy(config.get("rain_strategy_policy_path"))
    if config.get("enable_rain_vertical", False) else (None, None)
)
trade_opportunities = filter_opportunities_for_policy(all_opportunities, temp_policy)
if rain_policy:
    trade_opportunities += filter_opportunities_for_policy(all_opportunities, rain_policy)
```

(Replace the existing single-policy call with this combined version.)

- [ ] **Step 4: Run the integration test**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_main_rain_integration.py -v`

Expected: PASS.

- [ ] **Step 5: Add rain config keys to `config.example.json`**

In `config.example.json`, add near the other `enable_*` flags:

```json
"enable_rain_vertical": false,
"rain_watchlist": [
  "New York", "Chicago", "Boston", "Philadelphia", "Miami",
  "Atlanta", "Houston", "Seattle", "New Orleans", "Washington DC"
],
"rain_hrrr_blend_horizon_hours": 12,
"rain_bankroll_fraction": 0.20,
"rain_strategy_policy_path": "strategy/rain_policy_v1.json",
"rain_min_edge_threshold": 0.15,
"rain_min_volume24hr": 500,
"rain_max_candidates_per_scan": 2,
"rain_calibration_window_days": 90
```

- [ ] **Step 6: Run the entire test suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests -x`

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add main.py config.example.json tests/test_main_rain_integration.py
git commit -m "feat(main): wire rain vertical behind enable_rain_vertical flag"
```

---

## Task 13: Documentation updates

**Files:**
- Modify: `src/CLAUDE.md`, `data/CLAUDE.md`, `strategy/CLAUDE.md`
- Create: `.claude/rules/rain-vertical.md`

- [ ] **Step 1: Add `.claude/rules/rain-vertical.md`**

```markdown
# Rain Vertical Rules

## Scope

P1 ships KXRAIN binary any-rain markets (≥ 0.01 in threshold) only.
Moderate thresholds (≥ 0.25 in) and heavy thresholds (≥ 1.0 in) are P1.5
or later — the `rain_matcher.py` parser explicitly returns None for them.

## Routing

Rain opportunities carry `market_category=rain`. `strategy_policy.py`
filters by category when a policy declares `market_category: "rain"`.
Legacy (temperature) opportunities without `market_category` default to
`"temperature"`.

## Calibration

Two-stage: `LogisticRainCalibrator` (bias correction) + `IsotonicRainCalibrator`
(probability recalibration). Models live at
`data/calibration_models/{city_slug}_rain_binary_{logistic|isotonic}.pkl`
and are retrained weekly against a rolling `rain_calibration_window_days`
(default 90) of archived forecast-vs-actual pairs.

## HRRR Blend

Same-day rain markets (hours_to_settlement < `rain_hrrr_blend_horizon_hours`,
default 12) blend HRRR APCP into the calibrated probability. Ramp weight
linear, cap 0.7.

## Bankroll

Rain trades draw from a separate 20% slice of the Kelly bankroll
(`rain_bankroll_fraction`). Temperature Kelly is untouched.

## Promotion Gate

Paper-only for at least 30 days. Criteria to begin real-money discussion:
1. Rain calibration log-loss strictly less than climatology baseline on holdout
2. Paper ROI ≥ 0 net of fees
3. Pearson correlation of daily settled-PnL series (rain vs temperature) < 0.4
```

- [ ] **Step 2: Update `data/CLAUDE.md`**

Add a new directory entry to the layout table:

```markdown
| `precip_archive/` | Yes | Per-city precipitation forecast snapshots (Open-Meteo + backfilled Previous Runs) |
```

And a schema block:

```markdown
### precip_archive/*.csv
```
as_of_utc, date, forecast_prob_any_rain, forecast_amount_in,
ensemble_wet_fraction, ensemble_amount_std_in,
forecast_model, forecast_lead_days, forecast_source
```

### calibration_models/ (rain additions)
`{city_slug}_rain_binary_logistic.pkl` — sklearn LogisticRegression
`{city_slug}_rain_binary_isotonic.pkl` — sklearn IsotonicRegression
```

- [ ] **Step 3: Update `src/CLAUDE.md` module table**

Add rows for the new modules:

```markdown
| `fetch_precipitation.py` | Open-Meteo precip (deterministic + ensemble) + HRRR APCP extraction |
| `rain_matcher.py` | Parse KXRAIN binary outcomes, compute probability & edge |
| `rain_calibration.py` | Logistic + Isotonic rain-probability calibrators, RainCalibrationManager |
```

- [ ] **Step 4: Update `strategy/CLAUDE.md`**

Add a section:

```markdown
## Rain Policy (v1, 2026-04-20)

`rain_policy_v1.json` controls the rain-vertical opportunity filter. It
declares `market_category: "rain"`, which tells the category-aware filter
in `strategy_policy.py` to route only `market_category=rain` opportunities
through this policy. Temperature opportunities continue to flow through
`strategy_policy.json` (v4) unchanged.

Starting thresholds (conservative):
- `min_abs_edge`: 0.15
- `min_volume24hr`: 500 (lower than temp's 2000 — rain markets are smaller)
- `max_candidates_per_scan`: 2
- `max_hours_to_settlement`: 24
- `allowed_market_types`: ["rain_binary"]
- `allowed_position_sides`: ["yes"]
```

- [ ] **Step 5: Commit**

```bash
git add .claude/rules/rain-vertical.md src/CLAUDE.md data/CLAUDE.md strategy/CLAUDE.md
git commit -m "docs: document rain vertical rules, schemas, policies"
```

---

## Task 14: End-to-end smoke test + go-live

**Rationale:** Before flipping `enable_rain_vertical` on in production config, a live dry run confirms the whole pipeline plays together: fetch → calibrate → match → policy-filter → paper-log → settle.

- [ ] **Step 1: Confirm full test suite is green**

Run: `.\.venv\Scripts\python.exe -m pytest tests -v`

Expected: 0 failures, 0 errors.

- [ ] **Step 2: Run a single live scan with rain vertical enabled via CLI override**

Create `scripts/smoke_test_rain_vertical.py`:

```python
"""One-shot smoke test: force enable_rain_vertical=true and run one scan."""
from __future__ import annotations

from main import load_config, run_scan


def main() -> None:
    config = load_config()
    config["enable_rain_vertical"] = True
    # Keep paper-trading off for the dry run so we don't pollute the ledger.
    config["enable_paper_trading"] = False
    result = run_scan(config=config)

    rain_opps = [o for o in result["opportunities"] if o.get("market_category") == "rain"]
    print(f"total opportunities: {len(result['opportunities'])}")
    print(f"rain opportunities:  {len(rain_opps)}")
    for o in rain_opps[:5]:
        print(f"  {o['ticker']:<30} {o['city']:<14} "
              f"edge={o['edge']:+.3f} price={o['market_price']:.2f}")


if __name__ == "__main__":
    main()
```

Run: `.\.venv\Scripts\python.exe scripts/smoke_test_rain_vertical.py`

Expected: `rain opportunities: N` with N ≥ 0 and no errors. Inspect the first few rain opportunities for plausibility (prices in [0, 1], edges within ±0.5, cities drawn from the watchlist).

- [ ] **Step 3: Flip the flag on in local `config.json` (live deployment)**

Manually edit `config.json`:

```json
"enable_rain_vertical": true
```

This is a deliberate manual step — live config changes should not be automated by this plan.

- [ ] **Step 4: Run one production-shape scan**

Run: `.\.venv\Scripts\python.exe main.py --once --kalshi-only`

Expected: logs include a "Rain: N opportunities with edge >= 15%" line. Paper ledger (if paper-trading is enabled in config.json) contains `market_category=rain` rows. `summary.json` includes `category_breakdown`.

- [ ] **Step 5: Document the go-live timestamp**

Create `docs/superpowers/plans/2026-04-20-rain-vertical-p1-go-live.md`:

```markdown
# Rain Vertical P1 — Go-Live Log

**Go-live UTC:** <fill in from scan log>
**Commit at go-live:** <fill in>
**Evaluation window:** 30 trading days from go-live
**Promotion criteria check date:** <go-live + 30 days>
```

- [ ] **Step 6: Commit the smoke script and go-live log template**

```bash
git add scripts/smoke_test_rain_vertical.py docs/superpowers/plans/2026-04-20-rain-vertical-p1-go-live.md
git commit -m "chore(rain): smoke test + go-live tracking doc"
```

---

## Post-P1: Evaluation Gate

After 30 trading days of paper activity:

1. Run `.\.venv\Scripts\python.exe evaluate_rain_calibration.py --days 400 --holdout-days 30` and save the report.
2. Inspect `data/paper_trades/summary.json` `category_breakdown`:
   - Rain `roi` ≥ 0?
   - `correlation_30d` < 0.4?
3. Compare rain `brier_calibrated` vs `brier_climatology` in the eval report: strictly less for most cities?
4. If all three pass, begin P2 (temperature SELL-side / bucket rehab via tail-calibration). If any fail, diagnose and iterate on rain v2 before starting P2.

---

## Self-Review Notes

**Spec coverage:** all 11 rollout-order items in the spec map to tasks 1–14
(backfill → discovery → precip fetcher → matcher → calibration → policy →
ledger → main → docs → smoke → evaluation gate). The LogisticRainCalibrator
interface is deliberately generic over binary outcomes (per the P2 hook in the
spec). The `models` parameter in `fetch_precipitation_multi` and the
extensibility of `RainCalibrationManager` cover the P3 hook.

**Placeholder scan:** all tasks carry executable commands and concrete code.
No "TBD", no "fill in details."

**Type consistency:** `market_category` appears in four files (paper_trading,
strategy_policy, rain_matcher opportunity dicts, ledger columns) with
identical values (`"rain"` / `"temperature"`). `RainCalibrationManager`
method signature matches its single caller in `rain_matcher.py`
(`calibrate_rain_probability(city, raw_prob)` returning
`{calibrated_prob, forecast_calibration_source, probability_calibration_source}`).
`_slugify_city` is reused from `station_truth.py` across all new modules.
