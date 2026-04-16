# Calibration Pipeline Rules

## Routing Order

1. **EMOS** (bias correction) — applied first when a trained model exists for
   the city-market pair, unless the pair is in the selective fallback set.
2. **Isotonic** (probability recalibration) — applied after EMOS when a trained
   model exists. Bypassed for intraday markets (< 18 h to settlement) where
   HRRR blending already adjusts the forecast.
3. **HRRR blend** — blended into the forecast for same-day markets within the
   `hrrr_blend_horizon_hours` window (default 18 h). Uses Herbie + cfgrib.

## Selective Raw Fallback

These 5 city-market pairs bypass EMOS because raw forecasts outperform
calibrated ones in chronological holdout evaluation:

- Boston low
- Minneapolis low
- Philadelphia low
- New Orleans high
- San Francisco low

Trades on these pairs are labeled `forecast_calibration_source=raw_selective_fallback`.

**Validation requirement:** changes to this list must be validated by running:
```powershell
.\.venv\Scripts\python.exe evaluate_calibration.py --days 400 --holdout-days 30
```
Check the `comparison.targeted_pairs` section and confirm the new pair set
improves or preserves holdout metrics (MAE, Brier score, probability bias).

## Key Constraints

- `_MIN_SPREAD_F = 1.0` — EMOS spread term is floored to prevent degenerate
  narrow distributions.
- `_MIN_PROB = 0.001`, `_MAX_PROB = 0.999` — all probability outputs are
  clamped to avoid 0/1 edge cases.
- `_ENSEMBLE_SIGMA_FLOOR_F = 1.0`, `_ENSEMBLE_SIGMA_CAP_F = 6.0` — ensemble σ
  is bounded in `matcher.py` before probability computation.

## Retraining

```powershell
.\.venv\Scripts\python.exe train_calibration.py
```

Uses a rolling `calibration_window_days` (default 90) of forecast-vs-actual
pairs. Models are persisted to `data/calibration_models/` as pickle files.
