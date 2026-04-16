# data/ — Runtime Data and Model Artifacts

## Directory Layout

| Directory | Committed | Purpose |
|-----------|-----------|---------|
| `calibration_models/` | Yes | Pickled EMOS + isotonic models (`{city}_{type}_{model}.pkl`) |
| `forecast_archive/` | Yes | Per-city forecast history CSVs |
| `station_actuals/` | Yes | Per-city observed truth CSVs |
| `paper_trades/` | Yes | Trade ledger + summary |
| `deepseek_worker/` | Yes | Review state + decision artifacts |
| `hrrr_cache/` | No (.gitignored) | Herbie HRRR grib cache |
| `sandbox_test_dir/`, `test_runs/`, `tmp*` | No | Test/temp artifacts, safe to clean |

## Schemas

### forecast_archive/*.csv
```
as_of_utc, date, forecast_high_f, forecast_low_f, ensemble_high_std_f,
ensemble_low_std_f, forecast_model, forecast_lead_days, forecast_source
```

### station_actuals/*.csv
```
date, tmax_f, tmin_f, precip_in, precip_trace, cli_station,
source_url, city, source, archive_version
```

### paper_trades/ledger.csv
Key columns: `trade_id, scan_id, status (open/settled), source, ticker,
city, market_type, market_date, outcome, position_side (yes/no),
entry_price, entry_cost, our_probability, raw_probability, edge,
forecast_calibration_source, probability_calibration_source,
forecast_blend_source, hours_to_settlement, fee_model, entry_fee,
settlement_fee, total_fees, actual_value_f, payout, pnl, roi`

### paper_trades/summary.json
Top-level fields: `total_trades, open_trades, settled_trades, total_pnl,
roi, total_fees, forecast_calibration_source_breakdown,
missing_required_truth_blocker, missing_required_truth_dates,
settlement_cutoff_date`

### calibration_models/
Naming: `{city}_{market_type}_{model_type}.pkl`
- model_type: `emos` (LinearRegression) or `isotonic` (IsotonicRegression)
- Cities: 17 cities (slug form, e.g., `las_vegas`, `new_york`)
- Market types: `high`, `low`

### deepseek_worker/
- `state.json` — scan count, review runs, last review status
- `reviews/` — per-review JSON decision artifacts

## Safety Rules

- **Never delete** forecast_archive, station_actuals, calibration_models, or
  paper_trades without explicit user request.
- `hrrr_cache/` is local cache — safe to purge for disk space.
- `sandbox_test_dir/`, `test_runs/`, `tmp*` directories are transient — safe
  to clean.
- Calibration models are retrained via `train_calibration.py`; manual edits to
  `.pkl` files will be overwritten.
