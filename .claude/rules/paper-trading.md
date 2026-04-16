# Paper Trading Rules

## Fee Model

Model identifier: `flat_fee_per_contract_v1`

```
entry_cost  = (entry_price * contracts) + entry_fee
pnl         = payout - (entry_price * contracts) - total_fees
roi         = pnl / entry_cost
total_fees  = entry_fee + settlement_fee
```

Config keys: `paper_trade_entry_fee_per_contract`,
`paper_trade_settlement_fee_per_contract` (both default 0.0).

## Settlement Flow

1. `main.py --settle-paper-trades` triggers a bounded truth refresh.
2. Refresh targets only the cities and open-trade dates eligible to settle.
3. Primary source: NCEI CDO (when `ncei_api_token` is configured).
4. Fallback: NWS CLI archive for still-missing city/date windows.
5. Ledger is updated in place; `summary.json` is rewritten.

## Route Attribution

Every paper trade persists these metadata columns for post-hoc analysis:

- `forecast_calibration_source`: emos | raw | raw_selective_fallback
- `probability_calibration_source`: isotonic | raw
- `forecast_blend_source`: open-meteo | open-meteo+hrrr
- `hours_to_settlement`, `open_meteo_forecast_value_f`, `raw_forecast_value_f`

Legacy rows missing these columns are backfilled as `legacy_unknown`.

`summary.json` includes `forecast_calibration_source_breakdown` which splits
PnL, fees, ROI, and win rate by route — use this to evaluate whether specific
calibration paths are profitable.

## Truth Blocker Reporting

When station actuals are missing for open trades within the settlement cutoff,
`summary.json` exposes:

- `missing_required_truth_blocker: true`
- `missing_required_truth_trade_count` / `missing_required_truth_date_count`
- `missing_required_truth_dates` (per-city, per-date detail)
- `missing_required_truth_city_windows` (grouped ranges)

This prevents false "clean" summaries when settlement is actually blocked.

## Deduplication

The scanner records at most one open position per unique market side
(source + ticker + date + outcome + direction). Duplicate attempts within the
same scan are silently skipped.
