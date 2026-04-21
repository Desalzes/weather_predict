# strategy/ — Trading Policy

## What It Controls

`strategy_policy.json` defines the opportunity selection and execution rules
consumed by both the DeepSeek paper-gate worker and the between-review fallback
filter in `src/strategy_policy.py`.

## Current Policy (v2, 2026-04-05)

**Selection thresholds:**
- `min_abs_edge`: 0.15 (15% minimum absolute edge)
- `min_volume24hr`: 2000 ($2k daily volume floor)
- `max_candidates_per_scan`: 3
- `max_hours_to_settlement`: 24 (day-ahead only)
- `allowed_market_types`: high, low
- `blocked_cities`: Washington DC (pending more settled evidence)
- `sources`: kalshi only

**Execution limits:**
- `max_contracts_per_trade`: 1
- `max_new_orders_per_day`: 3
- `max_order_cost_dollars`: $10.00
- `time_in_force`: fill_or_kill

## How It's Used

1. **DeepSeek review scans** — the full policy JSON is serialized into the
   DeepSeek prompt via `compact_policy_for_prompt()`. The LLM decides
   approve/watch/reject per opportunity.
2. **Between-review fallback** — `strategy_policy.py` applies the selection
   filters mechanically to the opportunity list. No LLM involved.
3. **Paper trading** — approved opportunities are logged to the ledger with
   routing metadata for post-hoc attribution.

## Rules

- Policy changes should be **evidence-backed** by regression diagnostics from
  `evaluate_calibration.py --days 400 --holdout-days 30`.
- Thresholds were tightened from v1 (8% edge, 500 volume) to v2 based on
  paper-trade results and calibration holdout analysis.
- v3 added `allowed_settlement_rules: ["lte", "gt"]` to block bucket markets
  (between_inclusive) after 8/9 bucket trades lost.
- The `blocked_cities` list is a soft gate — cities can be unblocked once enough
  settled paper evidence accumulates.

## Polymarket Enablement

Polymarket integration is fully coded but disabled (`enable_polymarket: false`).
To enable:

1. Set `enable_polymarket: true` in `config.json`
2. Add `"polymarket"` to `strategy_policy.json` `selection.sources`
3. Note: Polymarket markets use different outcome formats (range strings like
   "60 to 70", "80 or above") parsed by `_parse_outcome_range()` in
   `fetch_polymarket.py`
4. The `allowed_settlement_rules` filter may need adjustment — Polymarket
   opportunities may not carry `settlement_rule` in the same format as Kalshi
5. Recommend paper-trading Polymarket alongside Kalshi for 2+ weeks before
   giving it real weight in the policy

## Rain Policy (v1, 2026-04-21)

`rain_policy_v1.json` controls the rain-vertical opportunity filter. It
declares `market_category: "rain"`, which tells the category-aware filter
in `strategy_policy.py` to route only `market_category=rain` opportunities
through this policy. Temperature opportunities continue to flow through
`strategy_policy.json` (v4) unchanged.

**P1 starting thresholds (NYC-only):**
- `min_abs_edge`: 0.15 (mirrors temp v4)
- `min_volume24hr`: 0 (NYC rain markets have blank yes_ask / zero volume;
  gating on volume would produce zero trades)
- `max_candidates_per_scan`: 2
- `max_hours_to_settlement`: 48 (rain markets close at 23:59 UTC; 24h
  horizon would cut out half the tradable window)
- `allowed_market_types`: `["rain_binary"]`
- `allowed_position_sides`: `["yes"]` (BUY-only, mirrors temp v4 posture)
- `allowed_cities`: `["New York"]` (Task 6 discovery — other watchlist
  cities have no open KXRAIN markets)

Re-evaluate thresholds after 30 days of paper-trade evidence, especially
`min_volume24hr` once NYC markets show real volume.
