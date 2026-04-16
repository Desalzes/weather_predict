# Weather Scanner Dashboard Integration Brief

Add a "Weather" monitoring page to the existing Polymarket dashboard at
`/opt/polymarket/dashboard/` on VPS `95.216.159.10`.

## Existing Dashboard Architecture

- **Framework:** Flask (`app.py`, ~987 lines)
- **Template:** Single `templates/index.html` (1321 lines, single-page app)
- **Auth:** HTTP Basic Auth (`DASH_USER`/`DASH_PASS` env vars)
- **Port:** 8088 via systemd `polymarket-dashboard.service`
- **Pattern:** Sidebar nav buttons with `data-page` attrs toggle `.page` divs.
  Top bar shows live metrics. Pages fetch from `/api/*` routes. Auto-refresh
  on a 15s countdown timer.

## Design System (match exactly)

```css
:root {
  --bg: #0d1117;    --bg2: #161b22;   --bg3: #21262d;
  --border: #30363d; --text: #e6edf3;  --text2: #8b949e;
  --green: #3fb950;  --red: #f85149;   --blue: #58a6ff;
  --yellow: #d29922; --orange: #db6d28; --purple: #bc8cff;
}
```

Components to reuse: `.card`, `.card-header`, `.card-body`, `table` styles,
`.badge-*`, `.metric`, `.status-dot`, `.log-viewer`, `.side-buy`/`.side-sell`.

## What to Build

### 1. Sidebar nav button

Add after the "Edit" button in the sidebar:

```html
<button class="nav-btn" data-page="weather">
  <span class="icon">&#9729;</span> Weather
</button>
```

### 2. Top bar metrics (weather-specific)

When the Weather page is active, optionally show these in the top bar or as
a sub-header within the page:

| Metric | Source | Format |
|--------|--------|--------|
| Total PnL | `summary.json` -> `total_pnl` | `$+0.32` green/red |
| ROI | `summary.json` -> `roi` | `47.1%` green/red |
| Win Rate | `summary.json` -> `win_rate` | `11.1%` |
| Open Trades | `summary.json` -> `open_trades` | `12` |
| Settled | `summary.json` -> `settled_trades` | `9` |
| Last Scan | `journalctl` last scan timestamp | `2m ago` |

### 3. Page: `page-weather`

Four cards:

#### Card 1: Paper Trade Summary

Show the `forecast_calibration_source_breakdown` from `summary.json` as a
table:

| Route | Total | Open | Settled | PnL | ROI | Win Rate |
|-------|-------|------|---------|-----|-----|----------|
| emos | 13 | 10 | 3 | -$0.08 | -100% | 0% |
| raw_selective_fallback | 2 | 2 | 0 | -- | -- | -- |
| legacy_unknown | 6 | 0 | 6 | +$0.40 | 67% | 17% |

Color PnL/ROI with `.positive`/`.negative` classes.

If `missing_required_truth_blocker` is true, show a yellow badge:
`<span class="badge badge-yellow">Truth Blocker: N dates missing</span>`

#### Card 2: Open Positions

Table of open trades from `ledger.csv` where `status == "open"`:

| City | Type | Outcome | Side | Entry | Edge | Route | Hours |
|------|------|---------|------|-------|------|-------|-------|

Color BUY green, SELL red. Sort by `abs_edge` descending.

#### Card 3: Recent Settled Trades

Table of last 20 settled trades from `ledger.csv` where `status == "settled"`:

| City | Type | Outcome | Side | Entry | Actual | PnL | ROI | Route |
|------|------|---------|------|-------|--------|-----|-----|-------|

Color PnL green/red.

#### Card 4: Scanner Log

Last 30 lines from `journalctl -u weather-scanner --no-pager -n 30` in a
`.log-viewer` div. Show scan timestamps, opportunity counts, and paper trade
activity.

### 4. Flask API route

Add to `app.py` (or create a separate `weather_routes.py` blueprint):

```python
WEATHER_DIR = Path("/opt/weather")
WEATHER_SUMMARY = WEATHER_DIR / "data" / "paper_trades" / "summary.json"
WEATHER_LEDGER = WEATHER_DIR / "data" / "paper_trades" / "ledger.csv"

@app.route("/api/weather")
@auth_required
def api_weather():
    # Read summary.json
    summary = {}
    if WEATHER_SUMMARY.exists():
        summary = json.loads(WEATHER_SUMMARY.read_text(encoding="utf-8"))

    # Read ledger.csv
    trades = []
    if WEATHER_LEDGER.exists():
        trades = read_csv_dicts(WEATHER_LEDGER)

    open_trades = [t for t in trades if t.get("status") == "open"]
    settled_trades = [t for t in trades if t.get("status") == "settled"]

    # Get last scanner log lines
    try:
        log_result = subprocess.run(
            ["journalctl", "-u", "weather-scanner", "--no-pager", "-n", "30"],
            capture_output=True, text=True, timeout=5
        )
        log_lines = log_result.stdout.strip().split("\n") if log_result.stdout else []
    except Exception:
        log_lines = ["(log unavailable)"]

    # Timer status
    try:
        timer_result = subprocess.run(
            ["systemctl", "list-timers", "weather-*", "--no-pager"],
            capture_output=True, text=True, timeout=5
        )
        timer_status = timer_result.stdout.strip()
    except Exception:
        timer_status = "(unknown)"

    return jsonify({
        "summary": summary,
        "open_trades": open_trades[-50:],
        "settled_trades": settled_trades[-20:],
        "log_lines": log_lines,
        "timer_status": timer_status,
    })
```

### 5. JavaScript fetch + render

Follow the existing pattern in the dashboard:

```javascript
async function loadWeather() {
  const resp = await fetch('/api/weather');
  const data = await resp.json();
  // Render summary metrics
  // Render route breakdown table
  // Render open positions table
  // Render settled trades table
  // Render log viewer
}
```

Hook into the existing auto-refresh cycle (the dashboard already has a 15s
countdown timer that calls refresh functions).

## Data File Locations on VPS

```
/opt/weather/data/paper_trades/summary.json   <- PnL, ROI, route breakdown
/opt/weather/data/paper_trades/ledger.csv     <- all trades (open + settled)
/opt/weather/logs/                            <- file-based logs (optional)
```

Systemd journal: `journalctl -u weather-scanner` / `journalctl -u weather-settle`

## summary.json Schema

```json
{
  "total_trades": 21,
  "open_trades": 12,
  "settled_trades": 9,
  "total_pnl": 0.32,
  "roi": 0.4706,
  "win_rate": 0.1111,
  "total_fees": 0.0,
  "forecast_calibration_source_breakdown": {
    "emos": { "total_trades": 13, "open_trades": 10, "settled_trades": 3,
              "total_pnl": -0.08, "roi": -1.0, "win_rate": 0.0 },
    "legacy_unknown": { ... },
    "raw_selective_fallback": { ... }
  },
  "missing_required_truth_blocker": false,
  "missing_required_truth_trade_count": 0,
  "settlement_cutoff_date": "2026-04-15"
}
```

## ledger.csv Key Columns

For open trades table: `city, market_type, outcome, position_side, entry_price,
edge, forecast_calibration_source, hours_to_settlement, market_date`

For settled trades table: add `actual_value_f, pnl, roi, settled_at_utc`

## Notes

- The Weather scanner runs on the same VPS but in `/opt/weather/` with its own
  venv. The dashboard at `/opt/polymarket/dashboard/` just reads Weather's data
  files — no cross-venv imports needed.
- The Weather scanner runs every 30 min via systemd timer, not continuously.
  Settlement runs daily at 6am UTC.
- The existing `read_csv_dicts()` helper in `app.py` works for reading
  `ledger.csv`.
