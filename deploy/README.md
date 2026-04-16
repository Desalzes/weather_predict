# VPS Deployment

Deploy the Weather scanner to the Hetzner VPS (`95.216.159.10`) alongside the
existing Polymarket copy-trader bot.

## Layout on VPS

```
/opt/polymarket/          <- existing Polymarket bot (untouched)
/opt/weather/             <- Weather scanner (new)
  .venv/                  <- isolated Python venv
  main.py, src/, ...      <- application code
  config.json             <- local config with API keys (not deployed)
  api-credentials.txt     <- Kalshi RSA key (not deployed)
  data/                   <- forecast archive, models, paper trades
  logs/                   <- scanner logs
  deploy/                 <- this directory
```

## First-Time Deploy

```powershell
# From your local machine:
.\deploy\deploy_to_vps.ps1

# Then SSH in to add secrets:
ssh root@95.216.159.10
nano /opt/weather/config.json          # add API keys
nano /opt/weather/api-credentials.txt  # paste Kalshi RSA private key

# Start the timers:
systemctl start weather-scanner.timer weather-settle.timer
```

## What Gets Deployed

- Source code (`main.py`, `src/`, `strategy/`)
- Config template (`config.example.json`)
- Trained models (`data/calibration_models/`)
- Historical data (`data/forecast_archive/`, `data/station_actuals/`)
- Deploy scripts and systemd units

## What Does NOT Get Deployed

- `config.json`, `api-credentials.txt` (secrets)
- `.venv/` (rebuilt on server)
- `data/hrrr_cache/` (local cache)
- `.git/`, tests, orchestrator, legacy dirs

## Systemd Services

| Unit | Type | Schedule | Purpose |
|------|------|----------|---------|
| `weather-scanner.service` | oneshot | via timer | Run `main.py --once` |
| `weather-scanner.timer` | timer | every 30 min | Trigger scanner |
| `weather-settle.service` | oneshot | via timer | Run `main.py --settle-paper-trades` |
| `weather-settle.timer` | timer | daily 6am UTC | Trigger settlement |

## Subsequent Deploys

```powershell
# Code-only update (skip venv rebuild):
.\deploy\deploy_to_vps.ps1 -SkipSetup

# Full redeploy (rebuild venv + reinstall systemd):
.\deploy\deploy_to_vps.ps1
```

## Monitoring

```bash
# Timer status
systemctl list-timers weather-*

# Recent scanner output
journalctl -u weather-scanner --since today

# Recent settlement output
journalctl -u weather-settle --since today

# Manual scan
cd /opt/weather && .venv/bin/python main.py --once

# Paper trade summary
cat /opt/weather/data/paper_trades/summary.json | python3 -m json.tool

# Evaluate calibration
cd /opt/weather && .venv/bin/python evaluate_calibration.py --days 400 --holdout-days 30
```

## Disk Usage

Estimated ~700 MB without HRRR cache, ~1-2 GB with HRRR enabled.
The Polymarket bot uses a separate venv so there's no dependency conflict.
