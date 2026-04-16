# Weather Project Agent Guide

This repository is a short-horizon weather prediction and weather-market
scanning system for Kalshi and Polymarket. Treat it as a production-style data
pipeline with local-only secrets and large runtime artifacts.

## Read First

1. `README.md`
2. `UPGRADE_PLAN.md`
3. `main.py`
4. `src/config.py`
5. The specific `src/*.py` modules relevant to the task

## Important Paths

- `main.py`: scanner entrypoint
- `backfill_training_data.py`: actuals + archive backfill
- `train_calibration.py`: calibration training entrypoint
- `src/`: application logic
- `tests/`: regression and paper-trading coverage
- `data/forecast_archive/`: historical forecast archive
- `data/station_actuals/`: station truth history
- `data/calibration_models/`: trained model artifacts
- `codex_loop/` and `codex_task/`: earlier orchestration experiments, not the
  default control plane for this repo
- `.orchestrator/`: project-local orchestration wrapper

## Commands

Use the repo-local virtualenv when running Python commands.

```powershell
.\.venv\Scripts\python.exe main.py --once
.\.venv\Scripts\python.exe main.py --settle-paper-trades
.\.venv\Scripts\python.exe backfill_training_data.py --days 365
.\.venv\Scripts\python.exe train_calibration.py
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe .orchestrator\manage.py doctor
.\.venv\Scripts\python.exe .orchestrator\manage.py status
```

## Safety Rules

- Never commit or rewrite local secrets such as `config.json`,
  `config.local.json`, `api-credentials.txt`, `*.pem`, or `*.key`.
- Preserve API keys, local credentials, and config material if the repository
  is reorganized. Those are more important than preserving the current folder
  layout.
- Do not delete or bulk rewrite archived data, calibration models, or paper
  trading ledgers unless the user explicitly asks for it.
- Treat `data/hrrr_cache/` and runtime output folders as local cache, not source
  code.
- When changing forecast matching, calibration, or settlement logic, inspect the
  corresponding tests first and update them if behavior changes.
- Keep new orchestration assets inside `.orchestrator/`, `.claude/`, `.codex/`,
  or `.agents/` instead of mixing them into the weather pipeline.

## Working Agreement

- Prefer small, bounded tasks with explicit output files.
- Store orchestration state under `.orchestrator/`.
- Treat the current repository shape as flexible. It can be reworked to better
  fit the active toolchain as long as secrets and durable data are preserved.
- Keep shared repo rules in this file. Put Claude-specific deltas in
  `CLAUDE.md`.
