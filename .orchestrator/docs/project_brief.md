# Project Brief

This repository predicts short-horizon weather outcomes for a 20-city watchlist
and scans live Kalshi and Polymarket weather contracts for edge.

## Core Flows

- fetch deterministic and optional ensemble forecasts
- fetch market listings
- map forecast distributions to contract probabilities
- archive forecast snapshots and station truth
- train and apply calibration models
- optionally blend same-day HRRR guidance

## Files To Read First

1. `README.md`
2. `UPGRADE_PLAN.md`
3. `main.py`
4. `src/config.py`
5. `src/matcher.py`
6. `src/calibration.py`

## Constraints

- repository structure is flexible and may be reworked if the tools need a
  better layout
- preserve API keys, credential files, and local config across any rework
- local secrets stay out of git
- archived data and trained models are valuable and should not be bulk rewritten
- tests under `tests/` should be updated when behavior changes
- orchestration assets should stay inside `.orchestrator/`
