# Worklog task_0006

## Task

task_0006

## Entries

### 2026-04-03T19:40:37+00:00

- route: plan
- provider: codex

#### Prompt

Plan how to reorganize this repo around the weather tools while preserving secrets and runtime data.

#### Summary

## Objective

Reorganize the weather repository around the active weather tooling while preserving API keys, local credential material, runtime data, trained calibration artifacts, and paper-trading state.

## Constraints

- Preserve `config.json`, `config.local.json`, `api-credentials.txt`, `*.pem`, and `*.key`.
- Preserve `data/forecast_archive/`, `data/station_actuals/`, `data/calibration_models/`, and any paper-trading ledger/summary files.
- Keep `.orchestrator/` as the durable control plane.
- Treat `codex_loop/` and `codex_task/` as legacy experiments to migrate or archive, not as the target structure.

## Recommended Structure

- Keep the repo root thin: `README.md`, top-level entrypoints, config templates, and orchestration files only.
- Consolidate reusable weather logic into one package under `src/`.
- Move operational one-off scripts into a dedicated `scripts/` or `jobs/` area if they are not library code.
- Preserve `data/` as the durable runtime and model area, but split clearly between archives, models, caches, and test runs.
- Keep local secret-bearing files in gitignored root or a dedicated local-only config area, and document them instead of relocating them aggressively.
- Either archive or retire `codex_loop/` and `codex_task/` once their useful behavior has been folded into `.orchestrator/`.

## Phased Plan

1. Inventory immutable assets.
   Record every secret/config/runtime path that must survive the rework unchanged or with a controlled migration path.
2. Define the target runtime layout.
   Decide which modules stay in `src/`, which scripts become operational entrypoints, and which legacy folders become archive candidates.
3. Separate durable data from disposable outputs.
   Keep archives and models stable, and isolate caches/test runs so future cleanup is safe.
4. Migrate orchestration responsibilities.
   Keep `.orchestrator/` as the active wrapper and stop relying on the legacy agent-loop folders for day-to-day control.
5. Update docs and command paths.
   Rewrite the read order, commands, and migration notes so the new layout is the documented canonical shape.
6. Validate end-to-end.
   Confirm scanner runs, calibration training runs, paper-trade settlement runs, and tests still pass against the reorganized structure.

## Immediate Next Task

Create a migration inventory document that maps:
- secrets/config paths that must be preserved
- durable data/model paths that must be preserved
- active code modules that should remain first-class
- legacy folders that can be archived or absorbed

Current open tasks:
- task_0006: Plan how to reorganize this repo around the weather tools while prese... [ready]
- task_0005: Plan how to reorganize this repo around the weather tools while prese... [planned]
- task_0004: Plan how to reorganize this repo around the weather tools while prese... [planned]
- task_0003: Plan how to reorganize this repo around the weather tools while prese... [planned]
- task_0002: Plan how to reorganize this repo around the weather tools while prese... [planned]

