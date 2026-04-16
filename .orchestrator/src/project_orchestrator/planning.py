from __future__ import annotations

import sqlite3


def _looks_like_acknowledgement(text: str) -> bool:
    lowered = text.lower()
    if not lowered.strip():
        return True
    required_headers = ("objective", "constraints", "recommended structure", "phased plan", "immediate next task")
    if all(header in lowered for header in required_headers):
        return False
    return True


def _open_task_lines(tasks: list[sqlite3.Row]) -> list[str]:
    lines: list[str] = []
    for row in tasks[:5]:
        lines.append(f"- {row['id']}: {row['title']} [{row['status']}]")
    return lines


def synthesize_plan(prompt: str, tasks: list[sqlite3.Row]) -> str:
    open_task_lines = _open_task_lines(tasks)
    open_task_block = "\n".join(open_task_lines) if open_task_lines else "- none"

    lowered = prompt.lower()
    is_reorg = any(token in lowered for token in ("reorganize", "reorganise", "rework", "restructure", "layout"))

    if is_reorg:
        return f"""## Objective

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
{open_task_block}
"""

    return f"""## Objective

{prompt}

## Constraints

- Keep the task bounded and repo-specific.
- Preserve local secrets, runtime data, and trained artifacts.
- Keep orchestration state inside `.orchestrator/`.

## Recommended Structure

- Use the existing weather code as the source of truth for domain behavior.
- Prefer a thin repo root, a coherent `src/` package, clear operational entrypoints, and isolated runtime data.

## Phased Plan

1. Inspect the relevant modules and current task state.
2. Define the smallest useful change or investigation slice.
3. Create or update bounded task artifacts.
4. Execute or review the slice with validation notes.
5. Fold durable learnings back into `.orchestrator/`.

## Immediate Next Task

Turn this request into one bounded executable task with explicit reads, constraints, and validation criteria.

Current open tasks:
{open_task_block}
"""


def finalize_plan_output(prompt: str, provider_output: str, tasks: list[sqlite3.Row]) -> str:
    if not _looks_like_acknowledgement(provider_output):
        return provider_output
    return synthesize_plan(prompt, tasks)
