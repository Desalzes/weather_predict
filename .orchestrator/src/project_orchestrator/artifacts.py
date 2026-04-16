from __future__ import annotations

from pathlib import Path
import sqlite3

from .config import ProjectPaths
from .db import utc_now


def worklog_path(paths: ProjectPaths, task_id: str) -> Path:
    suffix = task_id.split("_", 1)[1] if "_" in task_id else task_id
    return paths.worklogs_dir / f"worklog_{suffix}.md"


def append_worklog_entry(
    paths: ProjectPaths,
    task_id: str,
    route: str,
    provider: str,
    prompt: str,
    summary: str,
) -> Path:
    path = worklog_path(paths, task_id)
    timestamp = utc_now()

    if not path.exists():
        header = (
            f"# Worklog {task_id}\n\n"
            f"## Task\n\n{task_id}\n\n"
            "## Entries\n"
        )
        path.write_text(header, encoding="utf-8")

    entry = (
        f"\n### {timestamp}\n\n"
        f"- route: {route}\n"
        f"- provider: {provider}\n\n"
        "#### Prompt\n\n"
        f"{prompt}\n\n"
        "#### Summary\n\n"
        f"{summary}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)

    return path


def write_master_plan(
    paths: ProjectPaths,
    prompt: str,
    provider: str,
    summary: str,
    tasks: list[sqlite3.Row],
) -> Path:
    lines = [
        "# Master Plan",
        "",
        f"Last updated: {utc_now()}",
        "",
        "## Steering Notes",
        "",
        "- Repository structure is flexible and may be reworked to fit the active toolchain.",
        "- Preserve API keys, local credentials, and config material during any reorganization.",
        "- Keep durable orchestration state inside `.orchestrator/`.",
        "",
        "## Current Request",
        "",
        prompt,
        "",
        "## Latest Plan",
        "",
        summary,
        "",
        "## Open Tasks",
        "",
    ]

    if tasks:
        for row in tasks:
            lines.append(
                f"- {row['id']} [{row['status']}] {row['title']}"
                + (f" (provider={row['provider_hint']})" if row["provider_hint"] else "")
            )
    else:
        lines.append("- none")

    path = paths.docs_dir / "master_plan.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
