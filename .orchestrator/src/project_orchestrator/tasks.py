from __future__ import annotations

from pathlib import Path
import sqlite3

from .config import ProjectPaths
from .db import utc_now


def _next_task_id(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT id FROM tasks WHERE id LIKE 'task_%' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return "task_0001"
    number = int(row["id"].split("_", 1)[1]) + 1
    return f"task_{number:04d}"


def task_doc_path(paths: ProjectPaths, task_id: str) -> Path:
    return paths.tasks_dir / f"{task_id}.md"


def write_task_doc(
    paths: ProjectPaths,
    task_id: str,
    title: str,
    objective: str,
    route: str,
    provider_hint: str,
    worktree_required: bool,
) -> Path:
    content = f"""---
id: {task_id}
title: {title}
status: ready
priority: medium
role: {"reviewer" if route == "review" else "executor"}
provider_hint: {provider_hint}
worktree_required: {"true" if worktree_required else "false"}
depends_on: []
---

## Objective

{objective}

## Constraints

- Follow `AGENTS.md`.
- Keep scope bounded.
- Do not touch local-only secrets or bulk rewrite archived data unless explicitly requested.

## Required Reads

- `README.md`
- `UPGRADE_PLAN.md`

## Completion Criteria

- The requested outcome is clearly implemented or investigated.
- Validation steps are recorded in a worklog.
"""
    path = task_doc_path(paths, task_id)
    path.write_text(content, encoding="utf-8")
    return path


def create_task(
    conn: sqlite3.Connection,
    paths: ProjectPaths,
    title: str,
    objective: str,
    route: str,
    provider_hint: str,
    worktree_required: bool,
) -> tuple[str, Path]:
    task_id = _next_task_id(conn)
    now = utc_now()
    conn.execute(
        """
        INSERT INTO tasks (
          id, title, status, priority, role, provider_hint, worktree_required,
          active_session_id, worktree_id, parent_task_id, created_at, updated_at
        )
        VALUES (?, ?, 'ready', 'medium', ?, ?, ?, NULL, NULL, NULL, ?, ?)
        """,
        (
            task_id,
            title,
            "reviewer" if route == "review" else "executor",
            provider_hint,
            1 if worktree_required else 0,
            now,
            now,
        ),
    )
    doc_path = write_task_doc(paths, task_id, title, objective, route, provider_hint, worktree_required)
    conn.commit()
    return task_id, doc_path


def list_tasks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, title, status, provider_hint, worktree_required, worktree_id, updated_at
        FROM tasks
        ORDER BY id DESC
        """
    ).fetchall()


def fetch_task(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
