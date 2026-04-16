from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import sqlite3
import subprocess

from .config import ProjectPaths
from .db import utc_now
from .tasks import fetch_task


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _resolve_base_branch(paths: ProjectPaths, preferred: str) -> str:
    check = _git(paths.repo_root, "rev-parse", "--verify", preferred)
    if check.returncode == 0:
        return preferred
    current = _git(paths.repo_root, "branch", "--show-current")
    branch = current.stdout.strip()
    return branch or preferred


def create_worktree(
    conn: sqlite3.Connection,
    paths: ProjectPaths,
    config: dict,
    task_id: str,
    base_branch: str | None = None,
) -> sqlite3.Row:
    task = fetch_task(conn, task_id)
    if task is None:
        raise ValueError(f"Unknown task: {task_id}")
    if task["worktree_id"]:
        row = conn.execute("SELECT * FROM worktrees WHERE id = ?", (task["worktree_id"],)).fetchone()
        if row is not None:
            return row

    base = _resolve_base_branch(paths, base_branch or config["project"]["default_base_branch"])
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in task["title"]).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)[:32] or task_id
    prefix = config["worktrees"]["prefix"]
    worktree_dir = paths.repo_root.parent / f"{paths.repo_root.name}{prefix}{task_id}-{slug}"
    branch_name = f"orchestrator/{task_id}-{slug}"

    result = _git(paths.repo_root, "worktree", "add", "-b", branch_name, str(worktree_dir), base)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git worktree add failed")

    worktree_id = sha1(str(worktree_dir).encode("utf-8")).hexdigest()[:12]
    now = utc_now()
    conn.execute(
        """
        INSERT INTO worktrees (id, task_id, branch_name, path, base_branch, status, created_at, removed_at)
        VALUES (?, ?, ?, ?, ?, 'active', ?, NULL)
        """,
        (worktree_id, task_id, branch_name, str(worktree_dir), base, now),
    )
    conn.execute(
        "UPDATE tasks SET worktree_id = ?, updated_at = ? WHERE id = ?",
        (worktree_id, now, task_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM worktrees WHERE id = ?", (worktree_id,)).fetchone()
    if row is None:
        raise RuntimeError("Worktree record was not created")
    return row
