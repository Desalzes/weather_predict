from __future__ import annotations

import sqlite3

from .config import ProjectPaths


def classify_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    if any(token in lowered for token in ("review", "audit", "regression", "code review")):
        return "review"
    if any(token in lowered for token in ("plan", "roadmap", "spec", "brainstorm", "design")):
        return "plan"
    if any(token in lowered for token in ("implement", "fix", "refactor", "build", "add", "create")):
        return "execute"
    return "direct"


def choose_provider(route: str, config: dict, availability: dict[str, object]) -> str:
    project_cfg = config["project"]
    primary = project_cfg["primary_provider"]
    secondary = project_cfg["secondary_provider"]
    prefer_secondary_for_review = bool(config["routing"]["prefer_secondary_for_review"])

    ordered = [primary, secondary]
    if route == "review" and prefer_secondary_for_review:
        ordered = [secondary, primary]

    for name in ordered:
        provider = availability.get(name)
        if provider and getattr(provider, "available", False):
            return name
    return "none"


def fetch_open_tasks(conn: sqlite3.Connection, limit: int = 5) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, title, status, provider_hint, worktree_id
        FROM tasks
        WHERE status NOT IN ('done', 'abandoned')
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def build_provider_prompt(
    paths: ProjectPaths,
    route: str,
    user_prompt: str,
    tasks: list[sqlite3.Row],
    allow_edits: bool,
) -> str:
    task_lines = []
    for row in tasks:
        task_lines.append(
            f"- {row['id']} [{row['status']}] {row['title']}"
            + (f" (worktree={row['worktree_id']})" if row["worktree_id"] else "")
        )
    current_tasks = "\n".join(task_lines) if task_lines else "- none"

    write_policy = (
        "You may edit files if needed for this task."
        if allow_edits
        else "Do not edit files. Provide orchestration guidance only."
    )

    response_contract = {
        "direct": (
            "- answer concisely for this repository\n"
            "- reference the relevant code or docs when useful"
        ),
        "plan": (
            "- this is a one-shot planning task, not a conversation opener\n"
            "- do not acknowledge, do not ask for another objective, and do not say 'send the first task'\n"
            "- produce a concrete plan now\n"
            "- use these top-level headers exactly: Objective, Constraints, Recommended Structure, "
            "Phased Plan, Immediate Next Task\n"
            "- each section must contain substantive content\n"
            "- if repository restructuring is beneficial, say so explicitly\n"
            "- preserve secrets, local config, archives, calibration artifacts, and paper-trading data"
        ),
        "execute": (
            "- explain the implementation approach and the bounded task that should be executed next\n"
            "- call out whether a worktree is warranted"
        ),
        "review": (
            "- findings first, summary second\n"
            "- focus on bugs, regressions, missing tests, and rule drift"
        ),
    }[route]

    route_notes = {
        "direct": "This is a direct management or inspection request.",
        "plan": (
            "This is planning mode. The repository structure is flexible and may be reworked. "
            "Do not preserve the current folder layout for its own sake."
        ),
        "execute": "This is execution planning mode for a concrete repo change.",
        "review": "This is review mode.",
    }[route]

    return f"""You are the project manager session for the weather repository.

This is a non-interactive one-shot run. Complete the requested output in this response.

Read these files before answering:
- AGENTS.md
- .orchestrator/docs/project_brief.md
- .orchestrator/docs/operating_rules.md

Routing mode: {route}
Write policy: {write_policy}
Route note: {route_notes}

Current open tasks:
{current_tasks}

User request:
{user_prompt}

Response contract:
{response_contract}
"""
