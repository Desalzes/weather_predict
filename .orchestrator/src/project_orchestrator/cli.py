from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import textwrap
import uuid

from .artifacts import append_worklog_entry, write_master_plan
from .config import discover_paths, load_config
from .db import connect, init_db, utc_now
from .memory import extract_directive_candidates, upsert_directive
from .planning import finalize_plan_output, synthesize_plan
from .providers import detect_providers, run_provider_prompt
from .routing import build_provider_prompt, choose_provider, classify_prompt, fetch_open_tasks
from .tasks import create_task, list_tasks
from .worktrees import create_worktree


def _bootstrap() -> tuple[Path, dict, sqlite3.Connection]:
    paths = discover_paths()
    config = load_config(paths)
    conn = connect(paths)
    init_db(conn)
    return paths, config, conn


def _record_event(conn: sqlite3.Connection, event_type: str, payload: dict, session_id: str | None = None, task_id: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO events (event_type, session_id, task_id, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (event_type, session_id, task_id, json.dumps(payload, ensure_ascii=True), utc_now()),
    )
    conn.commit()


def _create_session_record(conn: sqlite3.Connection, provider: str, role: str, cwd: str, summary: str) -> str:
    session_id = str(uuid.uuid4())
    now = utc_now()
    conn.execute(
        """
        INSERT INTO sessions (
          id, provider, provider_session_id, role, task_id, worktree_id, status, cwd,
          started_at, ended_at, last_activity_at, summary
        )
        VALUES (?, ?, NULL, ?, NULL, NULL, 'complete', ?, ?, ?, ?, ?)
        """,
        (session_id, provider, role, cwd, now, now, now, summary),
    )
    conn.commit()
    return session_id


def cmd_status(_: argparse.Namespace) -> int:
    paths, config, conn = _bootstrap()
    availability = detect_providers(config)
    open_tasks = conn.execute(
        "SELECT COUNT(*) AS count FROM tasks WHERE status NOT IN ('done', 'abandoned')"
    ).fetchone()["count"]
    directives = conn.execute("SELECT COUNT(*) AS count FROM directives WHERE active = 1").fetchone()["count"]
    sessions = conn.execute("SELECT COUNT(*) AS count FROM sessions").fetchone()["count"]
    print(f"repo_root: {paths.repo_root}")
    print(f"orchestrator_dir: {paths.orchestrator_dir}")
    print(f"db_path: {paths.db_path}")
    print(f"project_name: {config['project']['name']}")
    print(f"open_tasks: {open_tasks}")
    print(f"directives: {directives}")
    print(f"sessions: {sessions}")
    for name, provider in availability.items():
        print(f"provider.{name}: {'available' if provider.available else 'unavailable'} ({provider.reason})")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    paths, config, conn = _bootstrap()
    availability = detect_providers(config)
    checks = [
        ("git repo", (paths.repo_root / ".git").exists(), str(paths.repo_root / ".git")),
        ("AGENTS.md", (paths.repo_root / "AGENTS.md").exists(), "shared project instructions"),
        ("CLAUDE.md", (paths.repo_root / "CLAUDE.md").exists(), "Claude project memory"),
        ("orchestrator config", paths.config_path.exists(), str(paths.config_path)),
        ("state db", paths.db_path.exists(), str(paths.db_path)),
        ("venv python", (paths.repo_root / ".venv" / "Scripts" / "python.exe").exists(), "repo-local python"),
    ]
    for label, ok, detail in checks:
        print(f"[{'ok' if ok else 'warn'}] {label}: {detail}")
    for name, provider in availability.items():
        print(f"[{'ok' if provider.available else 'warn'}] provider {name}: {provider.command or provider.reason}")
    recent = conn.execute(
        "SELECT COUNT(*) AS count FROM tasks WHERE status NOT IN ('done', 'abandoned')"
    ).fetchone()["count"]
    print(f"[info] open task count: {recent}")
    return 0


def _task_title_from_prompt(prompt: str) -> str:
    collapsed = " ".join(prompt.strip().split())
    if len(collapsed) <= 72:
        return collapsed
    return collapsed[:69].rstrip() + "..."


def _default_summary(route: str, provider: str, dry_run: bool) -> str:
    if dry_run:
        return (
            "Routing preview only. No provider call was made. Re-run without "
            "`--dry-run` to generate a full plan or execution note."
        )
    if provider == "none":
        return "No provider was available, so only wrapper-level artifacts were created."
    return f"{route.title()} request was routed to {provider}."


def _run_ask(
    prompt: str,
    provider_override: str,
    allow_edits: bool,
    dry_run: bool,
    no_task: bool,
    show_provider_stderr: bool,
) -> int:
    paths, config, conn = _bootstrap()
    availability = detect_providers(config)
    route = classify_prompt(prompt)
    provider = provider_override if provider_override != "auto" else choose_provider(route, config, availability)

    created_task_id: str | None = None
    if route in {"plan", "execute", "review"} and not no_task:
        worktree_required = route == "execute"
        task_id, doc_path = create_task(
            conn=conn,
            paths=paths,
            title=_task_title_from_prompt(prompt),
            objective=prompt,
            route=route,
            provider_hint=provider,
            worktree_required=worktree_required,
        )
        created_task_id = task_id
        print(f"created_task: {task_id}")
        print(f"task_doc: {doc_path}")

    for directive in extract_directive_candidates(prompt):
        directive_id = upsert_directive(conn, directive)
        print(f"captured_directive: {directive_id}")

    _record_event(conn, "ask_routed", {"route": route, "provider": provider, "prompt": prompt}, task_id=created_task_id)

    if dry_run or provider == "none":
        if created_task_id:
            summary = _default_summary(route, provider, dry_run)
            if route == "plan":
                summary = synthesize_plan(prompt, fetch_open_tasks(conn, limit=20))
            worklog = append_worklog_entry(
                paths=paths,
                task_id=created_task_id,
                route=route,
                provider=provider,
                prompt=prompt,
                summary=summary,
            )
            print(f"worklog: {worklog}")
            if route == "plan":
                plan_path = write_master_plan(
                    paths=paths,
                    prompt=prompt,
                    provider=provider,
                    summary=summary,
                    tasks=fetch_open_tasks(conn, limit=20),
                )
                conn.execute(
                    "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                    ("planned", utc_now(), created_task_id),
                )
                conn.commit()
                print(f"master_plan: {plan_path}")
        print(f"route: {route}")
        print(f"provider: {provider}")
        if provider == "none":
            print("provider_reason: no configured provider command was available")
        return 0

    selected = availability[provider]
    open_tasks = fetch_open_tasks(conn)
    provider_prompt = build_provider_prompt(paths, route, prompt, open_tasks, allow_edits)
    model = config["providers"][provider].get("model", "")
    result = run_provider_prompt(
        provider=provider,
        command=selected.command or provider,
        prompt=provider_prompt,
        cwd=str(paths.repo_root),
        model=model,
        allow_edits=allow_edits,
    )
    final_output = result.stdout
    if route == "plan":
        final_output = finalize_plan_output(prompt, result.stdout, fetch_open_tasks(conn, limit=20))
    session_id = _create_session_record(conn, provider=provider, role="manager", cwd=str(paths.repo_root), summary=_task_title_from_prompt(prompt))
    _record_event(
        conn,
        "provider_result",
        {
            "provider": provider,
            "returncode": result.returncode,
            "stdout": final_output[:2000],
            "stderr": result.stderr[:2000],
            "command": result.command,
        },
        session_id=session_id,
        task_id=created_task_id,
    )
    if created_task_id:
        worklog = append_worklog_entry(
            paths=paths,
            task_id=created_task_id,
            route=route,
            provider=provider,
            prompt=prompt,
            summary=final_output or _default_summary(route, provider, dry_run=False),
        )
        print(f"worklog: {worklog}")
        if route == "plan":
            plan_path = write_master_plan(
                paths=paths,
                prompt=prompt,
                provider=provider,
                summary=final_output or _default_summary(route, provider, dry_run=False),
                tasks=fetch_open_tasks(conn, limit=20),
            )
            conn.execute(
                "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                ("planned", utc_now(), created_task_id),
                )
            conn.commit()
            print(f"master_plan: {plan_path}")
    print(f"route: {route}")
    print(f"provider: {provider}")
    print(f"returncode: {result.returncode}")
    if final_output:
        print("\nstdout:\n")
        print(final_output)
    if result.stderr and (show_provider_stderr or result.returncode != 0):
        print("\nstderr:\n")
        print(result.stderr)
    return result.returncode


def cmd_ask(args: argparse.Namespace) -> int:
    return _run_ask(
        prompt=args.prompt,
        provider_override=args.provider,
        allow_edits=args.allow_edits,
        dry_run=args.dry_run,
        no_task=args.no_task,
        show_provider_stderr=args.show_provider_stderr,
    )


def cmd_chat(args: argparse.Namespace) -> int:
    print("weather-orchestrator chat")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            return 0
        rc = _run_ask(
            prompt=prompt,
            provider_override=args.provider,
            allow_edits=args.allow_edits,
            dry_run=args.dry_run,
            no_task=args.no_task,
            show_provider_stderr=args.show_provider_stderr,
        )
        if rc != 0:
            print(f"command failed with exit code {rc}")


def cmd_tasks_list(_: argparse.Namespace) -> int:
    _, _, conn = _bootstrap()
    rows = list_tasks(conn)
    if not rows:
        print("no tasks")
        return 0
    for row in rows:
        worktree = row["worktree_id"] or "-"
        print(f"{row['id']}  {row['status']:<8}  {row['provider_hint'] or '-':<6}  {worktree:<12}  {row['title']}")
    return 0


def cmd_tasks_create(args: argparse.Namespace) -> int:
    paths, _, conn = _bootstrap()
    task_id, doc_path = create_task(
        conn=conn,
        paths=paths,
        title=args.title,
        objective=args.objective or args.title,
        route=args.route,
        provider_hint=args.provider,
        worktree_required=args.worktree_required,
    )
    print(f"created_task: {task_id}")
    print(f"task_doc: {doc_path}")
    return 0


def cmd_worktree_create(args: argparse.Namespace) -> int:
    paths, config, conn = _bootstrap()
    row = create_worktree(conn, paths, config, args.task_id, base_branch=args.base_branch)
    print(f"worktree_id: {row['id']}")
    print(f"path: {row['path']}")
    print(f"branch: {row['branch_name']}")
    print(f"base_branch: {row['base_branch']}")
    return 0


def cmd_proposals_list(_: argparse.Namespace) -> int:
    _, _, conn = _bootstrap()
    rows = conn.execute(
        "SELECT id, proposal_type, status, target_path, title FROM proposals ORDER BY created_at DESC"
    ).fetchall()
    if not rows:
        print("no proposals")
        return 0
    for row in rows:
        print(f"{row['id']}  {row['proposal_type']:<8}  {row['status']:<10}  {row['target_path']}  {row['title']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="Project-local orchestrator for the weather repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python .orchestrator/manage.py doctor
              python .orchestrator/manage.py status
              python .orchestrator/manage.py ask "Plan the next calibration upgrade"
              python .orchestrator/manage.py tasks create "Backfill HRRR validation"
            """
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show orchestrator status")
    status.set_defaults(func=cmd_status)

    doctor = sub.add_parser("doctor", help="Validate local wrapper setup")
    doctor.set_defaults(func=cmd_doctor)

    ask = sub.add_parser("ask", help="Route one prompt through the wrapper")
    ask.add_argument("prompt", help="Prompt to route")
    ask.add_argument("--provider", choices=["auto", "claude", "codex", "none"], default="auto")
    ask.add_argument("--allow-edits", action="store_true", help="Allow provider write mode")
    ask.add_argument("--dry-run", action="store_true", help="Classify and persist without invoking a provider")
    ask.add_argument("--no-task", action="store_true", help="Do not create a task artifact for plan/execute/review routes")
    ask.add_argument("--show-provider-stderr", action="store_true", help="Print raw provider stderr on success")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="Run a simple interactive prompt loop")
    chat.add_argument("--provider", choices=["auto", "claude", "codex", "none"], default="auto")
    chat.add_argument("--allow-edits", action="store_true", help="Allow provider write mode")
    chat.add_argument("--dry-run", action="store_true", help="Classify and persist without invoking a provider")
    chat.add_argument("--no-task", action="store_true", help="Do not create a task artifact for plan/execute/review routes")
    chat.add_argument("--show-provider-stderr", action="store_true", help="Print raw provider stderr on success")
    chat.set_defaults(func=cmd_chat)

    tasks = sub.add_parser("tasks", help="Task operations")
    tasks_sub = tasks.add_subparsers(dest="tasks_command", required=True)

    tasks_list = tasks_sub.add_parser("list", help="List tasks")
    tasks_list.set_defaults(func=cmd_tasks_list)

    tasks_create = tasks_sub.add_parser("create", help="Create a task")
    tasks_create.add_argument("title")
    tasks_create.add_argument("--objective", default="")
    tasks_create.add_argument("--route", choices=["plan", "execute", "review"], default="execute")
    tasks_create.add_argument("--provider", choices=["claude", "codex", "none"], default="codex")
    tasks_create.add_argument("--worktree-required", action="store_true")
    tasks_create.set_defaults(func=cmd_tasks_create)

    worktree = sub.add_parser("worktree", help="Worktree operations")
    worktree_sub = worktree.add_subparsers(dest="worktree_command", required=True)

    worktree_create = worktree_sub.add_parser("create", help="Create a task worktree")
    worktree_create.add_argument("task_id")
    worktree_create.add_argument("--base-branch", default="")
    worktree_create.set_defaults(func=cmd_worktree_create)

    proposals = sub.add_parser("proposals", help="Proposal operations")
    proposals_sub = proposals.add_subparsers(dest="proposals_command", required=True)
    proposals_list = proposals_sub.add_parser("list", help="List proposals")
    proposals_list.set_defaults(func=cmd_proposals_list)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
