from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

from .config import ProjectPaths


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  provider_session_id TEXT,
  role TEXT NOT NULL,
  task_id TEXT,
  worktree_id TEXT,
  status TEXT NOT NULL,
  cwd TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  last_activity_at TEXT,
  summary TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  status TEXT NOT NULL,
  priority TEXT NOT NULL,
  role TEXT NOT NULL,
  provider_hint TEXT,
  worktree_required INTEGER NOT NULL DEFAULT 0,
  active_session_id TEXT,
  worktree_id TEXT,
  parent_task_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS worktrees (
  id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  branch_name TEXT NOT NULL,
  path TEXT NOT NULL,
  base_branch TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  removed_at TEXT
);

CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  scope_ref TEXT,
  memory_type TEXT NOT NULL,
  content TEXT NOT NULL,
  summary TEXT,
  importance REAL NOT NULL,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL,
  last_accessed_at TEXT
);

CREATE TABLE IF NOT EXISTS directives (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  content TEXT NOT NULL,
  priority TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
  id TEXT PRIMARY KEY,
  proposal_type TEXT NOT NULL,
  target_path TEXT NOT NULL,
  title TEXT NOT NULL,
  rationale TEXT NOT NULL,
  content TEXT NOT NULL,
  status TEXT NOT NULL,
  source_session_id TEXT,
  created_at TEXT NOT NULL,
  reviewed_at TEXT
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,
  session_id TEXT,
  task_id TEXT,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(paths: ProjectPaths) -> sqlite3.Connection:
    conn = sqlite3.connect(paths.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
