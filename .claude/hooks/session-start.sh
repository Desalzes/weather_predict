#!/usr/bin/env bash
# Injected into context at the start of every session.
# Not a suggestion. Paired with pre-tool-use.sh which enforces it.
set -euo pipefail

cat <<'EOF'
================================================================
BINDING SESSION ORIENTATION (not a suggestion)
================================================================
This project is LLM-managed. Before you use Edit, Write,
MultiEdit, NotebookEdit, or Bash, you MUST run the `orient`
skill. The PreToolUse hook will block mutation tools until a
session marker exists at .claude/markers/oriented-<session_id>.

Skills on this project:
  orient   REQUIRED first action. Loads ADRs (codebase-memory-
           mcp), docs/sessions/_state.md, open _backlog.md
           items, _learnings.md, latest handoff, git status.
           Writes the unblocking marker.
  handoff  REQUIRED before stopping, /compact, or /clear.
           Updates _state.md, reconciles _backlog.md, appends
           _learnings.md, writes docs/sessions/handoffs/<doc>.
  adr      Record an architectural decision via
           codebase-memory-mcp + docs/adr/ mirror.

Read tools are always allowed so orient can run unblocked.
For trivial Q&A that truly will not touch code, the user may
pass --skip <reason> to orient; this is logged in the marker.

Start with `orient` now. Do not pattern-match to "I know
what to do from the last conversation" — session memory is
not persistent; the docs are the source of truth.
================================================================
EOF
