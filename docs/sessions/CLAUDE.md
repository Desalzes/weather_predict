# Session Management

This directory is the session layer of LLM-managed project
maintenance. Every session reads this first (via `orient`) and
updates it before ending (via `handoff`).

## Files

- `CLAUDE.md` — this file; explains the system
- `_state.md` — live snapshot of the project (current focus,
  in-flight work, blockers). Rewritten by `handoff` each
  session. Treated as single source of truth for "what's
  happening right now".
- `_backlog.md` — open items in priority order. Items are
  never deleted; completed ones get `[x]` and a date.
- `_learnings.md` — append-only log of gotchas, failed
  approaches, surprising constraints. Appended by `handoff`.
- `handoffs/` — one file per session, named
  `YYYY-MM-DD-HHMM-<slug>.md`. Immutable once written.

## Hierarchy of knowledge

`orient` reads the four tiers in this order. Each tier can
override the previous one when they disagree:

1. **Project tier: ADRs** (codebase-memory-mcp) — *why* the
   architecture is the way it is
2. **Session hub: this directory** — *what's open, what we've
   learned*
3. **In-flight: `_state.md` + latest handoff** — *where we are
   right now*
4. **Branch: `git status`** — *uncommitted reality; beats
   docs when they disagree*

## Rules

- `_state.md` is rewritten each `handoff`, not appended to. It
  is a snapshot.
- `_backlog.md` items are never deleted. Completed items get
  `[x] YYYY-MM-DD`. This is the project's memory.
- `_learnings.md` is append-only. If a learning is obsoleted,
  add a new entry noting the correction rather than editing
  the old one.
- Handoffs are immutable once committed. Corrections go in the
  next handoff.
- Session docs are committed separately from code with
  `session: <slug>` subject lines so `git log docs/sessions/`
  is a clean session trail.
