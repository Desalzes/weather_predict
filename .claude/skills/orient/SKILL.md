---
name: orient
description: Load all essential project context before doing any work. ALWAYS use this as the very first action of every session on this project. Triggers include the start of any new session, any user request to "begin", "start", "pick up", "continue", "resume", "what were we doing", or any tool call blocked by the PreToolUse hook with a "session not oriented" message. This skill reads active ADRs from codebase-memory-mcp, the session hub (docs/sessions/CLAUDE.md, _state.md, _backlog.md, _learnings.md), the most recent handoff, and git status, then writes .claude/markers/oriented-<session_id> which unblocks mutation tools. Do not skip this — mutation tools WILL be blocked until this skill completes, and patterning-matching from training or apparent context is NOT a substitute for reading the actual docs. If the user invokes this with --skip <reason>, still produce the marker with the reason recorded.
---

# Orient

Entry discipline. First action of every session on this project.

The PreToolUse hook blocks `Edit`, `Write`, `MultiEdit`,
`NotebookEdit`, and `Bash` until `.claude/markers/oriented-<session_id>`
exists. This skill produces that marker — but only after loading
the four tiers of project knowledge honestly. If you shortcut the
reads and stamp the marker, you defeat the whole system.

## Why this exists

LLM sessions have no memory. Without a forcing function, Claude
will pattern-match from training, skim surface context, and miss
decisions captured in ADRs or constraints captured in
`_learnings.md`. The result is wasted tokens, rework, and
contradictions with prior decisions. This skill + hook pair is
the forcing function.

## The four tiers (load in order)

### Tier 1 — Project: ADRs via codebase-memory-mcp

ADRs are the "why" behind the architecture. Every ADR in
`accepted` status is in force and must be honored.

1. Call `mcp__codebase-memory-mcp__manage_adr` with
   `mode: "get"` and `project: "Weather"` (or whatever
   `basename $(git rev-parse --show-toplevel)` returns for this
   repo) to retrieve the ADR document. For finer-grained access,
   use `mode: "sections"` with a `sections` list of ADR numbers or
   headings. For status-filtered queries, fall back to
   `mcp__codebase-memory-mcp__query_graph` with a Cypher query
   that filters on `status`.
2. Read every ADR with status `accepted` in full. ADRs marked
   `superseded` or `deprecated` are history — note them but
   don't treat them as active constraints.
3. If codebase-memory-mcp is unreachable (use
   `mcp__codebase-memory-mcp__list_projects` to probe), fall back
   to reading `docs/adr/*.md` in filename order. Note in your
   summary that you used the filesystem fallback.

### Tier 2 — Session hub

Read in this exact order:

1. `docs/sessions/CLAUDE.md` — the hub explainer. Short; read
   it even if you think you know it.
2. `docs/sessions/_state.md` — live state. Trust this over
   handoffs if they disagree.
3. `docs/sessions/_backlog.md` — open items. Pay attention to
   P0 and P1.
4. `docs/sessions/_learnings.md` — append-only gotchas log.
   Skim all of it; you don't know which learning is relevant
   until you know the task.

### Tier 3 — In-flight

Find the most recent file in `docs/sessions/handoffs/` (sorted
by filename; convention is `YYYY-MM-DD-HHMM-<slug>.md`). Read it
in full. Pay particular attention to "Next action" and "Watch
out for".

### Tier 4 — Branch

```
git branch --show-current
git status --short
git log -n 10 --oneline
git diff --stat HEAD
```

Uncommitted changes are reality. If `_state.md` says "clean" but
`git status` shows changes, `git status` wins.

## After loading

Produce a short orientation brief for the user with these
sections, in this order:

- **Where we are** — branch, latest handoff's one-line summary,
  any uncommitted delta
- **ADRs in force** — bullet list of accepted ADRs by number
  and title
- **Top open items** — up to five from the backlog, by priority
- **In flight** — the one thing from `_state.md`
- **Watchouts** — any `_learnings.md` entries that touch the
  likely next work
- **Proposed first action** — one concrete step, not a vague
  direction

Keep this brief tight. The user does not want a recitation of
every doc; they want the synthesis.

## Write the marker

```
mkdir -p .claude/markers
cat > ".claude/markers/oriented-${CLAUDE_SESSION_ID:-default}" <<MARKER
oriented: $(date -Iseconds)
branch: $(git branch --show-current 2>/dev/null || echo "unknown")
latest_handoff: <filename>
adrs_loaded: <count>
first_action: <one-line>
MARKER
```

Do this only after you have actually read the docs. Stamping
the marker without loading the context is lying to the system.

## Skip path

The user may invoke with `--skip <reason>` for trivial Q&A:

```
mkdir -p .claude/markers
echo "skipped: <reason> at $(date -Iseconds)" \
  > ".claude/markers/oriented-${CLAUDE_SESSION_ID:-default}"
```

Use only when: (a) the user explicitly passed `--skip`, and
(b) the task is conversational, not code-touching. A user
saying "just quickly" is not a skip grant.

## What counts as "done"

- All four tiers actually read (not glanced at)
- User received a tight synthesis brief with a concrete first
  action
- Marker file written with metadata, not just a timestamp
