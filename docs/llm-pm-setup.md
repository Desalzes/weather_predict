# LLM-Managed Project Scaffolding

A drop-in structure that forces Claude (Opus via Claude Code)
to orient at the start of every session, write clean handoffs at
the end, and route architectural decisions through ADRs stored
in codebase-memory-mcp.

## Why

LLM sessions have no memory. Without a forcing function, every
session pattern-matches from training or a shallow skim of
available context, misses prior decisions, and drifts from the
project's actual state. Soft instructions (even SessionStart
hooks) fail the same way: Claude treats them as "context" and
carries on.

This scaffolding uses three reinforcing pieces:

1. **SessionStart hook** — injects a binding orientation
   instruction at the top of every session.
2. **PreToolUse hook** — blocks `Edit`/`Write`/`MultiEdit`/
   `NotebookEdit`/`Bash` until a session marker file exists.
   The marker is produced by the `orient` skill after it loads
   the four tiers of project knowledge. You cannot route
   around it.
3. **Three skills (`orient`, `handoff`, `adr`)** that encode
   the discipline: entry, exit, and decisions.

Entry discipline alone isn't enough — `orient` only works if
the *previous* session wrote a clean state. `handoff` is the
closing brace.

## File tree

```
.claude/
├── settings.json                    Hook wiring
├── hooks/
│   ├── session-start.sh             Injects binding instruction
│   └── pre-tool-use.sh              Enforces marker before mutations
├── skills/
│   ├── orient/SKILL.md              Entry discipline
│   ├── handoff/SKILL.md             Exit discipline
│   └── adr/SKILL.md                 Decision records
└── markers/                         (gitignored, per-session)
docs/
├── sessions/
│   ├── CLAUDE.md                    Session hub explainer
│   ├── _state.md                    Live snapshot (rewritten each handoff)
│   ├── _backlog.md                  Open items (append-only bar completion)
│   ├── _learnings.md                Gotchas (append-only)
│   └── handoffs/
│       └── _TEMPLATE.md             Handoff doc template
└── adr/
    └── README.md                    ADR mirror + index
.gitignore                           Excludes .claude/markers/
```

## Install

Drop the contents of this bundle into the root of your project.
Then:

```bash
chmod +x .claude/hooks/session-start.sh .claude/hooks/pre-tool-use.sh
mkdir -p .claude/markers
```

Verify `jq` is installed (the PreToolUse hook uses it; if
missing, the hook fails open with a warning):

```bash
command -v jq || echo "install jq"
```

Merge `.claude/settings.json` with any existing settings rather
than overwriting.

Add a first ADR to anchor the project. The session hub, backlog,
and learnings files start empty (templates) — fill them in your
first session.

## Configure for your codebase-memory-mcp

The `orient` and `adr` skills reference codebase-memory-mcp
generically because the exact tool names depend on your MCP
server configuration. Open `.claude/skills/orient/SKILL.md` and
`.claude/skills/adr/SKILL.md` and replace the generic
`list_memories` / `create_memory` placeholders with the actual
tool names your server exposes. Five minutes once, then every
skill invocation routes correctly.

If you want me to tailor the skills to specific MCP tool names,
paste the output of `/mcp` for the codebase-memory server and
I'll update them.

## Day-one usage

1. Start a Claude Code session in the project root.
2. SessionStart hook fires, injecting the binding instruction.
3. Claude invokes `orient`. Since the state files are empty
   templates, the marker is still written — orient just
   produces a thinner brief.
4. Work happens.
5. Before stopping, Claude invokes `handoff`. It populates
   `_state.md`, adds entries to `_backlog.md`, writes the first
   handoff in `docs/sessions/handoffs/`.
6. Next session starts. `orient` now has something real to
   read.

## Escape hatches

- **Quick Q&A that won't touch code**: user passes
  `--skip <reason>` to `orient`. Marker is written with the
  reason, session proceeds. Logged for audit.
- **Subagents**: PreToolUse hook detects `subagent_type` /
  `is_subagent` in the tool input and passes through. The
  parent session has already oriented.
- **Broken hook**: if `jq` is missing or the hook errors, it
  fails open with stderr warnings. Better a degraded session
  than a bricked one.

## Failure modes this prevents

- Claude pattern-matching from training and missing ADR-0007's
  rationale
- A fresh session editing a file based on stale assumptions
  when `_state.md` says it's mid-refactor
- Decisions made in chat and lost because they never became
  ADRs
- Three sessions in a row each re-discovering the same gotcha
  because nothing wrote to `_learnings.md`
- `/compact` destroying the only record of what the session
  accomplished

## What this does *not* solve

- Code review quality — orient just loads context; it doesn't
  review your diffs.
- Backlog grooming — you still need to decide priorities.
- ADR quality — the skill gives a template, not judgment.

Those stay human-in-the-loop.
