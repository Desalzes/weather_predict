---
name: handoff
description: Produce a clean session handoff so the next session can orient properly. ALWAYS use this before ending a work session, before /compact, before /clear, when the user says "wrap up", "save our progress", "end of session", "that's enough for today", "write a handoff", "checkpoint", or "stopping here". Also use proactively after completing a significant piece of work when a natural break point appears. Updates docs/sessions/_state.md, reconciles docs/sessions/_backlog.md, appends new gotchas to docs/sessions/_learnings.md, writes a new handoff doc in docs/sessions/handoffs/, and optionally records any ADR-worthy decisions via the adr skill. This is the exit discipline that makes orient's entry discipline actually work — without it, every future orient reads stale state and the whole LLM-managed-project system degrades.
---

# Handoff

Exit discipline. Paired with `orient` (entry discipline). If
this session doesn't produce a clean handoff, the next
session's `orient` reads stale state and every future session
drifts further from reality.

## When to run

- User signals end of session ("that's enough", "wrap up",
  "stopping")
- Before `/compact` or `/clear` — context compression will
  lose detail the next session needs
- Natural break point after completing a backlog item
- Checkpoint before a risky operation

If you're asking "should I run handoff now?" the answer is yes.

## Steps

### 1. Update `docs/sessions/_state.md`

Replace contents (don't append; this file is a snapshot, not a
log):

- `Updated` — timestamp
- `Branch` — current branch
- `Last commit` — hash + subject line
- `Current focus` — one sentence
- `In flight` — the ONE specific thing currently being worked
  on (or `none` if truly idle)
- `Blocked` — anything waiting on external input
- `Recently completed` — last 3–5 completions with handoff
  references
- `Active ADRs in scope` — ADRs that touch current work

Trust the git state. If you started the session thinking X was
done but `git status` shows an unstaged fix, document that.

### 2. Reconcile `docs/sessions/_backlog.md`

- Items completed this session: change `[ ]` to `[x]` and add
  the date. Do not delete — the backlog is the project's
  memory.
- New items discovered: add to the appropriate priority
  section.
- Items that grew or shrank in scope: edit in place, note the
  change.
- Re-prioritize only if priorities genuinely shifted; don't
  churn the file for its own sake.

### 3. Append to `docs/sessions/_learnings.md`

One bullet per learning, newest at the bottom, each prefixed
with `YYYY-MM-DD`. Learnings worth capturing:

- Failed approaches (so nobody tries them again)
- Surprising library/API behavior
- Non-obvious constraints from the domain
- Performance gotchas
- "Why it's done this weird way" context

Do NOT capture: things already in an ADR, things that are just
"Claude learned how X works" (that belongs in training, not in
project docs), or restatements of the task.

### 4. Write the new handoff

Path: `docs/sessions/handoffs/YYYY-MM-DD-HHMM-<slug>.md`.
Slug is kebab-case, descriptive of the session's theme.

Use `docs/sessions/handoffs/_TEMPLATE.md` as the starting
point. Required sections:

- **Context** — what we were doing and why (2–3 sentences)
- **What changed** — files touched, commits made (with hashes)
- **Current state** — working / not working / uncommitted
- **Next action** — one specific, immediately executable step
- **Watch out for** — anything surprising
- **ADRs touched** — any ADRs created or affected

"Next action" is the most important section. Vague next actions
("continue the refactor") waste the next session's first ten
minutes. A good next action names a file, a function, and a
specific change: "Edit `src/auth/session.rs:handle_expiry` to
use the new `ExpiryPolicy` enum from ADR-0007."

### 5. Record ADR-worthy decisions

If this session made any decision that future sessions would
ask "why did we do it this way?" about, invoke the `adr` skill
to record it. Rule of thumb: if reversing the decision would
take more than an hour, it's ADR-worthy.

### 6. Commit the session docs

```
git add docs/sessions/ docs/adr/
git commit -m "session: handoff <slug>"
```

Separate from code commits. The session docs committed
independently make `git log docs/sessions/` a useful trail.

## What a good handoff looks like

- Next session can `orient` in two minutes and know exactly
  what to do
- Nothing ambiguous about what was completed vs attempted
- Dead-ends documented so they aren't retried
- Next action is a single concrete step

## What a bad handoff looks like

- "We worked on the auth module" (no specifics)
- "Everything is looking good" (no state delta)
- "Continue where we left off" (no next action)
- Any mention of "I" or "we" without referring to specific
  artifacts

The handoff is written for a Claude with no memory of this
session. Anything implicit is lost.
