---
name: adr
description: Create, update, or supersede an Architectural Decision Record (ADR) stored via codebase-memory-mcp. Use when a non-trivial design choice is being made — library selection, data model shape, API contract, module boundary, deployment model, security posture, error-handling convention, testing strategy, or any decision where future sessions would ask "why did we do it this way?". Trigger proactively on phrases like "let's go with", "we've decided to", "I'll use X instead of Y", "the architecture will", "record this choice", or whenever a decision is made that would cost more than an hour to reverse. Also use when superseding a previous decision — write a new ADR rather than editing the old one. ADRs are load-bearing context that every future orient reads.
---

# ADR — Architectural Decision Record

## What an ADR is

A short document capturing one decision: the context, the
options considered, the choice made, and the consequences.
ADRs are immutable in spirit — to change your mind, write a
new ADR that supersedes the old one. The old rationale
remains readable.

## Storage

Primary: codebase-memory-mcp. This makes the ADR queryable
from any session regardless of working directory.

Mirror: `docs/adr/NNNN-<slug>.md`. Keeps ADRs visible in PR
review, greppable, and readable without the MCP.

Both sides must stay in sync. When writing, update both.

## When to write one

Rule of thumb: if reversing this decision three months from now
would cost more than an hour of work, write an ADR.

**ADR-worthy:**

- Database / persistence choice
- Auth model and session lifecycle
- State management library and scope
- Module boundaries and ownership
- Error handling and propagation convention
- Testing strategy (what's tested at which layer)
- Deploy target and environment model
- Secrets handling
- Observability stack
- Public API contract shape

**Not ADR-worthy** (put in code comments or a design doc):

- Variable naming
- Choice of library within a family you already committed to
- One-off refactors
- Bug fixes

## Numbering

ADRs are numbered sequentially starting at 0001. Never reuse a
number, even if an ADR is superseded.

To pick the next number:

1. Call `mcp__codebase-memory-mcp__manage_adr` with
   `mode: "get"`, `project: "Weather"` (or whatever
   `basename $(git rev-parse --show-toplevel)` returns) and scan
   the returned content for the highest `# ADR-NNNN` heading.
2. `ls docs/adr/` and take max.
3. Use the larger of the two, plus one.

## Template

```markdown
# ADR-NNNN: <title stating the decision, not the problem>

Status: proposed | accepted | superseded by ADR-MMMM | deprecated
Date: YYYY-MM-DD
Deciders: <session or person>

## Context

What forces are in play? What's the problem? Constraints,
non-negotiables, prior art. Keep it to a few paragraphs —
this is the "why now" not the company history.

## Decision

We will <the specific thing we are doing>.

Be concrete. "Use Postgres" is weak. "Use Postgres 16 as the
primary store for user and session data; use Redis only for
ephemeral rate-limit counters" is ADR-grade.

## Alternatives considered

- **<Option B>** — rejected because <reason>
- **<Option C>** — rejected because <reason>

Include the option that was almost picked. Future sessions will
ask "why not B" and the answer has to be here.

## Consequences

- Positive: ...
- Negative: ...
- Neutral: ...

Honest negatives are the most valuable section. Every decision
has them.

## Revisit if

Specific conditions under which this decision should be
reopened. Example: "revisit if we exceed 50 rps sustained" or
"revisit if the team grows past 5 engineers".
```

## Steps to write an ADR

1. **Confirm ADR-worthy.** Apply the one-hour-reversal test.
   If borderline, ask the user.
2. **Pick the number** (see Numbering).
3. **Draft** using the template. Keep it short — over one page
   suggests you're including implementation detail that belongs
   in code.
4. **Store in codebase-memory-mcp.** Call
   `mcp__codebase-memory-mcp__manage_adr` with
   `mode: "update"`, `project: "Weather"`, and pass the new
   ADR's full markdown as `content`. The server appends by
   section heading — make sure the ADR starts with
   `# ADR-NNNN: <title>` so future `mode: "get"` / `mode:
   "sections"` calls can find it. The Status line inside the
   ADR is the source of truth for `accepted` / `proposed`;
   there is no separate tag field.
5. **Mirror to filesystem:** `docs/adr/NNNN-<slug>.md`.
6. **Reference** in the current session's handoff and in any
   commit that implements the decision:
   `refs ADR-NNNN` in commit messages.

## Superseding an ADR

Decisions change. Do not edit the original ADR. Instead:

1. Write a new ADR. In its Context section, explain what
   changed and reference the prior ADR by number.
2. Update the original's Status line to
   `Superseded by ADR-MMMM`. This is the one exception to
   immutability — it's a pointer forward, not a rewrite.
3. Update the mirror file the same way.
4. In codebase-memory-mcp, call
   `mcp__codebase-memory-mcp__manage_adr` with
   `mode: "sections"`, `project: "Weather"`, and update only
   the old ADR's Status line to
   `Superseded by ADR-MMMM`. Use `sections` to target the single
   ADR section rather than rewriting the whole ADR document.

Deprecating (the decision no longer applies, but nothing
replaces it) works the same way with status `deprecated`.

## Common mistakes

- Writing the ADR after the decision has been implemented and
  forgotten. Write ADRs *as* decisions are made, not in
  retrospect.
- Burying the decision in the Context section. The Decision
  section should be boldly actionable.
- Omitting the rejected alternatives. Future sessions will
  propose them again if you don't.
- Editing an ADR instead of superseding it. The value is the
  historical record.
