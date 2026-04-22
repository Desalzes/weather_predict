@AGENTS.md

Session management lives in docs/sessions/. Run orient at session start before any mutation.

# Claude Notes

Use this file only for Claude-specific deltas. Shared repository rules belong in
`AGENTS.md`.

## Claude-Specific Deltas

- The project-local orchestration wrapper lives in `.orchestrator/` and should
  be treated as the active management surface for this repo.
- `codex_loop/` and `codex_task/` are legacy experiments. Do not update them
  unless the user explicitly asks.
- Prefer concise session memory. Stable repo facts belong here or in
  `.claude/rules/`; long playbooks should become skills or `.orchestrator/`
  templates instead.

## Documentation Map

Subdirectory docs (agent-oriented, concise):

- `src/CLAUDE.md` — module map, data flow, key constants, change rules
- `tests/CLAUDE.md` — test file map, conventions, coverage gaps
- `data/CLAUDE.md` — directory layout, CSV/JSON schemas, safety rules
- `strategy/CLAUDE.md` — policy structure, thresholds, evolution rules

Topic-specific rules in `.claude/rules/`:

- `calibration.md` — routing order, selective fallback targets, retraining
- `paper-trading.md` — fee model, settlement flow, route attribution, blockers
- `secrets.md` — files to never commit, env vars, config inheritance
