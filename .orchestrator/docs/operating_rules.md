# Operating Rules

- Use `AGENTS.md` as the shared rule source of truth.
- Keep orchestration writes inside `.orchestrator/`, `.claude/`, `.codex/`, or
  `.agents/`.
- Default to read-only provider calls unless the user explicitly asks for code
  execution.
- Create bounded task artifacts instead of dumping long free-form plans.
- Use adjacent git worktrees for implementation tasks when isolation is needed.
- Treat `codex_loop/` and `codex_task/` as legacy experiments, not active
  orchestration state.
