# Weather Orchestrator Architecture

The wrapper is intentionally local and file-first.

## Control Surface

- `manage.py`: command entrypoint
- `state.sqlite`: runtime state
- `docs/tasks/`: inspectable task files
- `docs/worklogs/`: inspectable execution logs

## Design

- shared repo rules come from `AGENTS.md`
- Claude-specific deltas live in `CLAUDE.md`
- provider adapters stay thin
- task and worktree state are stored in SQLite and mirrored to markdown when
  helpful

## Current Scope

- repo contract validation
- provider detection
- prompt routing
- task creation
- worktree creation
- directive capture

Provider-native hooks and self-modifying skills remain proposal-driven.
