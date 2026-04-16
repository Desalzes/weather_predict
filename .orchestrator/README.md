# Project Orchestrator

This directory contains the project-local orchestration wrapper for the weather
repo. It is designed for one repository and provides a single control surface
for Claude Code and Codex CLI sessions.

## Initial Commands

```powershell
.\.venv\Scripts\python.exe .orchestrator\manage.py doctor
.\.venv\Scripts\python.exe .orchestrator\manage.py status
.\.venv\Scripts\python.exe .orchestrator\manage.py ask "Plan the next calibration upgrade"
.\.venv\Scripts\python.exe .orchestrator\manage.py tasks list
```

## Scope

- `.orchestrator/docs/`: inspectable plans, tasks, and worklogs
- `.orchestrator/state.sqlite`: local runtime state
- `.orchestrator/runtime/`: local logs and session artifacts
- `.orchestrator/proposals/`: pending skills, hooks, and rules

This scaffold keeps provider-specific behavior thin and pushes shared project
rules into `AGENTS.md`.
