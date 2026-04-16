from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tomllib


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    orchestrator_dir: Path
    docs_dir: Path
    tasks_dir: Path
    worklogs_dir: Path
    decisions_dir: Path
    proposals_dir: Path
    runtime_dir: Path
    templates_dir: Path
    db_path: Path
    config_path: Path


def discover_paths() -> ProjectPaths:
    orchestrator_dir = Path(__file__).resolve().parents[2]
    repo_root = orchestrator_dir.parent
    return ProjectPaths(
        repo_root=repo_root,
        orchestrator_dir=orchestrator_dir,
        docs_dir=orchestrator_dir / "docs",
        tasks_dir=orchestrator_dir / "docs" / "tasks",
        worklogs_dir=orchestrator_dir / "docs" / "worklogs",
        decisions_dir=orchestrator_dir / "docs" / "decisions",
        proposals_dir=orchestrator_dir / "proposals",
        runtime_dir=orchestrator_dir / "runtime",
        templates_dir=orchestrator_dir / "templates",
        db_path=orchestrator_dir / "state.sqlite",
        config_path=orchestrator_dir / "config.toml",
    )


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


DEFAULT_CONFIG = {
    "project": {
        "name": "project",
        "primary_provider": "codex",
        "secondary_provider": "claude",
        "default_base_branch": "main",
    },
    "routing": {
        "default_mode": "auto",
        "prefer_secondary_for_review": True,
    },
    "memory": {
        "enabled": True,
        "max_prompt_memories": 8,
        "max_directives": 6,
    },
    "worktrees": {
        "enabled": True,
        "root_mode": "adjacent",
        "prefix": "__wt__",
    },
    "providers": {
        "claude": {
            "enabled": True,
            "command": "claude",
            "model": "",
        },
        "codex": {
            "enabled": True,
            "command": "codex.cmd",
            "model": "",
        },
    },
}


def load_config(paths: ProjectPaths) -> dict:
    data = DEFAULT_CONFIG
    if paths.config_path.exists():
        with paths.config_path.open("rb") as handle:
            loaded = tomllib.load(handle)
        data = _deep_merge(DEFAULT_CONFIG, loaded)
    return data


def resolve_command(preferred: str, candidates: list[str] | None = None) -> str | None:
    search = [preferred]
    if candidates:
        search.extend(candidates)
    seen: set[str] = set()
    for candidate in search:
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None
