"""Configuration helpers for Weather Signals."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config.example.json"

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def resolve_config_path(config_path: str | os.PathLike | None = None) -> Path:
    """Resolve the config file used by app entrypoints."""
    if config_path:
        return Path(config_path).expanduser().resolve()
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return EXAMPLE_CONFIG_PATH


def _load_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_app_config(config_path: str | os.PathLike | None = None) -> dict:
    """Load JSON config with committed example defaults for missing keys."""
    resolved = resolve_config_path(config_path)
    config = _load_json_object(resolved)

    if resolved == EXAMPLE_CONFIG_PATH or not EXAMPLE_CONFIG_PATH.exists():
        return config

    defaults = _load_json_object(EXAMPLE_CONFIG_PATH)
    merged = defaults.copy()
    merged.update(config)
    return merged


def _config_value(name: str, default: str = "") -> str:
    try:
        return str(load_app_config().get(name, default) or default).strip()
    except FileNotFoundError:
        return default


def get_kalshi_api_key_id() -> str:
    return os.getenv("KALSHI_API_KEY_ID", "").strip() or _config_value("kalshi_api_key_id")


def get_kalshi_private_key_path() -> str:
    env_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if env_path:
        return str(Path(env_path).expanduser())

    config_path = _config_value("kalshi_private_key_path", "api-credentials.txt")
    candidate = Path(config_path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate)


KALSHI_API_KEY_ID = get_kalshi_api_key_id()
KALSHI_PRIVATE_KEY_PATH = get_kalshi_private_key_path()
