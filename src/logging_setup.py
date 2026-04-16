"""Shared logging setup for Weather Signals scripts."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT, load_app_config, resolve_config_path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
_DEFAULT_LOG_MAX_BYTES = 1_000_000
_DEFAULT_LOG_BACKUP_COUNT = 5


def _load_logging_config(config_path: str | Path | None) -> dict:
    try:
        return load_app_config(resolve_config_path(config_path))
    except Exception:
        return {}


def _resolve_log_path(config_path: str | Path | None, log_filename: str) -> Path:
    config = _load_logging_config(config_path)
    log_dir_value = config.get("log_dir")
    log_dir = Path(log_dir_value).expanduser() if log_dir_value else _DEFAULT_LOG_DIR
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / log_filename


def _resolve_max_bytes(config_path: str | Path | None) -> int:
    config = _load_logging_config(config_path)
    return int(config.get("log_max_bytes", _DEFAULT_LOG_MAX_BYTES))


def _resolve_backup_count(config_path: str | Path | None) -> int:
    config = _load_logging_config(config_path)
    return int(config.get("log_backup_count", _DEFAULT_LOG_BACKUP_COUNT))


def configure_logging(
    logger_name: str,
    *,
    config_path: str | Path | None = None,
    log_filename: str = "weather.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure console + rotating file logging once per process."""
    root = logging.getLogger()
    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if not any(getattr(handler, "_weather_console_handler", False) for handler in root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler._weather_console_handler = True
        root.addHandler(console_handler)

    log_path = _resolve_log_path(config_path, log_filename)
    if not any(
        isinstance(handler, RotatingFileHandler) and Path(getattr(handler, "baseFilename", "")) == log_path
        for handler in root.handlers
    ):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=_resolve_max_bytes(config_path),
            backupCount=_resolve_backup_count(config_path),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger


def set_log_level(level: int) -> None:
    logging.getLogger().setLevel(level)
    logging.getLogger("weather").setLevel(level)
