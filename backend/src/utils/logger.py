"""
Centralised logging configuration for the Context-Aware Retrieval Engine.

Provides a factory function ``get_logger`` that returns a consistently
configured :class:`logging.Logger` with both console and rotating file
handlers.  All log levels, formats, and file paths are driven from the
top-level YAML config so the rest of the codebase never has to hardcode
logging parameters.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import pathlib
from typing import Optional

import yaml

# ── default constants (overridden by config when available) ──────────────────
_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "%(asctime)s | %(name)-35s | %(levelname)-8s | %(message)s"
_DEFAULT_LOG_FILE = "outputs/logs/retrieval.log"

_CONFIGURED = False  # module-level flag so we only call basicConfig once


def _load_config() -> dict:
    """Load config.yaml from the expected location relative to this file."""
    config_path = pathlib.Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _ensure_log_directory(log_file: str) -> None:
    log_dir = pathlib.Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)


def configure_root_logger(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger once per process.

    Parameters
    ----------
    level:
        Override log level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    log_format:
        Override the log format string.
    log_file:
        Override the log file path.  Pass ``""`` to disable file logging.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = _load_config()
    log_cfg: dict = cfg.get("logging", {})

    resolved_level = level or log_cfg.get("level", _DEFAULT_LEVEL)
    resolved_format = log_format or log_cfg.get("format", _DEFAULT_FORMAT)
    resolved_file = log_file if log_file is not None else log_cfg.get("file", _DEFAULT_LOG_FILE)

    numeric_level = getattr(logging, resolved_level.upper(), logging.INFO)
    formatter = logging.Formatter(resolved_format, datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # ── console handler ──────────────────────────────────────────────────────
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(numeric_level)
        root.addHandler(console)

    # ── rotating file handler ────────────────────────────────────────────────
    if resolved_file:
        _ensure_log_directory(resolved_file)
        file_handler = logging.handlers.RotatingFileHandler(
            resolved_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root.addHandler(file_handler)

    # Silence overly verbose third-party loggers
    for noisy in ("transformers", "sentence_transformers", "faiss", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named :class:`logging.Logger`, initialising the root logger
    on first call.

    Parameters
    ----------
    name:
        Logger name — typically ``__name__`` of the calling module.
    level:
        Optional override for this specific logger's level.

    Returns
    -------
    logging.Logger
    """
    configure_root_logger()
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
