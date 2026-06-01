"""Central logging setup — console + logs/retrieval.log for debugging search quality."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import config

_CONFIGURED = False


def setup_logging(level: str | None = None) -> logging.Logger:
    """
    Configure root retrieval logger.

    Set LOG_LEVEL=DEBUG in the environment (or .env) to see cosine
    distances, cleaned queries, and per-chunk scores.
    """
    global _CONFIGURED
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, log_level, logging.INFO)

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger("subtitle_search")
    root.setLevel(numeric)

    if not _CONFIGURED:
        root.handlers.clear()

        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        console.setLevel(numeric)
        root.addHandler(console)

        file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)  # file always keeps detail
        root.addHandler(file_handler)

        _CONFIGURED = True
    else:
        root.setLevel(numeric)
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stderr:
                h.setLevel(numeric)

    return root


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"subtitle_search.{name}")
