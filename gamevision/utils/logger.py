"""Structured logging setup."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logging.root.addHandler(handler)
        logging.root.setLevel(level)
        _CONFIGURED = True

    return logger
