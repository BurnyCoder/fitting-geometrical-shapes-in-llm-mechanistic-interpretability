"""Timestamped logging to the terminal and to logs/<timestamp>_<phase>.log.

Global context: the user requires every phase (and every LLM prompt/output during activation
extraction) to be logged in real time, untruncated, with timestamps, both to the terminal and to a
file. All modules call get_logger(__name__); run.py calls setup_logging() once per phase.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: str | Path, phase: str, level: int = logging.INFO) -> Path:
    """Attach a console handler and a fresh timestamped file handler to the root logger.

    Returns the log-file path so callers can record it in results metadata.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{stamp}_{phase}.log"
    root = logging.getLogger()
    root.setLevel(level)
    # Remove handlers from a previous phase so each phase writes its own file.
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)
    # Silence chatty third-party loggers that would drown the experiment log.
    for noisy in ("urllib3", "httpx", "matplotlib", "PIL", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Module-level logger; inherits the handlers configured by setup_logging()."""
    return logging.getLogger(name)
