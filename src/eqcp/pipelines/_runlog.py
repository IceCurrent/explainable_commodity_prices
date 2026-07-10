"""Shared run logger for pipelines (console + accumulated run_log.txt)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("eqcp.pipelines")


class RunLogger:
    """Log to the logging module and accumulate lines for run_log.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        logger.info(msg)
        self.lines.append(msg)

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")
