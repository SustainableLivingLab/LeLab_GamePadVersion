"""The task sentence of a running AI session, changeable while it runs.

LeLab writes the sentence to a small file and gives the rollout process its
path in LELAB_TASK_FILE; the process re-reads it (only when the file changed)
each time the policy plans. Switching the sentence (e.g. Lesson 6's blue /
green cube) then takes effect at the next plan, with the same model loaded.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

ENV_VAR = "LELAB_TASK_FILE"

logger = logging.getLogger(__name__)


def write(path: Path, task: str) -> None:
    # Write then rename, so a reader never sees a half-written sentence.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(task, encoding="utf-8")
    os.replace(tmp, path)


class LiveTask:
    """Current task: the file's sentence if there is one, else `default`."""

    def __init__(self, default: str):
        self._value = default
        self._path = Path(os.environ[ENV_VAR]) if os.environ.get(ENV_VAR) else None
        self._mtime: float | None = None

    def get(self) -> str:
        if self._path is None:
            return self._value
        try:
            mtime = self._path.stat().st_mtime
            if mtime != self._mtime:
                self._mtime = mtime
                task = self._path.read_text(encoding="utf-8").strip()
                if task and task != self._value:
                    logger.info("Task changed to: %s", task)
                    self._value = task
        except OSError:
            pass
        return self._value
