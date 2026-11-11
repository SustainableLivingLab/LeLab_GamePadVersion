"""lerobot's own rollout script, with a task that can change while it runs.

lerobot's inference engines keep the task sentence they were started with
(self._task) and read it on every plan. This swaps that attribute for one
that reads LeLab's live task file (utils/live_task.py), then runs the script
unchanged: python -m lelab.local_rollout <lerobot_rollout arguments>.

When LeLab asks for it, the top camera also picks the task by itself
(cube_colors.py): every observation the run reads from the arm passes
through here, and a few times a second the cube watcher looks at it.
"""

import os
import threading
import time

from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.inference.sync import SyncInferenceEngine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.scripts.lerobot_rollout import main

from lelab import cube_colors
from lelab.utils.live_task import LiveTask

# Seconds between the cube watcher's looks.
WATCH_EVERY_S = 0.33


def _get_task(self) -> str:
    return self.__dict__["_live_task"].get()


def _set_task(self, task: str) -> None:
    self.__dict__["_live_task"] = LiveTask(task)
    _watching.task = task


for engine in (SyncInferenceEngine, RTCInferenceEngine):
    engine._task = property(_get_task, _set_task)


class _Watching:
    """The cube watcher, made on the first observation (the camera names
    come from the robot)."""

    def __init__(self) -> None:
        self.task = ""
        self.watcher = None
        self.made = False
        self.last = 0.0
        self.lock = threading.Lock()

    def see(self, robot: ThreadSafeRobot, obs: dict) -> None:
        # RTC reads observations from a second thread too; one look at a time.
        if not self.lock.acquire(blocking=False):
            return
        try:
            if not self.made:
                self.made = True
                cameras = list(getattr(robot._robot, "cameras", {}) or {})
                self.watcher = cube_colors.make_watcher(self.task, cameras)
            now = time.monotonic()
            if self.watcher is None or now - self.last < WATCH_EVERY_S:
                return
            self.last = now
            cube_colors.watch(self.watcher, obs)
        except Exception as exc:  # noqa: BLE001 - the run works without it
            cube_colors.logger.warning("Cube colours: look failed (%s)", exc)
        finally:
            self.lock.release()


_watching = _Watching()
_get_observation = ThreadSafeRobot.get_observation


def _observe(self: ThreadSafeRobot) -> dict:
    obs = _get_observation(self)
    _watching.see(self, obs)
    return obs


if cube_colors.CUBE_WATCH_ENV in os.environ:
    ThreadSafeRobot.get_observation = _observe


if __name__ == "__main__":
    main()
