"""Drive the arm with a policy running on the cloud GPU (remote_server.py).

Run by LeLab like lerobot's own rollout (same robot/camera options, same log
markers), as: python -m lelab.remote_rollout --server_url=... --task=...
--duration=... --robot.type=so101_follower --robot.port=... --robot.id=...
--robot.cameras=...

The loop runs at 30 Hz on this computer. The policy plans a chunk of moves
(50 for pi0.5) per request; the next plan is requested while the current one
is still playing, so the arm doesn't stop between plans. A new plan is
time-aligned (moves already "in the past" while it was computed are skipped)
and blended into the current one over a few steps, so the arm doesn't jump.

Safety: the arm holds still until the first plan arrives, holds its position
whenever no plan is available, stops the run after repeated server errors,
and returns smoothly to where it started at the end.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import draccus
import numpy as np

# Registers the robot and camera config types draccus needs to parse.
import lerobot.cameras.opencv  # noqa: F401
import lerobot.robots.so_follower  # noqa: F401
from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig

# force: lerobot's imports above may already have set up logging (at a level
# that hid these INFO lines, including the "Rollout setup complete" marker
# LeLab watches for to show the run's progress).
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(asctime)s remote_rollout %(message)s", force=True
)
logger = logging.getLogger("remote_rollout")

FPS = 30
# Ask for the next plan when this many moves of the current one are left
# (~0.6 s at 30 Hz; a round trip measured ~0.4 s).
REPLAN_AT = 18
# Steps over which a new plan is blended into the old one.
BLEND_STEPS = 10
REQUEST_TIMEOUT_S = 5.0
MAX_FAILURES = 3
JPEG_QUALITY = 80


@dataclass
class RemoteRolloutConfig:
    robot: RobotConfig
    server_url: str
    task: str = ""
    duration: float = 60.0


class PlanClient:
    """Requests plans in a background thread so the 30 Hz loop never waits."""

    def __init__(self, url: str, token: str, task: str):
        import httpx

        self._http = httpx.Client(
            base_url=url, headers={"Authorization": f"Bearer {token}"}, timeout=REQUEST_TIMEOUT_S
        )
        self._task = task
        self._thread: threading.Thread | None = None
        self.result: tuple[list[list[float]], list[str] | None, int] | None = None
        self.error: str | None = None
        self.round_trips: list[float] = []

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request(self, images: dict[str, np.ndarray], state: list[float], step: int) -> None:
        payload = {"images": {k: _jpeg(v) for k, v in images.items()}, "state": state, "task": self._task}
        self.result = None
        self.error = None

        def run():
            started = time.perf_counter()
            try:
                reply = self._http.post("/act", json=payload)
                if reply.status_code != 200:
                    detail = reply.json().get("error") if reply.headers.get("content-type", "").startswith(
                        "application/json"
                    ) else reply.text[:200]
                    raise RuntimeError(f"server answered {reply.status_code}: {detail}")
                data = reply.json()
                self.round_trips.append(time.perf_counter() - started)
                self.result = (data["actions"], data.get("names"), step)
            except Exception as exc:  # noqa: BLE001 - reported to the loop
                self.error = f"{type(exc).__name__}: {exc}"

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._http.close()


def _jpeg(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("could not compress a camera image")
    return base64.b64encode(buf.tobytes()).decode()


def _blend(old: deque, new: list[list[float]]) -> deque:
    """Fade from the moves still queued into the new plan over BLEND_STEPS."""
    old_list = list(old)
    n = min(BLEND_STEPS, len(old_list), len(new))
    blended = []
    for i in range(n):
        w = (i + 1) / (n + 1)
        blended.append([(1 - w) * a + w * b for a, b in zip(old_list[i], new[i], strict=False)])
    return deque(blended + list(new[n:]))


def _return_to(robot, target: dict[str, float], duration_s: float = 3.0) -> None:
    try:
        current = {k: v for k, v in robot.get_observation().items() if k in target}
        steps = max(int(duration_s * 50), 1)
        for step in range(1, steps + 1):
            t = step / steps
            robot.send_action({k: current[k] * (1 - t) + target[k] * t for k in current})
            time.sleep(1 / 50)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not return to the start position: %s", exc)


def main() -> None:
    cfg = draccus.parse(RemoteRolloutConfig)
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise SystemExit("Not logged in to Hugging Face on this computer.")

    robot = make_robot_from_config(cfg.robot)
    logger.info("Connecting robot and cameras...")
    robot.connect()
    camera_keys = list(cfg.robot.cameras)
    motor_names = [f"{m}.pos" for m in robot.bus.motors]
    initial = {k: v for k, v in robot.get_observation().items() if k in motor_names}
    logger.info("Robot connected: %s, cameras %s", motor_names, camera_keys)

    client = PlanClient(cfg.server_url, token, cfg.task)
    queue: deque = deque()
    names = motor_names
    failures = 0
    step = 0
    started = None
    last_report = time.perf_counter()
    try:
        # First plan before anything moves.
        obs = robot.get_observation()
        client.request({k: obs[k] for k in camera_keys}, [float(obs[n]) for n in motor_names], step)
        while client.busy:
            time.sleep(0.01)
        if client.error or not client.result:
            raise RuntimeError(f"The cloud GPU did not answer: {client.error}")
        actions, server_names, _ = client.result
        names = server_names or motor_names
        queue = deque(actions)
        client.result = None
        logger.info("First plan: %d moves in %.0f ms", len(actions), client.round_trips[-1] * 1000)
        logger.info("Rollout setup complete, starting rollout...")
        started = time.perf_counter()

        while time.perf_counter() - started < cfg.duration:
            loop_start = time.perf_counter()
            obs = robot.get_observation()

            if client.result is not None:
                actions, _, sent_step = client.result
                client.result = None
                failures = 0
                # Moves the arm already made while this plan was computed.
                late = step - sent_step
                queue = _blend(queue, actions[late:]) if late < len(actions) else queue
            elif client.error is not None:
                failures += 1
                error, client.error = client.error, None
                logger.warning("Plan request failed (%d/%d): %s", failures, MAX_FAILURES, error)
                if failures >= MAX_FAILURES:
                    raise RuntimeError(f"The cloud GPU stopped answering: {error}")

            if len(queue) <= REPLAN_AT and not client.busy and client.result is None:
                client.request({k: obs[k] for k in camera_keys}, [float(obs[n]) for n in names], step)

            if queue:
                move = queue.popleft()
                robot.send_action(dict(zip(names, move, strict=False)))
            # else: no plan yet, hold position (send nothing).

            step += 1
            if time.perf_counter() - last_report > 10 and client.round_trips:
                recent = client.round_trips[-10:]
                logger.info("Plan round trip: median %.0f ms", 1000 * sorted(recent)[len(recent) // 2])
                last_report = time.perf_counter()
            sleep = 1 / FPS - (time.perf_counter() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
        logger.info("Duration limit reached (%.0fs)", cfg.duration)
    finally:
        client.close()
        logger.info("Returning robot to initial position before shutdown...")
        _return_to(robot, initial)
        robot.disconnect()
        logger.info("Rollout finished")


if __name__ == "__main__":
    main()
