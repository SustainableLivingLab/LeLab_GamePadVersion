# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-axis gamepad teleoperator for `ClawFollower` (K12 app Lesson 2.2).

`GamepadSO101Teleop` (gamepad_teleop.py) drives all six SO-101 joints and
hardcodes their names throughout, so it can't be pointed at a one-motor
"gripper"-only bus. This is the same L2/R2-only slice of that class's
behavior -- trigger deflection sets gripper velocity, Cross starts/pauses,
Triangle returns home, Circle quits -- with the five arm joints removed.
Shares its axis/button constants, deadzone helper, reconnect interval, and
home-move ramp with gamepad_teleop.py rather than redefining them, and
mirrors its connect()/get_action()/reconnect structure exactly (see
GamepadSO101Teleop's own docstrings for why each piece is shaped the way it
is -- eager connect() on the worker thread for SDL's thread affinity, and a
try/except + rate-limited reconnect in get_action() so a dropped controller
freezes the claw in place instead of killing the whole session).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_not_connected

from .gamepad_teleop import (
    AXIS_L2,
    AXIS_R2,
    BUTTON_HOME,
    BUTTON_QUIT,
    BUTTON_START_PAUSE,
    FPS,
    GRIPPER_MAX_PER_S,
    GRIPPER_SIGN,
    HOME_MOVE_DURATION_S,
    RECONNECT_INTERVAL_S,
    move_to_home,
)

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("gamepad_claw")
@dataclass
class GamepadClawTeleopConfig(TeleoperatorConfig):
    """Config for ClawGamepadTeleop. `id` is unused (no calibration file) but kept
    for interface parity with other teleoperator configs.
    """

    fps: int = FPS
    joystick_index: int = 0
    id: str = field(default="gamepad")


class ClawGamepadTeleop(Teleoperator):
    """Velocity-controlled gamepad teleoperator for a single-motor `ClawFollower`.

    Same seed-then-accumulate model as `GamepadSO101Teleop`: there's no leader
    position to read, so `seed(robot)` captures wherever the claw currently is
    as both the starting target and the "home" position Triangle returns to.
    """

    config_class = GamepadClawTeleopConfig
    name = "gamepad_claw"

    def __init__(self, config: GamepadClawTeleopConfig):
        super().__init__(config)
        self.config = config
        self._pygame = None
        self._joystick = None
        self._targets: dict[str, float] | None = None
        self._home_position: dict[str, float] | None = None
        self._running = False
        self._last_tick = None
        self._start_pause_held = False
        self._home_held = False
        self._quit_requested = False
        self._robot = None
        # A dropped controller (Bluetooth hiccup, dongle out of range, USB
        # unplug) shouldn't crash the whole teleop session -- see
        # get_action()'s try/except. These track that degraded state so the
        # GUI can show it and so reconnect attempts are rate-limited instead
        # of retrying a slow Joystick() open every single control tick.
        self._gamepad_connected = True
        self._last_reconnect_attempt = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        return {"gripper.pos": float}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._joystick is not None

    def connect(self, calibrate: bool = True) -> None:
        import pygame

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()
        pygame.event.pump()
        if pygame.joystick.get_count() == 0:
            # A Bluetooth controller paired *after* pygame's joystick subsystem
            # last initialized won't show up in get_count() until SDL
            # re-enumerates devices -- quit()+init() forces that rescan rather
            # than trusting a snapshot that may predate the pairing.
            pygame.joystick.quit()
            pygame.joystick.init()
            pygame.event.pump()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No gamepad detected. Connect a controller and try again. If you just paired "
                "it over Bluetooth, wait a few seconds after pairing completes, then retry."
            )
        self._joystick = pygame.joystick.Joystick(self.config.joystick_index)
        self._joystick.init()
        logger.info(f"Gamepad connected: {self._joystick.get_name()}")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def seed(self, robot) -> None:
        self._robot = robot
        present = dict(robot.bus.sync_read("Present_Position"))
        self._targets = dict(present)
        self._home_position = dict(present)
        self._running = False

    def _attempt_reconnect(self) -> bool:
        """Try to re-open the joystick after a drop. Rate-limited by
        RECONNECT_INTERVAL_S so a persistently-gone controller doesn't stall
        the control loop retrying every tick. Returns True once a fresh
        Joystick() open AND a real read both succeed -- a successful open
        alone can still be a zombie handle on some drivers.
        """
        now = time.perf_counter()
        if now - self._last_reconnect_attempt < RECONNECT_INTERVAL_S:
            return False
        self._last_reconnect_attempt = now

        pygame = self._pygame
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            pygame.event.pump()
            if pygame.joystick.get_count() == 0:
                return False
            joystick = pygame.joystick.Joystick(self.config.joystick_index)
            joystick.init()
            joystick.get_axis(0)  # confirms the handle actually reads, not just opens
        except Exception:
            return False

        self._joystick = joystick
        self._gamepad_connected = True
        logger.info("Gamepad reconnected: %s", joystick.get_name())
        return True

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        if self._targets is None:
            raise RuntimeError("ClawGamepadTeleop.seed(robot) must be called before get_action().")

        if not self._gamepad_connected:
            if not self._attempt_reconnect():
                # Still gone: hold the claw exactly where it is and skip
                # every button/stick read below rather than raise -- a
                # raised exception here would crash the whole teleop worker
                # and, worse, take the follower's connection down with it.
                return {f"{joint}.pos": val for joint, val in self._targets.items()}

        pygame = self._pygame
        try:
            pygame.event.pump()

            if self._joystick.get_button(BUTTON_QUIT):
                self._quit_requested = True

            if self._joystick.get_button(BUTTON_START_PAUSE):
                if not self._start_pause_held:
                    self._start_pause_held = True
                    self._running = not self._running
                    if self._running and self._robot is not None:
                        self._targets = dict(self._robot.bus.sync_read("Present_Position"))
                    logger.info("Gamepad claw motion %s.", "ON" if self._running else "OFF (holding position)")
            else:
                self._start_pause_held = False

            if self._joystick.get_button(BUTTON_HOME):
                if not self._home_held:
                    self._home_held = True
                    if self._running and self._robot is not None:
                        self._targets = move_to_home(
                            self._robot.bus, self._targets, self._home_position, self.config.fps, HOME_MOVE_DURATION_S
                        )
            else:
                self._home_held = False

            now = time.perf_counter()
            dt = (now - self._last_tick) if self._last_tick is not None else (1.0 / self.config.fps)
            self._last_tick = now

            if self._running:
                l2 = (self._joystick.get_axis(AXIS_L2) + 1.0) / 2.0  # 0 (released) .. 1 (pressed)
                r2 = (self._joystick.get_axis(AXIS_R2) + 1.0) / 2.0
                gripper_vel = GRIPPER_SIGN * (r2 - l2) * GRIPPER_MAX_PER_S
                new_gripper = self._targets["gripper"] + gripper_vel * dt
                self._targets["gripper"] = max(0.0, min(100.0, new_gripper))
        except Exception as e:
            # The controller went away mid-read (Bluetooth hiccup, dongle out
            # of range, USB unplug). Freeze the claw in place -- force
            # `_running` off so a later reconnect doesn't resume mid-motion
            # from stale stick state -- and let future ticks retry the
            # connection instead of tearing down the session.
            logger.warning("Gamepad read failed, will retry: %s", e)
            self._gamepad_connected = False
            self._running = False
            self._start_pause_held = False
            self._home_held = False

        return {f"{joint}.pos": val for joint, val in self._targets.items()}

    def get_teleop_events(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "quit_requested": self._quit_requested,
            "gamepad_connected": self._gamepad_connected,
        }

    def clear_quit_requested(self) -> None:
        self._quit_requested = False

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self._joystick is not None:
            try:
                self._joystick.quit()
            except Exception:
                # The handle may already be dead if the controller dropped
                # and was never successfully reconnected -- teardown must
                # still proceed.
                pass
            self._joystick = None
        if self._pygame is not None:
            self._pygame.joystick.quit()
            self._pygame.quit()
            self._pygame = None
        self._targets = None
        self._robot = None
        logger.info("Gamepad disconnected.")
