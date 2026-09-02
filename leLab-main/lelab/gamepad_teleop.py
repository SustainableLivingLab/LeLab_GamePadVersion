# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Joint-space gamepad teleoperator for the SO-101 follower arm.

Ports the mapping/tuning from the standalone `gamepad_teleop.py` / `gamepad_control.py`
scripts (velocity control: stick/trigger deflection sets how FAST a joint moves, not a
target position) into a `Teleoperator` subclass so it drops into both lelab's
teleoperation worker loop and lerobot's stock `record_loop()` exactly like `SO101Leader`
does -- `get_action()` returns the same `{motor}.pos: float}` shape.

This intentionally does NOT reuse lerobot's built-in `lerobot.teleoperators.gamepad`
class: that one emits end-effector deltas (`delta_x/y/z` + gripper) for IK-driven
robots, a different shape than the joint-space `SOFollower` this app drives.

Confirmed working with a PS5 DualSense and a wired Logitech G F310. The one thing that
varies by controller is how the D-pad is reported (buttons vs. a hat), which is
auto-detected. Always re-verify axis/button indices with a real controller before
trusting a new one -- axis order and polarity are not guaranteed across brands/drivers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_not_connected

logger = logging.getLogger(__name__)

FPS = 50
DEADZONE = 0.08

# Axis indices confirmed via a pygame debug probe for the reference controllers
# (PS5 DualSense, Logitech G F310) -- both report 6 axes, 0-5, in the same order.
AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_RIGHT_X = 2
AXIS_RIGHT_Y = 3
AXIS_L2 = 4
AXIS_R2 = 5
# D-pad fallback for controllers that report it as buttons rather than a hat.
BUTTON_DPAD_UP = 11
BUTTON_DPAD_DOWN = 12
BUTTON_START_PAUSE = 0  # Cross
BUTTON_QUIT = 1  # Circle
BUTTON_HOME = 3  # Triangle

# How long the smooth return-to-start move takes, so pressing Triangle from far
# away doesn't yank the arm at full speed.
HOME_MOVE_DURATION_S = 1.5

# How often to retry opening the joystick after it drops mid-session
# (Bluetooth hiccup, dongle out of range, USB unplug). A Joystick() open can
# take noticeably longer than one control tick, so this is throttled rather
# than attempted every get_action() call.
RECONNECT_INTERVAL_S = 1.0

# Which stick axis drives each joint, its sign, and how fast it moves
# (degrees/sec at full stick deflection).
JOINT_CONFIG = {
    "shoulder_pan": {"axis": AXIS_LEFT_X, "sign": 1, "max_deg_per_s": 40.0},
    "shoulder_lift": {"axis": AXIS_LEFT_Y, "sign": -1, "max_deg_per_s": 40.0},
    "elbow_flex": {"axis": AXIS_RIGHT_Y, "sign": -1, "max_deg_per_s": 40.0},
    # Wrist roll gets a wider deadzone -- this axis can show small phantom values
    # at rest that the global deadzone doesn't fully catch, causing slow drift.
    "wrist_roll": {"axis": AXIS_RIGHT_X, "sign": -1, "max_deg_per_s": 60.0, "deadzone": 0.2},
}
WRIST_FLEX_SIGN = -1
WRIST_FLEX_MAX_DEG_PER_S = 40.0
GRIPPER_SIGN = -1  # positive = R2 closes, L2 opens (matches "lower = more closed")
GRIPPER_MAX_PER_S = 60.0  # units/sec on the 0-100 gripper scale

# Soft position limits (degrees) so a runaway stick can't drive a joint past a
# sane range even without hitting the servo's own hard limits.
JOINT_MIN_DEG = -100.0
JOINT_MAX_DEG = 100.0


def apply_deadzone(value: float, deadzone: float = DEADZONE) -> float:
    return 0.0 if abs(value) < deadzone else value


def get_dpad_delta(js) -> float:
    """+1 for D-pad up, -1 for down, 0 otherwise -- works whether the controller
    reports the D-pad as a hat (most Xbox-layout pads) or as buttons (DualSense).
    """
    if js.get_numhats() > 0:
        _, hat_y = js.get_hat(0)
        return float(hat_y)

    dpad_delta = 0.0
    if js.get_button(BUTTON_DPAD_UP):
        dpad_delta += 1.0
    if js.get_button(BUTTON_DPAD_DOWN):
        dpad_delta -= 1.0
    return dpad_delta


def step_targets(js, current_targets: dict[str, float], dt: float) -> dict[str, float]:
    """Advance current_targets by one control-loop tick based on live stick/trigger state.

    Mutates and returns current_targets.
    """
    for joint, cfg in JOINT_CONFIG.items():
        raw = apply_deadzone(js.get_axis(cfg["axis"]), cfg.get("deadzone", DEADZONE))
        vel = cfg["sign"] * raw * cfg["max_deg_per_s"]
        new_val = current_targets[joint] + vel * dt
        current_targets[joint] = max(JOINT_MIN_DEG, min(JOINT_MAX_DEG, new_val))

    dpad_delta = get_dpad_delta(js)
    if dpad_delta != 0.0:
        vel = WRIST_FLEX_SIGN * dpad_delta * WRIST_FLEX_MAX_DEG_PER_S
        new_val = current_targets["wrist_flex"] + vel * dt
        current_targets["wrist_flex"] = max(JOINT_MIN_DEG, min(JOINT_MAX_DEG, new_val))

    l2 = (js.get_axis(AXIS_L2) + 1.0) / 2.0  # 0 (released) .. 1 (pressed)
    r2 = (js.get_axis(AXIS_R2) + 1.0) / 2.0
    gripper_vel = GRIPPER_SIGN * (r2 - l2) * GRIPPER_MAX_PER_S
    new_gripper = current_targets["gripper"] + gripper_vel * dt
    current_targets["gripper"] = max(0.0, min(100.0, new_gripper))

    return current_targets


def move_to_home(bus, start: dict, home_position: dict, fps: int, duration_s: float = HOME_MOVE_DURATION_S) -> dict:
    """Blocking, smooth ramp from `start` to `home_position`. Returns the final target dict."""
    steps = max(1, int(duration_s * fps))
    control_interval = 1.0 / fps
    for step in range(1, steps + 1):
        alpha = step / steps
        interp = {joint: start[joint] + (home_position[joint] - start[joint]) * alpha for joint in start}
        bus.sync_write("Goal_Position", interp)
        time.sleep(control_interval)
    return dict(home_position)


@TeleoperatorConfig.register_subclass("gamepad_so101")
@dataclass
class GamepadSO101TeleopConfig(TeleoperatorConfig):
    """Config for GamepadSO101Teleop. `id` is unused (no calibration file) but kept
    for interface parity with other teleoperator configs.
    """

    fps: int = FPS
    joystick_index: int = 0
    id: str = field(default="gamepad")


class GamepadSO101Teleop(Teleoperator):
    """Velocity-controlled joint-space gamepad teleoperator for the SO-101 follower.

    Unlike a leader arm, this teleoperator has no position of its own to read --
    it accumulates a target position starting from wherever the follower currently
    is. Call `seed(robot)` once after the follower connects (and again any time the
    arm's position changes outside gamepad control, e.g. after a reset phase) so the
    first `get_action()` doesn't jump the arm to a stale target.
    """

    config_class = GamepadSO101TeleopConfig
    name = "gamepad_so101"

    def __init__(self, config: GamepadSO101TeleopConfig):
        super().__init__(config)
        self.config = config
        self._pygame = None
        self._joystick = None
        self._targets: dict[str, float] | None = None
        self._running = False  # gated by Cross (BUTTON_START_PAUSE); arm holds position while False
        self._last_tick = None
        # Debounces Cross/Triangle so a held button doesn't retrigger every tick.
        self._start_pause_held = False
        self._home_held = False
        self._quit_requested = False
        # Set via seed(); used to service a Triangle press without a robot reference.
        self._robot = None
        # A dropped controller (Bluetooth hiccup, dongle out of range, USB
        # unplug) shouldn't crash the whole teleop/recording session -- see
        # get_action()'s try/except. These track that degraded state so the
        # GUI can show it and so reconnect attempts are rate-limited instead
        # of retrying a slow Joystick() open every single control tick.
        self._gamepad_connected = True
        self._last_reconnect_attempt = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        return {f"{motor}.pos": float for motor in motors}

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
        """Capture the follower's current position as the gamepad's starting target
        and "home" position (what Triangle returns to). Call once after the follower
        connects, and again after any reset phase where the arm moved without the
        gamepad driving it.
        """
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
            raise RuntimeError("GamepadSO101Teleop.seed(robot) must be called before get_action().")

        if not self._gamepad_connected:
            if not self._attempt_reconnect():
                # Still gone: hold the arm exactly where it is and skip
                # every button/stick read below rather than raise -- a
                # raised exception here would crash the whole teleop worker
                # (or lerobot's record_loop, in recording mode) and, worse,
                # take the follower arm's connection down with it.
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
                    logger.info("Gamepad arm motion %s.", "ON" if self._running else "OFF (holding position)")
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
                step_targets(self._joystick, self._targets, dt)
        except Exception as e:
            # The controller went away mid-read (Bluetooth hiccup, dongle out
            # of range, USB unplug). Freeze the arm in place -- force
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
        """Extra button state the web layer polls independently of get_action()
        (e.g. to surface a "quit requested" toast). Circle (BUTTON_QUIT) latches
        `quit_requested` until explicitly cleared by the caller.
        """
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
