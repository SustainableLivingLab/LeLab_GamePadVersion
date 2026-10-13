"""Which gamepad control drives which arm action (K12 lessons' Reconfigure).

An *action* is one motor moving one way (e.g. M2 up) or a session command
(start/pause, back to start, end / re-record episode). An *input* is one physical control in one
direction (e.g. left stick up, L2, D-pad left, Cross). The saved mapping is
{action: input}; each input drives at most one action. It's saved once per
computer and read by GamepadSO101Teleop whenever arm control or a recording
starts, so a change applies from the next Start Control.

DEFAULT_MAPPING reproduces the original fixed layout exactly (same axes,
directions and speeds), so nothing changes for anyone who never reconfigures.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from lerobot.utils.constants import HF_LEROBOT_HOME

logger = logging.getLogger(__name__)

MAPPING_FILE = str(HF_LEROBOT_HOME / "lelab_gamepad_mapping.json")

# Motor actions: action id -> (joint, sign, speed). sign/speed reproduce the
# original JOINT_CONFIG / WRIST_FLEX / GRIPPER constants so the default
# mapping drives every joint exactly as before. Speeds are degrees/sec (0-100
# units/sec for the gripper) at full input.
MOTOR_ACTIONS: dict[str, tuple[str, int, float]] = {
    "m1_left": ("shoulder_pan", -1, 40.0),
    "m1_right": ("shoulder_pan", 1, 40.0),
    "m2_up": ("shoulder_lift", 1, 40.0),
    "m2_down": ("shoulder_lift", -1, 40.0),
    "m3_up": ("elbow_flex", 1, 40.0),
    "m3_down": ("elbow_flex", -1, 40.0),
    "m4_up": ("wrist_flex", -1, 40.0),
    "m4_down": ("wrist_flex", 1, 40.0),
    "m5_left": ("wrist_roll", 1, 60.0),
    "m5_right": ("wrist_roll", -1, 60.0),
    "m6_open": ("gripper", 1, 60.0),
    "m6_close": ("gripper", -1, 60.0),
}
# Commands, fired once when their input is pressed. end_episode /
# rerecord_episode only do something during a recording (same as the page's
# Finish Episode / Re-record Episode buttons); they're unmapped by default so
# nobody triggers them by accident.
COMMAND_ACTIONS = ("start_pause", "home", "end_episode", "rerecord_episode")
ACTIONS = tuple(MOTOR_ACTIONS) + COMMAND_ACTIONS

INPUTS = (
    "ls_up", "ls_down", "ls_left", "ls_right",
    "rs_up", "rs_down", "rs_left", "rs_right",
    "l1", "r1", "l2", "r2",
    "dpad_up", "dpad_down", "dpad_left", "dpad_right",
    "cross", "circle", "square", "triangle",
)

DEFAULT_MAPPING: dict[str, str] = {
    "m1_left": "ls_left",
    "m1_right": "ls_right",
    "m2_up": "ls_up",
    "m2_down": "ls_down",
    "m3_up": "rs_up",
    "m3_down": "rs_down",
    "m4_up": "dpad_up",
    "m4_down": "dpad_down",
    "m5_left": "rs_left",
    "m5_right": "rs_right",
    "m6_open": "l2",
    "m6_close": "r2",
    "start_pause": "cross",
    "home": "triangle",
}

# Stick deadzones. The right stick's X axis (wrist roll by default) can show
# small phantom values at rest, so it keeps the wider deadzone it always had.
STICK_DEADZONE = 0.08
RIGHT_X_DEADZONE = 0.2
# Triggers had no deadzone in the original layout; keep it that way.
TRIGGER_DEADZONE = 0.0
# An analog input counts as "pressed" for a command above this.
COMMAND_THRESHOLD = 0.5


def validate_mapping(mapping: Any) -> dict[str, str]:
    """A clean {action: input} dict, or ValueError: known actions and inputs
    only, and no input used twice."""
    if not isinstance(mapping, dict):
        raise ValueError("The mapping must be an object of action -> input.")
    clean: dict[str, str] = {}
    used: dict[str, str] = {}
    for action, control in mapping.items():
        if control in (None, ""):
            continue
        if action not in ACTIONS:
            raise ValueError(f"Unknown action {action!r}.")
        if control not in INPUTS:
            raise ValueError(f"Unknown control {control!r}.")
        if control in used:
            raise ValueError(f"{control!r} is assigned to both {used[control]!r} and {action!r}.")
        used[control] = action
        clean[action] = control
    return clean


def load_mapping() -> dict[str, str]:
    """The saved mapping, or the default when none is saved (or it's broken)."""
    try:
        with open(MAPPING_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        saved = data.get("mapping", {})
        # Actions that no longer exist (e.g. the old "stop") are dropped
        # rather than throwing away the whole saved mapping.
        if isinstance(saved, dict):
            saved = {action: control for action, control in saved.items() if action in ACTIONS}
        return validate_mapping(saved)
    except FileNotFoundError:
        return dict(DEFAULT_MAPPING)
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable gamepad mapping %s (%s); using the default", MAPPING_FILE, exc)
        return dict(DEFAULT_MAPPING)


def save_mapping(mapping: Any) -> dict[str, str]:
    clean = validate_mapping(mapping)
    os.makedirs(os.path.dirname(MAPPING_FILE), exist_ok=True)
    tmp = MAPPING_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "mapping": clean}, fh, indent=2)
    os.replace(tmp, MAPPING_FILE)
    logger.info("Saved gamepad mapping (%d actions)", len(clean))
    return clean


def reset_mapping() -> dict[str, str]:
    try:
        os.remove(MAPPING_FILE)
    except FileNotFoundError:
        pass
    return dict(DEFAULT_MAPPING)


class InputReader:
    """Reads logical inputs (0..1) from a pygame joystick.

    Sticks and triggers are axes 0-5 on the supported controllers (DualSense,
    Xbox-style XInput pads such as the Logitech F310 in X mode). Face buttons
    are 0-3 on both. Where they differ: XInput pads report the D-pad as a hat
    and L1/R1 as buttons 4/5; a DualSense (no hat) reports the D-pad as
    buttons 11-14 and L1/R1 as 9/10.
    """

    def __init__(self, js) -> None:
        self.js = js
        self.has_hat = js.get_numhats() > 0
        self.n_axes = js.get_numaxes()
        self.n_buttons = js.get_numbuttons()

    def _axis(self, index: int, default: float = 0.0) -> float:
        return self.js.get_axis(index) if index < self.n_axes else default

    def _button(self, index: int) -> float:
        return 1.0 if index < self.n_buttons and self.js.get_button(index) else 0.0

    def _half(self, index: int, positive: bool, deadzone: float) -> float:
        value = self._axis(index)
        if abs(value) < deadzone:
            return 0.0
        value = value if positive else -value
        return max(0.0, min(1.0, value))

    def _trigger(self, index: int) -> float:
        value = (self._axis(index, -1.0) + 1.0) / 2.0  # -1 released .. 1 pressed
        return 0.0 if value <= TRIGGER_DEADZONE else min(1.0, value)

    def _dpad(self, direction: str) -> float:
        if self.has_hat:
            hat_x, hat_y = self.js.get_hat(0)
            return float(
                {"up": hat_y > 0, "down": hat_y < 0, "left": hat_x < 0, "right": hat_x > 0}[direction]
            )
        return self._button({"up": 11, "down": 12, "left": 13, "right": 14}[direction])

    def read(self, name: str) -> float:
        if name == "ls_up":
            return self._half(1, False, STICK_DEADZONE)
        if name == "ls_down":
            return self._half(1, True, STICK_DEADZONE)
        if name == "ls_left":
            return self._half(0, False, STICK_DEADZONE)
        if name == "ls_right":
            return self._half(0, True, STICK_DEADZONE)
        if name == "rs_up":
            return self._half(3, False, STICK_DEADZONE)
        if name == "rs_down":
            return self._half(3, True, STICK_DEADZONE)
        if name == "rs_left":
            return self._half(2, False, RIGHT_X_DEADZONE)
        if name == "rs_right":
            return self._half(2, True, RIGHT_X_DEADZONE)
        if name == "l2":
            return self._trigger(4)
        if name == "r2":
            return self._trigger(5)
        if name == "l1":
            return self._button(4 if self.has_hat else 9)
        if name == "r1":
            return self._button(5 if self.has_hat else 10)
        if name.startswith("dpad_"):
            return self._dpad(name[5:])
        if name in ("cross", "circle", "square", "triangle"):
            return self._button(("cross", "circle", "square", "triangle").index(name))
        return 0.0


def joint_velocities(reader: InputReader, mapping: dict[str, str]) -> dict[str, float]:
    """Per-joint velocity this tick from every mapped motor action."""
    velocities: dict[str, float] = {}
    for action, (joint, sign, speed) in MOTOR_ACTIONS.items():
        control = mapping.get(action)
        if not control:
            continue
        amount = reader.read(control)
        if amount:
            velocities[joint] = velocities.get(joint, 0.0) + sign * amount * speed
    return velocities


def command_pressed(reader: InputReader, mapping: dict[str, str], command: str) -> bool:
    control = mapping.get(command)
    return bool(control) and reader.read(control) > COMMAND_THRESHOLD
