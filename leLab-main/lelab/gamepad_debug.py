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

"""Live gamepad probe for the web UI -- the GUI equivalent of the standalone
`gamepad_debug.py` script: lets a user confirm a controller is detected and see
live axis/button/hat values *before* trusting it to drive the robot, without
needing a terminal.

Independent of any active teleoperation/recording session -- opens its own
short-lived pygame joystick handle on demand. Refuses to run concurrently with
an active gamepad teleop session (both would fight over the same pygame
joystick subsystem); callers should point the user at the live session's own
status instead in that case.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Idle handles are released after this long so a forgotten-open debug panel
# doesn't hold the joystick open (and out of reach of a teleop session) forever.
_IDLE_TIMEOUT_S = 30.0


class GamepadProbe:
    """Owns a single on-demand pygame joystick handle for the debug endpoint."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pygame = None
        self._joystick = None
        self._last_poll = 0.0

    def _ensure_open(self) -> None:
        if self._joystick is not None:
            return
        import pygame

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected.")
        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()
        logger.info(f"Gamepad debug probe connected: {self._joystick.get_name()}")

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._joystick is not None:
            try:
                self._joystick.quit()
            except Exception:
                pass
            self._joystick = None
        if self._pygame is not None:
            try:
                self._pygame.joystick.quit()
            except Exception:
                pass
            self._pygame = None

    def read(self) -> dict[str, Any]:
        """Open (or reuse) a joystick handle and return its current state."""
        with self._lock:
            try:
                self._ensure_open()
            except Exception as e:
                self._close_locked()
                return {"connected": False, "message": str(e)}

            pygame = self._pygame
            js = self._joystick
            try:
                pygame.event.pump()
                axes = [round(js.get_axis(i), 3) for i in range(js.get_numaxes())]
                buttons = [bool(js.get_button(i)) for i in range(js.get_numbuttons())]
                hats = [list(js.get_hat(i)) for i in range(js.get_numhats())]
                name = js.get_name()
            except Exception as e:
                # Controller likely unplugged mid-read -- drop the stale handle so
                # the next poll re-detects instead of erroring forever.
                logger.warning(f"Gamepad debug probe read failed, releasing handle: {e}")
                self._close_locked()
                return {"connected": False, "message": str(e)}

            self._last_poll = time.time()
            return {
                "connected": True,
                "name": name,
                "num_axes": len(axes),
                "num_buttons": len(buttons),
                "num_hats": len(hats),
                "axes": axes,
                "buttons": buttons,
                "hats": hats,
            }

    def release_if_idle(self) -> None:
        """Drop the handle if nobody has polled in a while. Call this from a
        background sweep so a forgotten-open browser tab doesn't hold the
        joystick forever and block a real teleop session from opening it."""
        with self._lock:
            if self._joystick is not None and time.time() - self._last_poll > _IDLE_TIMEOUT_S:
                logger.info("Gamepad debug probe idle -- releasing handle.")
                self._close_locked()


# Global probe instance, mirroring the CalibrationManager singleton pattern.
gamepad_probe = GamepadProbe()


def handle_gamepad_status() -> dict[str, Any]:
    """Handle GET /gamepad-status. Refuses to open a handle while a gamepad
    teleop session already owns the joystick -- reports that session's live
    state instead so the two never fight over the same pygame subsystem.
    """
    from . import teleoperate as _teleoperate
    from .gamepad_teleop import GamepadSO101Teleop

    if _teleoperate.teleoperation_active and isinstance(_teleoperate.current_teleop, GamepadSO101Teleop):
        teleop = _teleoperate.current_teleop
        name = None
        if teleop._joystick is not None:
            try:
                name = teleop._joystick.get_name()
            except Exception:
                name = None
        events = teleop.get_teleop_events()
        return {
            "connected": teleop.is_connected,
            "name": name,
            "message": "Gamepad is in use by the active teleoperation session.",
            "in_session": True,
            "running": events["running"],
        }

    gamepad_probe.release_if_idle()
    return {**gamepad_probe.read(), "in_session": False}
