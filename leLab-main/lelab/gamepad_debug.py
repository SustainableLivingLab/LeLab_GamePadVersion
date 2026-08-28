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

All pygame/SDL calls happen on a single dedicated background thread. SDL's
joystick backend on Windows is thread-affine -- reading a Joystick from a
different thread than the one that opened it can silently return a frozen
snapshot instead of live values. FastAPI serves each request on a threadpool
thread (a different one per request), so touching pygame directly from the
request handler was exactly this bug: the debug view showed a static first
reading that never updated. The poller thread below owns pygame end-to-end;
HTTP handlers only ever read a lock-protected cache of its last reading.
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
# How often the poller thread re-reads the joystick.
_POLL_INTERVAL_S = 0.05  # 20 Hz -- comfortably faster than the frontend's 10 Hz poll


class GamepadProbe:
    """Owns a single on-demand pygame joystick handle for the debug endpoint.

    All actual pygame calls happen inside `_poll_loop`, which runs on one
    dedicated thread for the handle's whole lifetime. `read()` (called from
    FastAPI's threadpool) never touches pygame itself -- it only reads
    `self._latest` under `self._lock`, which `_poll_loop` keeps fresh.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest: dict[str, Any] = {"connected": False, "message": "No gamepad detected."}
        self._last_read_request = 0.0

    def _start_poll_thread_locked(self) -> None:
        """Caller must hold self._lock."""
        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="gamepad-debug-poll", daemon=True)
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        import pygame

        try:
            pygame.init()
            pygame.joystick.init()
            pygame.event.pump()
            if pygame.joystick.get_count() == 0:
                # A Bluetooth controller paired *after* pygame's joystick
                # subsystem last initialized won't show up in get_count() until
                # SDL re-enumerates devices -- quit()+init() forces that rescan
                # rather than trusting a snapshot that may predate the pairing.
                pygame.joystick.quit()
                pygame.joystick.init()
                pygame.event.pump()
            if pygame.joystick.get_count() == 0:
                with self._lock:
                    self._latest = {
                        "connected": False,
                        "message": (
                            "No gamepad detected. If you just paired it over Bluetooth, wait a "
                            "few seconds after pairing completes, then try again."
                        ),
                    }
                return

            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            name = joystick.get_name()
            logger.info(f"Gamepad debug probe connected: {name}")

            while not self._stop_event.is_set():
                try:
                    pygame.event.pump()
                    axes = [round(joystick.get_axis(i), 3) for i in range(joystick.get_numaxes())]
                    buttons = [bool(joystick.get_button(i)) for i in range(joystick.get_numbuttons())]
                    hats = [list(joystick.get_hat(i)) for i in range(joystick.get_numhats())]
                except Exception as e:
                    # Controller likely unplugged mid-read.
                    logger.warning(f"Gamepad debug probe read failed, stopping poll: {e}")
                    with self._lock:
                        self._latest = {"connected": False, "message": str(e)}
                    return

                with self._lock:
                    self._latest = {
                        "connected": True,
                        "name": name,
                        "num_axes": len(axes),
                        "num_buttons": len(buttons),
                        "num_hats": len(hats),
                        "axes": axes,
                        "buttons": buttons,
                        "hats": hats,
                    }

                # Auto-release if nobody's actually polling the HTTP endpoint
                # anymore (e.g. the browser tab was closed without the modal's
                # cleanup running), so the joystick handle doesn't stay open
                # forever and block a real teleop session from claiming it.
                if time.time() - self._last_read_request > _IDLE_TIMEOUT_S:
                    logger.info("Gamepad debug probe idle -- releasing handle.")
                    return

                self._stop_event.wait(_POLL_INTERVAL_S)
        finally:
            try:
                pygame.joystick.quit()
            except Exception:
                pass
            try:
                pygame.quit()
            except Exception:
                pass

    def read(self) -> dict[str, Any]:
        """Return the most recent reading, starting the poll thread if needed."""
        with self._lock:
            self._last_read_request = time.time()
            thread_alive = self._poll_thread is not None and self._poll_thread.is_alive()
            if not thread_alive:
                self._latest = {"connected": False, "message": "No gamepad detected."}
                self._start_poll_thread_locked()
            return dict(self._latest)

    def close(self) -> None:
        with self._lock:
            self._stop_event.set()
            thread = self._poll_thread
        if thread is not None:
            thread.join(timeout=2.0)
        with self._lock:
            self._poll_thread = None


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
        # A session is live: stop our own poll thread (if one happened to be
        # running from before the session started) so it isn't fighting the
        # session's teleop thread over the same joystick handle.
        gamepad_probe.close()

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

    return {**gamepad_probe.read(), "in_session": False}
