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

"""One-time setup that moves a standalone claw's servo onto its own ID.

The standalone claw's servo used to share ID 6 with the arm's gripper, so a
claw session could connect to a whole arm (and drive its gripper with the
claw's calibration) whenever the wrong servo cable was on the board. Claws
now use CLAW_MOTOR_ID (7) instead; this moves an existing servo -- on 6, or
factory-fresh on 1 -- to it, using lerobot's own setup_motor() (which also
finds the servo's baud rate and sets the bus default).

Only ever touches a board where exactly ONE servo answers, so it can't
re-ID a motor inside an arm by mistake.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import suppress
from typing import Any

from pydantic import BaseModel

from .claw_follower import CLAW_MOTOR_ID
from .utils.config import FOLLOWER_CONFIG_PATH, list_robot_records

logger = logging.getLogger(__name__)

# IDs worth looking for at the default baud rate: the arm's 1-6 plus the
# claw's own -- enough to spot "more than one servo on this board".
_SCAN_IDS = (1, 2, 3, 4, 5, 6, CLAW_MOTOR_ID)


class ClawServoSetupRequest(BaseModel):
    port: str


def _update_claw_calibration_ids(port: str) -> list[str]:
    """Point claw robots' calibration files on `port` at the new ID.

    The homing offset and limits live in the servo itself and survive the ID
    change, so the calibration is still valid -- only its recorded ID is
    stale. Returns the files updated."""
    updated = []
    for record in list_robot_records():
        if record.get("robot_type") != "claw_follower" or record.get("follower_port") != port:
            continue
        path = os.path.join(FOLLOWER_CONFIG_PATH, record.get("follower_config") or "")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("gripper"), dict) and data["gripper"].get("id") != CLAW_MOTOR_ID:
                data["gripper"]["id"] = CLAW_MOTOR_ID
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                updated.append(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not update claw calibration %s: %s", path, exc)
    return updated


def _board_busy() -> bool:
    # Lazy, like the other start handlers: avoids import cycles.
    from . import calibrate as _calibrate
    from . import gripper_preview as _gripper_preview
    from . import record as _record
    from . import rollout as _rollout
    from . import teleoperate as _teleoperate

    return (
        _teleoperate.teleoperation_active
        or _calibrate.calibration_manager.status.calibration_active
        or _record.recording_active
        or _rollout.inference_active
        or _gripper_preview.gripper_preview_active
    )


def handle_claw_servo_status(request: ClawServoSetupRequest) -> dict[str, Any]:
    """Read-only: which servo IDs answer on `port`, and whether that's a claw
    already on its own ID. Lets the claw calibration screen skip the setup
    step when it's already been done (on any computer -- the ID lives in the
    servo)."""
    if _board_busy():
        return {"success": False, "state": "busy", "message": "Something is using the arm/claw right now. Stop it first."}

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
        port=request.port,
        motors={"gripper": Motor(CLAW_MOTOR_ID, "sts3215", MotorNormMode.RANGE_0_100)},
    )
    try:
        bus.connect(handshake=False)
        found = [motor_id for motor_id in _SCAN_IDS if bus.ping(motor_id, num_retry=2) is not None]
    except Exception as exc:
        return {"success": False, "state": "error", "message": f"Could not open {request.port}: {exc}"}
    finally:
        with suppress(Exception):
            bus.port_handler.is_using = False
            bus.port_handler.closePort()

    if found == [CLAW_MOTOR_ID]:
        state, message = "ready", f"Claw servo is set up (ID {CLAW_MOTOR_ID})."
    elif len(found) > 1:
        state = "wrong_device"
        message = (
            f"{len(found)} servos answer on {request.port} -- this looks like the arm, not a standalone claw. "
            "Plug in only the claw's servo cable."
        )
    elif found:
        state, message = "needs_setup", f"This claw's servo is on ID {found[0]} and needs its one-time setup."
    else:
        # Nothing at the default baud rate: a factory servo on another baud
        # rate, or nothing plugged in -- setup will find out.
        state, message = "needs_setup", "No servo answered at the usual settings -- check it's plugged in and powered."
    return {"success": True, "state": state, "found_ids": found, "message": message}


def handle_setup_claw_servo(request: ClawServoSetupRequest) -> dict[str, Any]:
    if _board_busy():
        return {"success": False, "message": "Something is using the arm/claw right now. Stop it first."}

    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
        port=request.port,
        motors={"gripper": Motor(CLAW_MOTOR_ID, "sts3215", MotorNormMode.RANGE_0_100)},
    )
    try:
        try:
            bus.connect(handshake=False)
        except Exception as exc:
            return {"success": False, "message": f"Could not open {request.port}: {exc}"}

        found = [motor_id for motor_id in _SCAN_IDS if bus.ping(motor_id, num_retry=2) is not None]
        if found == [CLAW_MOTOR_ID]:
            return {"success": True, "message": f"This claw servo is already set up (ID {CLAW_MOTOR_ID})."}
        if len(found) > 1:
            return {
                "success": False,
                "message": (
                    f"{len(found)} servos answer on {request.port} (IDs {', '.join(map(str, found))}) -- this "
                    "looks like the arm. Plug in ONLY the standalone claw's servo cable, then try again."
                ),
            }

        # Exactly one servo (found at the default baud rate), or none found
        # there -- setup_motor() scans every baud rate itself in that case,
        # and refuses if it finds more than one.
        bus.setup_motor("gripper", initial_id=found[0] if found else None)
        if bus.ping(CLAW_MOTOR_ID, num_retry=2) is None:
            return {"success": False, "message": "The servo didn't answer on its new ID. Power-cycle it and try again."}
    except Exception as exc:
        logger.error("Claw servo setup failed: %s", exc)
        return {
            "success": False,
            "message": f"Couldn't set up the claw servo: {exc}. Check only the claw is plugged in and powered.",
        }
    finally:
        with suppress(Exception):
            bus.port_handler.is_using = False
            bus.port_handler.closePort()

    updated = _update_claw_calibration_ids(request.port)
    old = f"ID {found[0]}" if found else "its old ID"
    logger.info("Claw servo on %s moved from %s to ID %d; updated %s", request.port, old, CLAW_MOTOR_ID, updated)
    return {
        "success": True,
        "message": f"Claw servo set up: moved from {old} to ID {CLAW_MOTOR_ID}. You only need to do this once.",
    }
