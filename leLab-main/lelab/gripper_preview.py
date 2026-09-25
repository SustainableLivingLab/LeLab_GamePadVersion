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

"""Standalone real-arm gripper preview, for K12 Lesson 4.1's gesture trainer.

Lets a student move just the gripper on an already-calibrated SO-101 arm on
demand, independent of a full teleoperation session -- handle_start_teleoperation's
gamepad mode requires a physical controller to even connect (see
teleoperate.py), which shouldn't be a requirement just to test a trained
gesture model against the real claw. This mirrors that same file's
follower-arm connect sequence (bus.connect() -> write_calibration -> configure()),
just for the gripper motor alone, via the SAME action-dict path
(``{"gripper.pos": value}``) GamepadSO101Teleop already drives it through.
"""

import logging
import threading
from typing import Any

from pydantic import BaseModel

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from .utils.config import setup_calibration_files
from .utils.devices import (
    MotorFaultError,
    connect_bus_with_fault_recovery,
    safe_disconnect_device,
    sync_goal_to_present,
)

logger = logging.getLogger(__name__)

# Mirrors teleoperate.py's own module-level state pattern.
gripper_preview_active = False
current_robot = None
_state_lock = threading.Lock()


class StartGripperPreviewRequest(BaseModel):
    follower_port: str
    follower_config: str


class GripperPreviewPositionRequest(BaseModel):
    value: float  # 0-100, same scale as GripperOverrideRequest


def handle_start_gripper_preview(request: StartGripperPreviewRequest) -> dict[str, Any]:
    global gripper_preview_active, current_robot

    # Imported lazily (matches handle_start_teleoperation's own pattern) to
    # avoid a module-import cycle -- these all import from server.py's
    # request models indirectly, and this module is imported by teleoperate.py
    # too (see its own start-guard below).
    from . import calibrate as _calibrate
    from . import record as _record
    from . import rollout as _rollout
    from . import teleoperate as _teleoperate

    with _state_lock:
        if gripper_preview_active:
            return {"success": False, "message": "Gripper preview is already active"}
        if _teleoperate.teleoperation_active:
            return {"success": False, "message": "Teleoperation is currently active. Stop it first."}
        if _calibrate.calibration_manager.status.calibration_active:
            return {"success": False, "message": "Calibration is currently active. Stop it first."}
        if _record.recording_active:
            return {"success": False, "message": "Recording is currently active. Stop it first."}
        if _rollout.inference_active:
            return {"success": False, "message": "Inference is currently active. Stop it first."}
        gripper_preview_active = True

    robot = None
    try:
        _, follower_config_name = setup_calibration_files("", request.follower_config)
        robot_config = SO101FollowerConfig(port=request.follower_port, id=follower_config_name)
        robot = SO101Follower(robot_config)

        try:
            connect_bus_with_fault_recovery(robot.bus, logger)
        except MotorFaultError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the arm on {request.follower_port}. "
                "Make sure it's plugged in and powered on, then try again."
            ) from e

        robot.bus.write_calibration(robot.calibration)
        sync_goal_to_present(robot.bus, logger)
        robot.configure()

        current_robot = robot
        return {"success": True, "message": "Gripper preview connected"}
    except Exception as e:
        logger.error(f"Error starting gripper preview: {e}")
        safe_disconnect_device(robot, logger)
        gripper_preview_active = False
        current_robot = None
        return {"success": False, "message": str(e)}


def handle_set_gripper_preview_position(request: GripperPreviewPositionRequest) -> dict[str, Any]:
    if not gripper_preview_active or current_robot is None:
        return {"success": False, "message": "Gripper preview is not active"}
    try:
        value = max(0.0, min(100.0, request.value))
        current_robot.send_action({"gripper.pos": value})
        return {"success": True}
    except Exception as e:
        logger.error(f"Error setting gripper preview position: {e}")
        return {"success": False, "message": str(e)}


def handle_stop_gripper_preview() -> dict[str, Any]:
    global gripper_preview_active, current_robot
    with _state_lock:
        if not gripper_preview_active:
            return {"success": True, "message": "Gripper preview was not active"}
        safe_disconnect_device(current_robot, logger)
        gripper_preview_active = False
        current_robot = None
        return {"success": True, "message": "Gripper preview stopped"}
