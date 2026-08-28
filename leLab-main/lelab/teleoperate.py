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

import logging
import math
import threading
import time
from typing import Any

from pydantic import BaseModel

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from .gamepad_teleop import GamepadSO101Teleop, GamepadSO101TeleopConfig
from .utils.config import setup_calibration_files
from .utils.devices import safe_disconnect_device

logger = logging.getLogger(__name__)

# Global variables for teleoperation state
teleoperation_active = False
teleoperation_thread: threading.Thread | None = None
current_robot = None
current_teleop = None
# Guards the start path; the worker owns disconnect so stop() does not race.
_state_lock = threading.Lock()


class TeleoperateRequest(BaseModel):
    input_mode: str = "leader"  # "leader" or "gamepad"
    leader_port: str = ""
    follower_port: str
    leader_config: str = ""
    follower_config: str


def get_joint_positions_from_robot(robot) -> dict[str, float]:
    """
    Extract current joint positions from the robot and convert to URDF joint format.

    lerobot drives the SO-101 with ``use_degrees=True`` by default, so each
    ``observation["<motor>.pos"]`` is already the joint angle in degrees relative
    to the calibration center — which is also the URDF's zero pose. The URDF
    joint value is therefore just that angle converted to radians, for every
    joint. (The gripper reports 0–100 rather than degrees, but that range lands
    inside the Jaw limit, matching the open/closed sweep.)

    Args:
        robot: The robot instance (SO101Follower)

    Returns:
        Dictionary mapping URDF joint names to radian values
    """
    motor_to_urdf_mapping = {
        "shoulder_pan": "Rotation",
        "shoulder_lift": "Pitch",
        "elbow_flex": "Elbow",
        "wrist_flex": "Wrist_Pitch",
        "wrist_roll": "Wrist_Roll",
        "gripper": "Jaw",
    }

    try:
        observation = robot.get_observation()

        joint_positions: dict[str, float] = {}
        debug_rows = []
        for motor_name, urdf_joint_name in motor_to_urdf_mapping.items():
            motor_key = f"{motor_name}.pos"
            if motor_key not in observation:
                logger.warning(f"Motor {motor_key} not found in observation")
                joint_positions[urdf_joint_name] = 0.0
                continue

            angle_degrees = observation[motor_key]
            joint_positions[urdf_joint_name] = angle_degrees * math.pi / 180.0
            debug_rows.append(f"{motor_name:14s} {angle_degrees:+8.2f}° → {urdf_joint_name:11s}")

        # Throttled debug print (~once per second at 20 Hz broadcast).
        now = time.time()
        if now - getattr(get_joint_positions_from_robot, "_last_log", 0) > 1.0:
            get_joint_positions_from_robot._last_log = now
            logger.info("[joint-debug]\n  " + "\n  ".join(debug_rows))

        return joint_positions

    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return dict.fromkeys(motor_to_urdf_mapping.values(), 0.0)


def _safe_disconnect(device) -> None:
    """Disconnect a robot/teleop device, swallowing (but logging) any error.

    Used on the connection-failure cleanup path so one device's failure can't
    leave the other holding its serial port open.
    """
    safe_disconnect_device(device, logger)


_GAMEPAD_CONNECT_TIMEOUT_S = 10.0


def handle_start_teleoperation(request: TeleoperateRequest, websocket_manager=None) -> dict[str, Any]:
    """Handle start teleoperation request.

    Connects to both arms *synchronously* so that a connection failure (arm
    unplugged, port busy, power off) is reported back to the caller, rather than
    dying silently in the worker thread while the API has already claimed
    success.

    In gamepad mode, the gamepad's connect() and every later get_action() call
    must run on the *same* thread: pygame's SDL joystick backend is thread-
    affine on Windows, and reading a Joystick from a different thread than the
    one that opened it silently returns a frozen snapshot instead of live
    input (this is exactly the bug that made the arm never move even though a
    session started successfully). So for gamepad mode, connect() happens
    inside the worker thread itself -- the request thread blocks on a
    threading.Event until that connect step reports success or failure,
    keeping the synchronous-error-reporting contract without ever touching
    pygame from two different threads.
    """
    global teleoperation_active, teleoperation_thread, current_robot, current_teleop

    from . import record as _record, rollout as _rollout

    with _state_lock:
        if teleoperation_active:
            return {"success": False, "message": "Teleoperation is already active"}
        if _record.recording_active:
            return {"success": False, "message": "Recording is currently active. Stop it first."}
        if _rollout.inference_active:
            return {"success": False, "message": "Inference is currently active. Stop it first."}
        teleoperation_active = True

    gamepad_mode = request.input_mode == "gamepad"

    robot = None
    teleop_device = None
    try:
        logger.info(
            f"Starting teleoperation (input_mode={request.input_mode}) with "
            f"leader port: {request.leader_port}, follower port: {request.follower_port}"
        )

        # Setup calibration files. Gamepad mode has no leader arm -- ignore
        # request.leader_config even if it's non-empty (e.g. a robot that was
        # leader-configured before being switched to gamepad mode still has a
        # stale leader_config on its record), so a leader calibration file
        # that was never created for this robot can't block startup.
        leader_config_name, follower_config_name = setup_calibration_files(
            "" if gamepad_mode else request.leader_config, request.follower_config
        )

        # Create robot config
        robot_config = SO101FollowerConfig(
            port=request.follower_port,
            id=follower_config_name,
        )

        logger.info("Initializing robot and teleop device...")
        robot = SO101Follower(robot_config)
        teleop_device = GamepadSO101Teleop(GamepadSO101TeleopConfig()) if gamepad_mode else SO101Leader(
            SO101LeaderConfig(port=request.leader_port, id=leader_config_name)
        )

        # Connect the follower first so the error names which device failed
        # instead of a generic "failed to start".
        logger.info("Connecting to follower arm...")
        try:
            robot.bus.connect()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the follower arm on {request.follower_port}. "
                "Make sure it's plugged in and powered on, then try again."
            ) from e

        if not gamepad_mode:
            logger.info("Connecting to leader arm...")
            try:
                teleop_device.bus.connect()
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to the leader arm on {request.leader_port}. "
                    "Make sure it's plugged in and powered on, then try again."
                ) from e

            # Write calibration to motors' memory
            logger.info("Writing calibration to motors...")
            robot.bus.write_calibration(robot.calibration)
            teleop_device.bus.write_calibration(teleop_device.calibration)

            # Connect cameras and configure motors
            logger.info("Connecting cameras and configuring motors...")
            for cam in robot.cameras.values():
                cam.connect()
            robot.configure()
            teleop_device.configure()
            logger.info("Successfully connected to both devices")
        else:
            # Robot-side calibration/camera/configure still happens here on
            # the request thread -- none of it touches pygame.
            logger.info("Writing calibration to motors...")
            robot.bus.write_calibration(robot.calibration)
            logger.info("Connecting cameras and configuring motors...")
            for cam in robot.cameras.values():
                cam.connect()
            robot.configure()

        current_robot = robot
        current_teleop = teleop_device

        # Gamepad connect (and seed) happens inside the worker thread; the
        # request thread waits on this event for that step's outcome before
        # responding, so a bad/missing controller is still reported
        # synchronously to the caller.
        gamepad_ready = threading.Event()
        gamepad_connect_error: list[str] = []

        # Stream the arms in the background; the worker owns disconnect so stop()
        # does not race the serial bus from the request thread.
        def teleoperation_worker():
            global teleoperation_active, current_robot, current_teleop

            if gamepad_mode:
                try:
                    logger.info("Connecting to gamepad...")
                    teleop_device.connect()
                    teleop_device.configure()
                    # The gamepad has no position of its own; seed its target
                    # from wherever the follower currently is so the first
                    # tick doesn't jump it.
                    teleop_device.seed(robot)
                    logger.info("Gamepad connected.")
                except Exception as e:
                    gamepad_connect_error.append(f"Could not connect to a gamepad: {e}")
                    gamepad_ready.set()
                    _safe_disconnect(robot)
                    _safe_disconnect(teleop_device)
                    global_state_reset()
                    return
                gamepad_ready.set()

            logger.info("Starting teleoperation loop...")
            try:
                last_broadcast_time = 0
                broadcast_interval = 0.05  # 20 FPS
                last_gamepad_debug_log = 0.0

                while teleoperation_active:
                    action = teleop_device.get_action()
                    sent = robot.send_action(action)

                    if gamepad_mode:
                        now_dbg = time.time()
                        if now_dbg - last_gamepad_debug_log > 1.0:
                            last_gamepad_debug_log = now_dbg
                            js = teleop_device._joystick
                            axes = [round(js.get_axis(i), 3) for i in range(js.get_numaxes())] if js else None
                            logger.info(
                                "[gamepad-debug] running=%s axes=%s action=%s sent=%s",
                                teleop_device._running,
                                axes,
                                action,
                                sent,
                            )

                    if gamepad_mode and teleop_device.get_teleop_events()["quit_requested"]:
                        logger.info("Gamepad Circle pressed -- stopping teleoperation.")
                        teleoperation_active = False
                        break

                    current_time = time.time()
                    if current_time - last_broadcast_time >= broadcast_interval:
                        try:
                            joint_positions = get_joint_positions_from_robot(robot)
                            joint_data = {
                                "type": "joint_update",
                                "joints": joint_positions,
                                "timestamp": current_time,
                            }
                            if websocket_manager and websocket_manager.active_connections:
                                websocket_manager.broadcast_joint_data_sync(joint_data)
                            last_broadcast_time = current_time
                        except Exception as e:
                            logger.error(f"Error broadcasting joint data: {e}")

                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error during teleoperation loop: {e}")
            finally:
                _safe_disconnect(robot)
                _safe_disconnect(teleop_device)
                logger.info("Teleoperation stopped")
                global_state_reset()

        def global_state_reset():
            global teleoperation_active, current_robot, current_teleop
            teleoperation_active = False
            current_robot = None
            current_teleop = None

        teleoperation_thread = threading.Thread(
            target=teleoperation_worker, name="teleoperation-worker", daemon=True
        )
        teleoperation_thread.start()

        if gamepad_mode:
            if not gamepad_ready.wait(timeout=_GAMEPAD_CONNECT_TIMEOUT_S):
                # Worker never reported back -- treat as a failure rather than
                # hang the request indefinitely.
                teleoperation_active = False
                return {"success": False, "message": "Timed out connecting to the gamepad."}
            if gamepad_connect_error:
                return {"success": False, "message": gamepad_connect_error[0]}

        return {
            "success": True,
            "message": "Teleoperation started successfully",
            "leader_port": request.leader_port,
            "follower_port": request.follower_port,
        }

    except Exception as e:
        # Connection (or setup) failed before the loop started: release any
        # device that did open, reset state, and surface the error.
        _safe_disconnect(robot)
        _safe_disconnect(teleop_device)
        teleoperation_active = False
        current_robot = None
        current_teleop = None
        logger.error(f"Failed to start teleoperation: {e}")
        # str(e) is already a user-facing message for the connection failures
        # raised above; the toast title supplies the "error starting" context.
        return {"success": False, "message": str(e)}


def handle_stop_teleoperation() -> dict[str, Any]:
    """Handle stop teleoperation request.

    Signals the worker via `teleoperation_active = False` and waits for it to
    exit. The worker owns the disconnect call, so this avoids racing the
    serial bus from the request thread.
    """
    global teleoperation_active, teleoperation_thread

    if not teleoperation_active:
        return {"success": False, "message": "No teleoperation session is active"}

    logger.info("Stop teleoperation triggered from web interface")
    teleoperation_active = False

    worker = teleoperation_thread
    if worker is not None and worker.is_alive():
        worker.join(timeout=5.0)
        if worker.is_alive():
            logger.warning("Teleoperation worker did not exit within 5s")
    teleoperation_thread = None

    return {"success": True, "message": "Teleoperation stopped successfully"}


def handle_teleoperation_status() -> dict[str, Any]:
    """Handle teleoperation status request"""
    status: dict[str, Any] = {
        "teleoperation_active": teleoperation_active,
        "available_controls": {
            "stop_teleoperation": teleoperation_active,
        },
        "message": "Teleoperation status retrieved successfully",
    }

    if teleoperation_active and isinstance(current_teleop, GamepadSO101Teleop):
        gamepad_name = None
        if current_teleop._joystick is not None:
            try:
                gamepad_name = current_teleop._joystick.get_name()
            except Exception:
                gamepad_name = None
        events = current_teleop.get_teleop_events()
        status["gamepad"] = {
            "connected": current_teleop.is_connected,
            "name": gamepad_name,
            "running": events["running"],
        }

    return status


def handle_get_joint_positions() -> dict[str, Any]:
    """Handle get current robot joint positions request"""
    global current_robot

    if not teleoperation_active or current_robot is None:
        return {"success": False, "message": "No active teleoperation session"}

    try:
        joint_positions = get_joint_positions_from_robot(current_robot)
        return {"success": True, "joint_positions": joint_positions, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return {"success": False, "message": f"Failed to get joint positions: {str(e)}"}
