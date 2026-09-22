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

"""
Calibration module for the web interface.

This module provides calibration functionality similar to the CLI calibrate.py,
but adapted for the web interface with step-by-step guidance.

Motors are calibrated ONE AT A TIME, in the device's own bus order (which for
every robot/teleop this app supports is already the physical bottom-up chain:
shoulder_pan -> shoulder_lift -> elbow_flex -> wrist_flex -> wrist_roll ->
gripper for the SO-101, just "gripper" alone for the K12 claw rig). This
replaces an earlier all-at-once design where the student had to manually
center EVERY joint simultaneously before a single shared homing step, and a
discontinuity on any one motor aborted the whole calibration, losing every
other joint's already-recorded range. Per-motor homing only ever asks for ONE
joint to be centered at a time, and a failure on one motor (see
MotorCalibrationRetryable) offers a retry scoped to just that motor via
retry_current_step(), without touching motors already completed.
"""

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Literal

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots import Robot
from lerobot.teleoperators import (
    Teleoperator,
    make_teleoperator_from_config,
)
from lerobot.utils.utils import init_logging

from .utils.devices import safe_disconnect_device

logger = logging.getLogger(__name__)

# Feetech sts3215 Present_Position readings are 12-bit (0-4095). A reading at or
# below 0, or at/above this loose ceiling, is a bad frame (disconnected motor or
# an encoder wrap-around) rather than a real joint position, so it's filtered out
# everywhere a raw position is consumed.
_MIN_VALID_POSITION = 0
_MAX_VALID_POSITION = 5000

# A single-frame jump larger than this is the encoder wrapping past 0/4095
# (a ~4096-step delta), not real motion — see MotorCalibrationRetryable.
_MAX_POSITION_JUMP = 2000

# A recorded min..max sweep smaller than this many encoder steps means the joint
# barely moved; below this the motor is sent back through its own retry loop
# instead of being accepted.
_MIN_CALIBRATION_RANGE = 100

# wrist_roll has real mechanical stops (it's NOT free-spinning), but on at
# least some SO-101 builds those stops are further apart than one 4096-step
# encoder turn -- so a genuine sweep to both physical limits can cross the
# encoder's 0/4095 wrap boundary as part of completely normal motion. This
# matters because lerobot's own MotorsBus._normalize/_unnormalize (see
# motors_bus.py) has no concept of "which turn" a raw Present_Position
# reading is in -- it's a straight linear map against calibration's
# range_min/range_max, with no persistent turn-count tracked anywhere during
# ordinary teleoperation. So even if this module recorded a technically-true
# range wider than one turn, that calibration would alias/misread positions
# unpredictably the moment the joint crossed back over a wrap boundary during
# later real use -- the single-turn representation is a hard limit of the
# underlying hardware/software, not something fixable by recording more
# carefully. The practical, honest fix: still sweep this motor for real (see
# _step_range_recording_one), but once a reading looks like it wrapped, stop
# extending the tracked min/max instead of raising a fault -- the recorded
# range naturally settles on the largest single-turn window reachable from
# wherever it was centered, which is the most this setup can represent
# correctly. (Every other SO-101 motor has a real single-turn mechanical
# range, so a large jump for THEM really does mean bad homing -- only this
# one motor gets the more forgiving treatment.)
_MULTI_TURN_MOTORS = {"wrist_roll"}


def _is_valid_position(pos: float) -> bool:
    """True when a raw Present_Position reading sits within the plausible encoder
    range, filtering out 0/negative/extreme bad frames before they pollute the
    recorded min/max."""
    return _MIN_VALID_POSITION < pos < _MAX_VALID_POSITION


class MotorCalibrationRetryable(Exception):
    """Raised for a per-motor failure (discontinuity or insufficient range)
    that should offer a Retry on just the current motor, rather than aborting
    the whole calibration. Wraps the underlying reason as its message."""


@dataclass
class CalibrationStatus:
    """Status information for calibration process"""

    calibration_active: bool = False
    # "idle", "connecting", "homing", "recording", "motor_error", "completed",
    # "error", "stopping". "homing" and "motor_error" are scoped to
    # current_motor; "motor_error" offers a retry of just that motor.
    status: str = "idle"
    device_type: str | None = None
    error: str | None = None
    message: str = ""
    # Every motor on this device's bus, in bottom-up physical order.
    motor_order: list[str] = field(default_factory=list)
    current_motor: str | None = None
    current_motor_index: int = 0
    completed_motors: list[str] = field(default_factory=list)
    current_positions: dict[str, float] | None = None
    # Keyed by motor name; only the current (and previously completed, left
    # as-is) motors have entries. {motor: {min, max, current}}.
    recorded_ranges: dict[str, dict[str, float]] | None = None


@dataclass
class CalibrationRequest:
    """Request parameters for starting calibration"""

    device_type: Literal["robot", "teleop"]
    port: str
    config_file: str
    robot_name: str | None = None  # When set, write port + config back into the robot record on success
    # Which Robot subclass to build for device_type=="robot". "so101_follower" is the
    # full 6-motor arm (default, unchanged behavior); "claw_follower" is the K12 app's
    # single claw-servo rig (see lelab/claw_follower.py).
    robot_type: str = "so101_follower"


class CalibrationManager:
    """Manages calibration process for the web interface"""

    def __init__(self):
        self.status = CalibrationStatus()
        self.device: Robot | Teleoperator | None = None
        self.calibration_thread: threading.Thread | None = None
        self.stop_calibration = False
        self._status_lock = threading.Lock()
        # Set by complete_step() to advance out of a blocking homing/recording
        # wait; set by retry_current_step() to advance out of a blocking
        # motor_error wait. Two separate events so a stray late click on one
        # button can't be misread as the other while the worker has already
        # moved on.
        self._step_complete = threading.Event()
        self._retry_requested = threading.Event()
        self._recording_active = False
        self._mins: dict[str, float] = {}
        self._maxes: dict[str, float] = {}
        self._homing_offsets: dict[str, float] = {}
        self._current_request: CalibrationRequest | None = None

        # Initialize logging
        init_logging()

    def get_status(self) -> CalibrationStatus:
        """Get current calibration status"""
        with self._status_lock:
            # Live position feed while a step is actively waiting on the
            # student (homing: watching them center the joint; recording:
            # watching them sweep it) -- device reads happen here, off the
            # worker thread, same as before.
            if self.status.status in ("homing", "recording") and self.device and self.device.is_connected:
                positions = self._safe_read_all_positions()
                if positions:
                    self.status.current_positions = positions
                    motor = self.status.current_motor
                    if self.status.status == "recording" and motor and motor in positions:
                        pos = positions[motor]
                        if _is_valid_position(pos):
                            # Mirrors the worker's own _mins/_maxes rather than
                            # tracking min/max independently here -- those are
                            # the single source of truth for what actually
                            # gets saved (and, for a _MULTI_TURN_MOTORS entry
                            # like wrist_roll, already apply the same
                            # wrap-tolerant clamping the worker's own sweep
                            # loop does; duplicating that logic here would
                            # risk the displayed range drifting out of sync
                            # with what's really being recorded). `current`
                            # always follows the latest raw read regardless,
                            # so the live marker keeps moving.
                            if not self.status.recorded_ranges:
                                self.status.recorded_ranges = {}
                            self.status.recorded_ranges[motor] = {
                                "min": self._mins.get(motor, pos),
                                "max": self._maxes.get(motor, pos),
                                "current": pos,
                            }

            return self.status

    def _safe_read_all_positions(self) -> dict[str, float] | None:
        """Best-effort Present_Position read with a couple of quick retries on
        transient port contention (a concurrent request racing this same
        bus) -- used by both the status poller above and the worker's own
        loops below."""
        for attempt in range(2):
            try:
                return self.device.bus.sync_read("Present_Position", normalize=False)
            except Exception as read_error:
                if "Port is in use" in str(read_error) and attempt < 1:
                    time.sleep(0.005)
                    continue
                if "Port is in use" not in str(read_error):
                    logger.debug(f"Failed to read positions: {read_error}")
                return None

    def _update_status(self, **kwargs):
        """Update calibration status thread-safely"""
        with self._status_lock:
            for key, value in kwargs.items():
                if hasattr(self.status, key):
                    setattr(self.status, key, value)

    def start_calibration(self, request: CalibrationRequest) -> dict[str, Any]:
        """Start calibration process"""
        try:
            if self.status.calibration_active:
                return {"success": False, "message": "Calibration already active"}

            from . import gripper_preview as _gripper_preview

            if _gripper_preview.gripper_preview_active:
                return {"success": False, "message": "A gripper preview is currently active. Stop it first."}

            # Reset status and clear any previous calibration data
            self._mins = {}
            self._maxes = {}
            self._homing_offsets = {}

            self._update_status(
                calibration_active=True,
                status="connecting",
                device_type=request.device_type,
                error=None,
                message=f"Starting calibration for {request.device_type}",
                motor_order=[],
                current_motor=None,
                current_motor_index=0,
                completed_motors=[],
                current_positions=None,
                recorded_ranges=None,
            )
            self._current_request = request

            # Start calibration in a separate thread
            self.calibration_thread = threading.Thread(
                target=self._calibration_worker, args=(request,), daemon=True
            )
            self.stop_calibration = False
            self._step_complete.clear()
            self._retry_requested.clear()
            self.calibration_thread.start()

            return {"success": True, "message": "Calibration started"}

        except Exception as e:
            logger.error(f"Error starting calibration: {e}")
            self._update_status(
                calibration_active=False, status="error", error=str(e), message="Failed to start calibration"
            )
            return {"success": False, "message": str(e)}

    def complete_step(self) -> dict[str, Any]:
        """Advance out of the current blocking step: confirms this motor is
        centered (status=="homing") or that its range sweep is done
        (status=="recording")."""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            if self.status.status not in ("homing", "recording"):
                return {"success": False, "message": f"Cannot complete step in status: {self.status.status}"}

            self._recording_active = False
            self._step_complete.set()
            return {"success": True, "message": "Step completed"}

        except Exception as e:
            logger.error(f"Error completing step: {e}")
            return {"success": False, "message": str(e)}

    def retry_current_step(self) -> dict[str, Any]:
        """Retry the motor currently sitting in a "motor_error" state: redoes
        its homing + range recording from scratch, without disturbing any
        other motor's already-recorded calibration."""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            if self.status.status != "motor_error":
                return {"success": False, "message": f"Nothing to retry in status: {self.status.status}"}

            self._retry_requested.set()
            return {"success": True, "message": "Retrying current motor"}

        except Exception as e:
            logger.error(f"Error retrying step: {e}")
            return {"success": False, "message": str(e)}

    def stop_calibration_process(self) -> dict[str, Any]:
        """Stop calibration process"""
        try:
            if not self.status.calibration_active:
                return {"success": False, "message": "No calibration active"}

            logger.info("Stopping calibration process...")
            self.stop_calibration = True
            self._recording_active = False
            self._step_complete.set()  # Unblock any waiting step
            self._retry_requested.set()  # Unblock a waiting retry too

            self._update_status(status="stopping", message="Stopping calibration...")

            # Wait for thread to finish
            if self.calibration_thread and self.calibration_thread.is_alive():
                self.calibration_thread.join(timeout=5.0)

            # Ensure cleanup is called if thread didn't finish properly
            if self.calibration_thread and self.calibration_thread.is_alive():
                logger.warning("Calibration thread did not finish within timeout, forcing cleanup")

            # Force cleanup and finish
            self._cleanup_and_finish("Calibration stopped", status="idle")

            logger.info("Calibration stop completed")
            return {"success": True, "message": "Calibration stopped"}

        except Exception as e:
            logger.error(f"Error stopping calibration: {e}")
            # Force cleanup on error too
            self._cleanup_and_finish("Calibration stopped with error", status="error")
            return {"success": False, "message": str(e)}

    def _calibration_worker(self, request: CalibrationRequest):
        """Worker thread for calibration process"""
        try:
            logger.info(f"Starting calibration worker for {request.device_type}")

            self._update_status(status="connecting", message="Connecting to device...")

            # Create and connect device. Robot construction is dispatched directly
            # (not via make_robot_from_config) so a K12-only robot type like
            # "claw_follower" doesn't need to be registered in vendored lerobot.
            if request.device_type == "robot":
                if request.robot_type == "claw_follower":
                    from .claw_follower import ClawFollower
                    from .config_claw_follower import ClawFollowerConfig

                    config = ClawFollowerConfig(port=request.port, id=request.config_file)
                    self.device = ClawFollower(config)
                else:
                    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

                    config = SO101FollowerConfig(port=request.port, id=request.config_file)
                    self.device = SO101Follower(config)
            elif request.device_type == "teleop":
                from lerobot.teleoperators.so_leader import SO101LeaderConfig

                config = SO101LeaderConfig(port=request.port, id=request.config_file)
                self.device = make_teleoperator_from_config(config)
            else:
                raise ValueError(f"Unknown device type: {request.device_type}")

            logger.info("Connecting to device...")
            self.device.connect(calibrate=False)

            if self.stop_calibration:
                logger.info("Calibration stopped after device connection")
                self._cleanup_and_finish("Calibration cancelled")
                return

            # Bus order is already the physical bottom-up chain for every
            # device this app supports (see this module's own docstring).
            motor_order = list(self.device.bus.motors.keys())
            self._update_status(motor_order=motor_order, completed_motors=[], current_motor_index=0)

            # Disable torque and switch to position mode ONCE up front, same
            # as before -- this doesn't move anything or depend on any
            # motor's own homing, just prepares the bus for manual movement
            # and later Goal_Position writes.
            self.device.bus.disable_torque()
            for motor in self.device.bus.motors:
                self.device.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.device.bus.reset_calibration()

            for index, motor in enumerate(motor_order):
                if self.stop_calibration:
                    self._cleanup_and_finish("Calibration cancelled")
                    return

                self._update_status(current_motor_index=index)
                self._calibrate_one_motor(motor)

                if self.stop_calibration:
                    self._cleanup_and_finish("Calibration cancelled")
                    return

            # Complete calibration
            self._complete_calibration()

            logger.info("Calibration completed successfully")
            self._cleanup_and_finish("Calibration completed successfully", status="completed")

        except Exception as e:
            logger.error(f"Calibration error: {e}")
            logger.error(traceback.format_exc())
            # Ensure cleanup happens even on error
            self._cleanup_and_finish(f"Calibration failed: {e}", status="error")
        finally:
            # Ensure we always clean up and reset the active flag
            logger.info("Calibration worker thread finishing")
            if self.status.calibration_active:
                logger.warning(
                    "Worker thread ending but calibration still marked as active - forcing cleanup"
                )
                self._cleanup_and_finish("Calibration stopped", status="idle")

    def _calibrate_one_motor(self, motor: str):
        """Home + record range for a single motor, looping back to redo both
        (re-prompting the student to re-center it) on a retryable failure,
        until it succeeds or the whole calibration is stopped/cancelled.
        Motors already appended to completed_motors before this call are
        untouched regardless of how many retries this one takes."""
        while True:
            try:
                self._step_homing_one(motor)
                if self.stop_calibration:
                    return

                self._step_range_recording_one(motor)
                if self.stop_calibration:
                    return

                with self._status_lock:
                    if motor not in self.status.completed_motors:
                        self.status.completed_motors = [*self.status.completed_motors, motor]
                return

            except MotorCalibrationRetryable as e:
                if self.stop_calibration:
                    return
                logger.warning(f"Retryable calibration failure on {motor}: {e}")
                self._enter_motor_error(motor, str(e))
                self._wait_for_retry_or_stop()
                if self.stop_calibration:
                    return
                # Loop back around and redo this same motor's homing.

    def _enter_motor_error(self, motor: str, message: str):
        self._retry_requested.clear()
        self._update_status(
            status="motor_error",
            current_motor=motor,
            error=message,
            message=message,
        )

    def _wait_for_retry_or_stop(self):
        while not self._retry_requested.is_set() and not self.stop_calibration:
            time.sleep(0.05)
        self._retry_requested.clear()

    def _step_homing_one(self, motor: str):
        """Blocks until the student confirms this ONE motor is centered
        (via complete_step()), then captures its homing offset from
        whatever position it's actually sitting at right then. Only ever
        asks for one joint at a time -- the earlier all-at-once design
        needed the WHOLE arm centered simultaneously before this step,
        which is what made it hard to get right."""
        logger.info(f"Waiting for {motor} to be centered")
        self._update_status(
            status="homing",
            current_motor=motor,
            message=f"Move {motor.replace('_', ' ')} to the middle of its range of motion, then click Continue.",
        )
        self._recording_active = False
        self._step_complete.clear()

        while not self._step_complete.is_set() and not self.stop_calibration:
            time.sleep(0.05)
        if self.stop_calibration:
            return
        self._step_complete.clear()

        positions = self._safe_read_all_positions() or {}
        pos = positions.get(motor)
        if pos is None or not _is_valid_position(pos):
            # Retry the read a few times before giving up on this attempt --
            # a single bad frame here would otherwise mis-home the motor.
            for _ in range(5):
                time.sleep(0.05)
                positions = self._safe_read_all_positions() or {}
                pos = positions.get(motor)
                if pos is not None and _is_valid_position(pos):
                    break
            if pos is None or not _is_valid_position(pos):
                raise MotorCalibrationRetryable(
                    f"Could not read {motor}'s position to set its center. Check the connection and retry."
                )

        logger.info(f"Homing {motor} from current position {pos}")
        offset = self.device.bus._get_half_turn_homings({motor: pos})[motor]
        self._homing_offsets[motor] = offset
        self.device.bus.write("Homing_Offset", motor, offset)

    def _step_range_recording_one(self, motor: str):
        """Record min/max for a single motor as the student sweeps it
        through its range, blocking until they click Continue. Raises
        MotorCalibrationRetryable (caught by _calibrate_one_motor, which
        re-prompts homing for just this motor) on an insufficient recorded
        range -- and, for a normal (single-turn) motor, on a discontinuity
        too. For a _MULTI_TURN_MOTORS entry, a jump is NOT treated as a
        fault (see that set's own comment): the sample is just skipped, so
        min/max stop extending once the sweep runs past the edge of what's
        safely trackable in one encoder turn, instead of erroring out."""
        multi_turn = motor in _MULTI_TURN_MOTORS
        logger.info(f"Starting range recording for {motor}" + (" (multi-turn-safe)" if multi_turn else ""))

        start_pos = None
        for attempt in range(5):
            positions = self._safe_read_all_positions() or {}
            candidate = positions.get(motor)
            if candidate is not None and _is_valid_position(candidate):
                start_pos = candidate
                break
            logger.warning(f"Attempt {attempt + 1}: invalid initial position for {motor}, retrying...")
            time.sleep(0.1)

        if start_pos is None:
            raise MotorCalibrationRetryable(f"Could not get a valid starting position for {motor}.")

        logger.info(f"Starting position for {motor}: {start_pos}")
        self._mins[motor] = start_pos
        self._maxes[motor] = start_pos

        with self._status_lock:
            recorded_ranges = dict(self.status.recorded_ranges or {})
            recorded_ranges[motor] = {"min": start_pos, "max": start_pos, "current": start_pos}
            self.status.recorded_ranges = recorded_ranges

        message = f"Move {motor.replace('_', ' ')} through its FULL range of motion, then click Continue."
        if multi_turn:
            message += (
                " If the range stops growing near one end before you reach the physical stop, that's the "
                "sensor's own limit, not a fault -- click Continue whenever you're satisfied."
            )
        self._update_status(status="recording", current_motor=motor, message=message)

        self._recording_active = True
        self._step_complete.clear()
        prev_pos = start_pos

        while not self._step_complete.is_set() and not self.stop_calibration:
            positions = self._safe_read_all_positions()
            if positions:
                pos = positions.get(motor)
                if pos is not None and _is_valid_position(pos):
                    if abs(pos - prev_pos) > _MAX_POSITION_JUMP:
                        if multi_turn:
                            logger.debug(
                                f"{motor}: ignoring apparent wrap (prev={prev_pos}, new={pos}) -- past the "
                                "edge of what's safely trackable in one turn."
                            )
                        else:
                            raise MotorCalibrationRetryable(
                                f"{motor.replace('_', ' ')} jumped too far in one step (an encoder "
                                "wrap-around) -- it likely wasn't centered closely enough. Try again, moving "
                                "it to the middle of its range before continuing."
                            )
                    else:
                        prev_pos = pos
                        self._mins[motor] = min(self._mins[motor], pos)
                        self._maxes[motor] = max(self._maxes[motor], pos)

            time.sleep(0.05)

        if self.stop_calibration:
            return
        self._step_complete.clear()

        range_diff = self._maxes[motor] - self._mins[motor]
        logger.info(f"Recorded range for {motor}: min={self._mins[motor]}, max={self._maxes[motor]}, range={range_diff}")

        if range_diff < _MIN_CALIBRATION_RANGE:
            raise MotorCalibrationRetryable(
                f"{motor.replace('_', ' ')} only moved {range_diff} steps -- move it through more of its "
                "range before continuing."
            )

    def _complete_calibration(self):
        """Complete the calibration and save results"""
        logger.info("Completing calibration...")

        # Log motor information for debugging
        logger.info("Motor configuration:")
        for motor, m in self.device.bus.motors.items():
            logger.info(f"  {motor}: ID={m.id}, Model={m.model}")

        # Create calibration dict
        calibration = {}
        for motor, m in self.device.bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=self._homing_offsets[motor],
                range_min=self._mins[motor],
                range_max=self._maxes[motor],
            )
            logger.info(
                f"Calibration for {motor}: "
                f"ID={m.id}, "
                f"homing_offset={self._homing_offsets[motor]}, "
                f"range_min={self._mins[motor]}, "
                f"range_max={self._maxes[motor]}"
            )

        # Write and save calibration
        self.device.calibration = calibration
        self.device.bus.write_calibration(calibration)
        self.device._save_calibration()

        logger.info(f"Calibration saved to {self.device.calibration_fpath}")

        # Robot-record write-back: if this calibration was launched from a tile,
        # update the robot's port + config field for the side that was just calibrated.
        request = self._current_request
        if request is not None and request.robot_name:
            from .utils.config import save_robot_record

            if request.device_type == "teleop":
                patch = {"leader_port": request.port, "leader_config": f"{request.config_file}.json"}
            else:
                patch = {"follower_port": request.port, "follower_config": f"{request.config_file}.json"}
            try:
                save_robot_record(request.robot_name, patch, allow_create=False)
            except Exception as e:
                logger.warning(f"Robot-record write-back failed for {request.robot_name}: {e}")

    def _cleanup_and_finish(self, message: str, status: str = "completed"):
        """Clean up and finish calibration"""
        self._cleanup_device()
        self._recording_active = False
        self._update_status(calibration_active=False, status=status, message=message)

    def _cleanup_device(self):
        """Clean up device connection.

        Uses safe_disconnect_device so a failed disconnect (flaky USB/serial)
        force-releases the port/cameras instead of leaving the device busy and
        blocking the next calibration/teleop/record run.
        """
        if self.device:
            logger.info("Disconnecting device...")
            safe_disconnect_device(self.device, logger, context="calibration cleanup")
            self.device = None


# Global calibration manager instance
calibration_manager = CalibrationManager()
