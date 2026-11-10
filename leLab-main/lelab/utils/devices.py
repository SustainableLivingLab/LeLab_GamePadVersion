"""Device cleanup helpers for LeRobot hardware wrappers.

Serial ports are a trust boundary on Windows: if a normal disconnect fails
while disabling torque, the COM handle can stay open until the Python process
exits. These helpers preserve LeRobot's normal disconnect behavior, then force
close the underlying port/cameras as a last resort.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from contextlib import suppress
from typing import Any

from lerobot.motors.motors_bus import get_address

# A Feetech servo that trips its overload protection (stalled against the
# table / its own hard stop, or yanked by a sudden big goal jump) cuts its own
# torque and then flags EVERY status packet it sends with an overload error
# until it recovers a few seconds later. lerobot's connect handshake pings each
# motor and treats a flagged reply as "motor missing", so starting control in
# that window fails with a misleading "could not connect" -- retrying a few
# seconds later works. connect_bus_with_fault_recovery() does that retry itself.
FAULT_RETRY_ATTEMPTS = 5
FAULT_RETRY_WAIT_S = 1.5


class MotorFaultError(RuntimeError):
    """The port opened fine but a motor kept failing the handshake -- almost
    always a servo still in overload protection, not a cabling problem."""


# Feetech status-packet error bit for overload protection (scservo_sdk's
# ERRBIT_OVERLOAD). Protection stays latched until the motors lose power.
OVERLOAD_ERROR_BIT = 32
# Motor IDs seen in overload protection, with when, so the error shown to the
# student can name every one of them, not just the motor lerobot's exception
# happened to stop at. Forgotten after a minute (and once reported).
OVERLOAD_MEMORY_S = 60
_recent_overloads: dict[int, float] = {}


def _note_overload(motor_id: int) -> None:
    _recent_overloads[motor_id] = time.monotonic()


def overload_message(exc: BaseException | None = None) -> str | None:
    """The student-facing message for motors stuck in overload protection,
    naming them (M1-M6 are the motor IDs), or None if nothing points to an
    overload. Reads the exception text and the motors noted by
    release_torque_best_effort() in the last minute."""
    text = str(exc) if exc is not None else ""
    now = time.monotonic()
    ids = {motor_id for motor_id, seen in _recent_overloads.items() if now - seen < OVERLOAD_MEMORY_S}
    if "Overload error" in text:
        ids.update(int(found) for found in re.findall(r"id_=(\d+)", text))
    elif not ids:
        return None
    _recent_overloads.clear()

    names = [f"M{motor_id}" for motor_id in sorted(ids)]
    if not names:
        subject = "A motor is"
    elif len(names) == 1:
        subject = f"{names[0]} is"
    else:
        subject = f"{', '.join(names[:-1])} and {names[-1]} are"
    return f"{subject} overloaded. Unplug the arm's power, wait 15 seconds, and plug it back in."


def describe_device_error(exc: BaseException) -> str:
    """str(exc), unless it comes from motors in overload protection: then
    the plain message saying which ones and how to clear it."""
    return overload_message(exc) or str(exc)


def release_torque_best_effort(bus: Any, logger: logging.Logger) -> list[str]:
    """Turn torque off on every motor, one at a time, ignoring per-motor
    errors -- lerobot's own disable_torque() raises on the first faulted motor
    and leaves every motor after it still holding. Returns the motors that
    didn't acknowledge cleanly.
    """
    failed: list[str] = []
    for name, motor in bus.motors.items():
        try:
            addr, length = get_address(bus.model_ctrl_table, motor.model, "Torque_Enable")
            comm, error = bus._write(addr, length, motor.id, 0, num_retry=1, raise_on_error=False)
            if not bus._is_comm_success(comm) or bus._is_error(error):
                failed.append(name)
            if error & OVERLOAD_ERROR_BIT:
                _note_overload(motor.id)
        except Exception:
            failed.append(name)
    if failed:
        logger.warning("Torque release not acknowledged cleanly by: %s", ", ".join(failed))
    return failed


class WrongDeviceError(RuntimeError):
    """The motors answering on this port don't match the robot that was
    picked -- e.g. the standalone claw is plugged into the board but the arm
    robot was selected, or vice versa."""


# The full arm's motors are IDs 1-6 (base to gripper); the standalone claw's
# single servo is ID 7 (see claw_follower.CLAW_MOTOR_ID). Claws set up before
# that still answer on 6 -- the same ID as the arm's gripper -- and are told
# to run the one-time claw servo setup.
_ARM_IDS = (1, 2, 3, 4, 5, 6)
_OLD_CLAW_ID = 6
_CLAW_ID = 7
_SCAN_IDS = (*_ARM_IDS, _CLAW_ID)


def check_plugged_in_device(bus: Any, expect: str, logger: logging.Logger) -> None:
    """Ping IDs 1-6 on `bus` and refuse to continue if they clearly belong to
    the other device ("arm" or "claw") than `expect`.

    Classrooms swap one driver board between the arm and the standalone claw
    by moving the servo cable, so "which one is plugged in" is a real,
    common mistake. Opens and closes the port itself and writes nothing.
    Only refuses on an unambiguous mismatch: an arm with a motor or two
    tripped into overload (which then doesn't answer) still passes, and the
    normal connect path reports that. A port that won't open raises the
    usual connection error.
    """
    bus.connect(handshake=False)
    try:
        found = [motor_id for motor_id in _SCAN_IDS if bus.ping(motor_id, num_retry=2) is not None]
    finally:
        port_handler = getattr(bus, "port_handler", None)
        with suppress(Exception):
            port_handler.is_using = False
            port_handler.closePort()
    logger.info("Motors answering on %s: %s (expected %s)", getattr(bus, "port", "?"), found or "none", expect)

    arm_body_ids = [motor_id for motor_id in found if motor_id in _ARM_IDS and motor_id != _OLD_CLAW_ID]
    port = getattr(bus, "port", "this port")
    if not found:
        raise WrongDeviceError(
            f"No motors are answering on {port}. Check the {expect}'s servo cable is plugged into the board and "
            "its power supply is on, then try again."
        )
    if expect == "arm":
        if found == [_CLAW_ID]:
            raise WrongDeviceError(
                f"It looks like the standalone claw is plugged into the board on {port}, not the arm. Pick your "
                "claw robot instead, or plug the arm's servo cable back in."
            )
        if found == [_OLD_CLAW_ID]:
            raise WrongDeviceError(
                f"Only motor {_OLD_CLAW_ID} answers on {port} -- that's most likely a standalone claw, not the arm. "
                "Pick your claw robot instead, or plug the arm's servo cable back in."
            )
    if expect == "claw":
        if len(arm_body_ids) >= 3:
            raise WrongDeviceError(
                f"It looks like the whole arm is plugged into the board on {port}, not the standalone claw (motors "
                f"{', '.join(map(str, found))} answer). Pick your arm robot instead, or plug the claw's servo "
                "cable in."
            )
        if _CLAW_ID not in found:
            raise WrongDeviceError(
                f"This claw's servo isn't on its own ID yet (it answers as motor {found[0]}). Open Start "
                "Calibration for this claw -- it walks you through the one-time servo setup first."
            )


def connect_bus_with_fault_recovery(bus: Any, logger: logging.Logger) -> None:
    """bus.connect(), retried while motors are recovering from an overload.

    A port that won't even open (unplugged, wrong COM port) fails immediately
    with the original error. A port that opens but fails the motor handshake
    gets its torque released and is retried a few times, then raises
    MotorFaultError.
    """
    for attempt in range(1, FAULT_RETRY_ATTEMPTS + 1):
        try:
            bus.connect()
            if attempt > 1:
                logger.info("Motor bus connected on attempt %d", attempt)
            return
        except Exception as exc:
            port_handler = getattr(bus, "port_handler", None)
            if not getattr(port_handler, "is_open", False):
                raise
            # Handshake failed with the port open -- lerobot leaves it open in
            # that case, so close it ourselves before the next attempt.
            release_torque_best_effort(bus, logger)
            with suppress(Exception):
                port_handler.is_using = False
                port_handler.closePort()
            if attempt == FAULT_RETRY_ATTEMPTS:
                # Almost always overload protection (the arm was pushed,
                # stalled, or jerked hard), even when no reply said so.
                raise MotorFaultError(
                    overload_message(exc)
                    or "A motor is overloaded. Unplug the arm's power, wait 15 seconds, and plug it back in."
                ) from exc
            logger.warning(
                "Motor handshake failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt,
                FAULT_RETRY_ATTEMPTS,
                FAULT_RETRY_WAIT_S,
                exc,
            )
            time.sleep(FAULT_RETRY_WAIT_S)


# How long a bus read/write keeps retrying after a communication failure
# before giving up (see make_bus_io_resilient).
BUS_IO_RETRY_WINDOW_S = 1.0
BUS_IO_RETRY_PAUSE_S = 0.02


def make_bus_io_resilient(bus: Any, logger: logging.Logger) -> None:
    """Retry the bus's sync_read/sync_write for up to BUS_IO_RETRY_WINDOW_S
    after a communication failure, instead of failing on the first one.

    lerobot reads every motor once per frame with a single attempt, so one
    glitch -- a USB hub hiccup, electrical noise, a brief power dip -- makes
    sync_read raise "There is no status packet!" for all motors at once, and
    in recording that exception ends the whole session mid-episode.
    Teleoperation already survives these by retrying the next tick; this gives
    recording the same tolerance. A failure that outlasts the window (power
    really off, cable out) still raises, so real faults aren't hidden.
    """

    def resilient(name: str):
        original = getattr(bus, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            deadline = time.monotonic() + BUS_IO_RETRY_WINDOW_S
            failures = 0
            while True:
                try:
                    result = original(*args, **kwargs)
                    if failures:
                        logger.warning("Bus %s recovered after %d failed attempt(s)", name, failures)
                    return result
                except ConnectionError:
                    failures += 1
                    if time.monotonic() >= deadline:
                        raise
                    # scservo_sdk can leave the port flagged busy after a
                    # failed transfer, which would reject every retry.
                    port_handler = getattr(bus, "port_handler", None)
                    if port_handler is not None and getattr(port_handler, "is_using", False):
                        port_handler.is_using = False
                    time.sleep(BUS_IO_RETRY_PAUSE_S)

        return call

    bus.sync_read = resilient("sync_read")
    bus.sync_write = resilient("sync_write")


# How long a camera may go without a new picture before a recording gives up
# (lerobot's own limit is 0.5 s, after which it ends the whole session).
CAMERA_STALL_WINDOW_S = 3.0
CAMERA_RETRY_PAUSE_S = 0.02


def make_cameras_resilient(cameras: dict[str, Any], logger: logging.Logger) -> None:
    """Let each camera's read_latest wait out a short stall instead of failing.

    lerobot reads only a camera's newest picture and raises as soon as it is
    older than 0.5 s, which ended a 23-minute recording session on a 506 ms
    hiccup (two cameras on one USB hub, a brief Windows pause, auto exposure
    adjusting). This waits up to CAMERA_STALL_WINDOW_S for a fresh picture,
    and reopens a camera once if its background reader stopped. A camera that
    stays silent past the window still ends the session, with a clear reason.
    """

    for name, camera in cameras.items():
        original = camera.read_latest

        def read_latest(max_age_ms: int = 500, *, _camera=camera, _name=name, _original=original):
            try:
                return _original(max_age_ms)
            except (TimeoutError, RuntimeError) as first_error:
                started = time.monotonic()
                deadline = started + CAMERA_STALL_WINDOW_S
                reopened = False
                last_error: Exception = first_error
                while time.monotonic() < deadline:
                    reader = getattr(_camera, "thread", None)
                    if (reader is None or not reader.is_alive()) and not reopened:
                        # The background reader died (e.g. the USB link
                        # dropped for a moment): reopen the camera once.
                        reopened = True
                        try:
                            with contextlib.suppress(Exception):
                                _camera.disconnect()
                            _camera.connect()
                            logger.warning("Camera %s: reopened after its picture feed stopped", _name)
                        except Exception as exc:  # noqa: BLE001 - reported below if it never recovers
                            last_error = exc
                    time.sleep(CAMERA_RETRY_PAUSE_S)
                    try:
                        frame = _original(max_age_ms)
                    except (TimeoutError, RuntimeError) as exc:
                        last_error = exc
                        continue
                    logger.warning(
                        "Camera %s: no new picture for a moment (%s); continued after %.0f ms",
                        _name,
                        first_error,
                        (time.monotonic() - started) * 1000,
                    )
                    return frame
                raise TimeoutError(
                    f"Camera {_name} stopped sending pictures for {CAMERA_STALL_WINDOW_S:.0f} seconds "
                    f"({last_error}). Check its USB cable, or plug it straight into the computer "
                    "instead of a hub."
                ) from last_error

        camera.read_latest = read_latest


def sync_goal_to_present(bus: Any, logger: logging.Logger) -> None:
    """Set every motor's Goal_Position register to where it physically is now.

    Call after write_calibration() and before configure(): configure() ends by
    re-enabling torque, and a servo with torque on drives straight to whatever
    Goal_Position it still holds -- a leftover from the last session, or, after
    a recalibration changed its homing offset, a completely different spot.
    That's the "jerks to a position by itself" on Start Control, and on the
    shoulder/elbow (carrying the arm's weight) it's what trips overload
    protection. Raw (unnormalized) values on purpose: an arm resting just past
    its calibrated range would otherwise get clipped, i.e. moved.
    """
    present = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    bus.sync_write("Goal_Position", present, normalize=False)
    logger.info("Synced goal positions to present positions before enabling torque")


def safe_disconnect_device(device: Any, logger: logging.Logger, context: str = "cleanup") -> None:
    """Disconnect a LeRobot device and force-release resources on failure."""
    if device is None:
        return

    try:
        device.disconnect()
        return
    except Exception as exc:
        logger.warning("Error disconnecting device during %s: %s", context, exc)

    _force_close_device_resources(device, logger)


def _force_close_device_resources(device: Any, logger: logging.Logger) -> None:
    """Best-effort release for serial/camera resources after disconnect fails."""
    bus = getattr(device, "bus", None)
    port_handler = getattr(bus, "port_handler", None)
    if port_handler is not None:
        with suppress(Exception):
            port_handler.clearPort()
        with suppress(Exception):
            port_handler.is_using = False
        # The normal disconnect stops at the first motor that errors, so the
        # ones after it would stay torqued up after the port closes.
        if getattr(port_handler, "is_open", False) and getattr(bus, "motors", None):
            with suppress(Exception):
                release_torque_best_effort(bus, logger)
        try:
            port_handler.closePort()
            logger.info("Force-closed serial port after disconnect failure")
        except Exception as exc:
            logger.warning("Failed to force-close serial port after disconnect failure: %s", exc)

    cameras = getattr(device, "cameras", None)
    if isinstance(cameras, dict):
        for cam in cameras.values():
            try:
                cam.disconnect()
            except Exception as exc:
                logger.warning("Failed to disconnect camera after device cleanup failure: %s", exc)
