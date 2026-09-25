"""Device cleanup helpers for LeRobot hardware wrappers.

Serial ports are a trust boundary on Windows: if a normal disconnect fails
while disabling torque, the COM handle can stay open until the Python process
exits. These helpers preserve LeRobot's normal disconnect behavior, then force
close the underlying port/cameras as a last resort.
"""

from __future__ import annotations

import logging
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
        except Exception:
            failed.append(name)
    if failed:
        logger.warning("Torque release not acknowledged cleanly by: %s", ", ".join(failed))
    return failed


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
                raise MotorFaultError(
                    "A motor on the arm isn't responding normally -- it most likely tripped its overload "
                    "protection (the arm was pushed, stalled, or jerked hard). Wait about 10 seconds, or switch "
                    "the arm's power off and on, then try again."
                ) from exc
            logger.warning(
                "Motor handshake failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt,
                FAULT_RETRY_ATTEMPTS,
                FAULT_RETRY_WAIT_S,
                exc,
            )
            time.sleep(FAULT_RETRY_WAIT_S)


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
