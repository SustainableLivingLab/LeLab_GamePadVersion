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
"""Tests for lelab.calibrate — manager initial state and request schema."""

from __future__ import annotations

import threading
import time

import pytest


@pytest.mark.parametrize(
    "pos,expected",
    [
        (-5, False),  # negative — bad frame
        (0, False),  # lower bound is exclusive (old check: pos > 0)
        (1, True),
        (100, True),
        (2000, True),
        (4095, True),  # encoder max
        (4999, True),
        (5000, False),  # upper bound is exclusive (old check: pos < 5000)
        (6000, False),  # extreme — bad frame
    ],
)
def test_is_valid_position_boundaries(pos, expected) -> None:
    """Pins the plausible-encoder-range filter that replaced three duplicated
    inline `pos > 0 and pos < 5000` checks. Boundaries are exclusive on both ends."""
    from lelab.calibrate import _is_valid_position

    assert _is_valid_position(pos) is expected


def test_calibration_status_defaults_to_idle() -> None:
    from lelab.calibrate import CalibrationStatus

    status = CalibrationStatus()
    assert status.calibration_active is False
    assert status.status == "idle"
    assert status.device_type is None
    assert status.error is None
    assert status.motor_order == []
    assert status.current_motor is None
    assert status.current_motor_index == 0
    assert status.completed_motors == []


def test_calibration_request_dataclass_round_trip() -> None:
    from lelab.calibrate import CalibrationRequest

    req = CalibrationRequest(
        device_type="teleop",
        port="/dev/ttyUSB0",
        config_file="my_calib",
    )
    assert req.device_type == "teleop"
    assert req.port == "/dev/ttyUSB0"
    assert req.config_file == "my_calib"
    assert req.robot_name is None


def test_calibration_manager_starts_idle() -> None:
    from lelab.calibrate import CalibrationManager

    mgr = CalibrationManager()
    assert mgr.status.calibration_active is False
    assert mgr.status.status == "idle"
    assert mgr.device is None
    assert mgr.calibration_thread is None


def test_calibration_manager_get_status_when_idle_returns_status_object() -> None:
    from lelab.calibrate import CalibrationManager, CalibrationStatus

    mgr = CalibrationManager()
    s = mgr.get_status()
    assert isinstance(s, CalibrationStatus)
    assert s.status == "idle"


def test_calibration_manager_rejects_double_start_via_message() -> None:
    """When calibration_active is True, start_calibration returns success=False."""
    from lelab.calibrate import CalibrationManager, CalibrationRequest

    mgr = CalibrationManager()
    mgr.status.calibration_active = True  # simulate already running

    result = mgr.start_calibration(
        CalibrationRequest(device_type="teleop", port="/dev/null", config_file="x")
    )
    assert result.get("success") is False
    assert "already" in result.get("message", "").lower()


def test_cleanup_device_force_releases_and_clears_when_disconnect_fails() -> None:
    """A failed device.disconnect() must still force-close the port and clear the
    device handle — otherwise the COM port stays busy and blocks the next run."""
    from lelab.calibrate import CalibrationManager

    class PortHandler:
        def __init__(self) -> None:
            self.closed = False

        def closePort(self) -> None:  # noqa: N802 - mirrors LeRobot port handler API
            self.closed = True

    class Device:
        def __init__(self) -> None:
            self.bus = type("Bus", (), {"port_handler": PortHandler()})()

        def disconnect(self) -> None:
            raise RuntimeError("Failed to write 'Torque_Enable' on id_=6")

    mgr = CalibrationManager()
    device = Device()
    mgr.device = device

    mgr._cleanup_device()

    assert device.bus.port_handler.closed is True  # force-released despite failure
    assert mgr.device is None  # handle cleared so a new calibration can start


class _FakeMotor:
    def __init__(self, motor_id: int) -> None:
        self.id = motor_id
        self.model = "sts3215"


class _FakeBus:
    """Just enough of FeetechMotorsBus's surface for _calibrate_one_motor's
    own calls: sync_read/write/_get_half_turn_homings/write_calibration.
    `positions` is mutated by the test to simulate the student moving the
    real servo between polls."""

    def __init__(self, motor_names: list[str], start_positions: dict[str, int]) -> None:
        self.motors = {name: _FakeMotor(i + 1) for i, name in enumerate(motor_names)}
        self.positions = dict(start_positions)
        self.calibration_written = None

    def disable_torque(self) -> None:
        pass

    def write(self, key: str, motor: str, value) -> None:  # noqa: ANN001
        pass

    def reset_calibration(self) -> None:
        pass

    def sync_read(self, key: str, normalize: bool = False) -> dict[str, int]:  # noqa: ARG002
        return dict(self.positions)

    def _get_half_turn_homings(self, positions: dict[str, int]) -> dict[str, int]:
        return dict.fromkeys(positions, 2048)

    def write_calibration(self, calibration) -> None:  # noqa: ANN001
        self.calibration_written = calibration

    @property
    def is_connected(self) -> bool:
        return True


class _FakeDevice:
    def __init__(self, bus: _FakeBus) -> None:
        self.bus = bus
        self.calibration = None
        self.calibration_fpath = "fake-calibration-path"

    def _save_calibration(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True


def _wait_for_status(mgr, status: str, timeout: float = 2.0) -> None:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mgr.status.status == status:
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out waiting for status={status!r}; got {mgr.status.status!r}")


def test_per_motor_calibration_retries_one_motor_without_losing_others() -> None:
    """Drives the new bottom-up, one-motor-at-a-time state machine directly
    (bypassing _calibration_worker's real device connection, same as the
    other tests in this file): motor_a gets a discontinuity, is retried in
    place, then both motors complete -- pinning that a retry re-homes and
    re-records only the failing motor, and that completed_motors accumulates
    across both."""
    from lelab.calibrate import CalibrationManager

    bus = _FakeBus(["motor_a", "motor_b"], {"motor_a": 2000, "motor_b": 2000})
    device = _FakeDevice(bus)

    mgr = CalibrationManager()
    mgr.device = device
    mgr.status.calibration_active = True

    def worker() -> None:
        for motor in ("motor_a", "motor_b"):
            mgr._calibrate_one_motor(motor)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # motor_a: homing.
    _wait_for_status(mgr, "homing")
    assert mgr.status.current_motor == "motor_a"
    mgr.complete_step()

    # motor_a: recording -- force a discontinuity (encoder wrap-around). Stays
    # within _is_valid_position's own (0, 5000) exclusive range -- an out-of-
    # range reading would be filtered out before the jump is even checked.
    _wait_for_status(mgr, "recording")
    bus.positions["motor_a"] = 2000 + 2500
    _wait_for_status(mgr, "motor_error")
    assert mgr.status.current_motor == "motor_a"
    assert mgr.status.completed_motors == []

    # Retry motor_a: back to homing, then a real (small but valid) sweep.
    bus.positions["motor_a"] = 2000
    mgr.retry_current_step()
    _wait_for_status(mgr, "homing")
    assert mgr.status.current_motor == "motor_a"
    mgr.complete_step()

    _wait_for_status(mgr, "recording")
    bus.positions["motor_a"] = 2300
    time.sleep(0.1)
    bus.positions["motor_a"] = 1800
    time.sleep(0.1)
    mgr.complete_step()

    # motor_b: homing + recording, no errors this time.
    _wait_for_status(mgr, "homing")
    assert mgr.status.current_motor == "motor_b"
    assert mgr.status.completed_motors == ["motor_a"]
    mgr.complete_step()

    _wait_for_status(mgr, "recording")
    bus.positions["motor_b"] = 2300
    time.sleep(0.1)
    bus.positions["motor_b"] = 1800
    time.sleep(0.1)
    mgr.complete_step()

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert mgr.status.completed_motors == ["motor_a", "motor_b"]
    assert mgr._mins["motor_a"] == 1800
    assert mgr._maxes["motor_a"] == 2300
    assert mgr._mins["motor_b"] == 1800
    assert mgr._maxes["motor_b"] == 2300


def test_wrist_roll_sweeps_normally_but_ignores_an_apparent_wrap() -> None:
    """wrist_roll has real mechanical stops -- on some builds further apart
    than one 4096-step encoder turn, and lerobot's own MotorsBus calibration
    math (_normalize/_unnormalize) has no concept of "which turn" a raw
    reading is in, so a calibrated range wider than one turn would alias
    during later real teleoperation. The fix: still sweep it for real and
    track min/max normally (see _MULTI_TURN_MOTORS), but once a sample looks
    like it wrapped past 0/4095, skip it (don't fault, don't count it)
    instead of erroring out the way a normal motor's discontinuity check
    would -- pinning that this is genuine range tracking, not a "just assume
    the full circle" shortcut. Also pins that a skipped sample doesn't get
    folded into min/max, and that tracking resumes normally afterward."""
    from lelab.calibrate import CalibrationManager

    bus = _FakeBus(["wrist_roll"], {"wrist_roll": 2000})
    device = _FakeDevice(bus)

    mgr = CalibrationManager()
    mgr.device = device
    mgr.status.calibration_active = True

    def worker() -> None:
        mgr._calibrate_one_motor("wrist_roll")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    _wait_for_status(mgr, "homing")
    mgr.complete_step()

    _wait_for_status(mgr, "recording")
    # A real, trackable sweep down to 400...
    bus.positions["wrist_roll"] = 400
    time.sleep(0.1)
    # ...then an apparent wrap (a jump well past _MAX_POSITION_JUMP) -- must
    # NOT raise motor_error, and must NOT extend max to 3800.
    bus.positions["wrist_roll"] = 3800
    time.sleep(0.1)
    assert mgr.status.status == "recording"
    # Back to a normal position near where tracking left off (400) -- confirms
    # the skip didn't corrupt the running "previous position" baseline either.
    bus.positions["wrist_roll"] = 450
    time.sleep(0.1)
    mgr.complete_step()

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert mgr.status.completed_motors == ["wrist_roll"]
    assert mgr._mins["wrist_roll"] == 400
    assert mgr._maxes["wrist_roll"] == 2000  # the starting position -- never exceeded by a valid sample
