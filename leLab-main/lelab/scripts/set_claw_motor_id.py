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

"""One-time setup for a fresh Lesson 2.2 claw servo.

A Feetech STS3215 ships from the factory at ID 1 (and its own factory
baud-rate) -- `claw_follower.py` hard-codes the single "gripper" motor as ID
6 (matching motor 6 on the full SO-101 arm), so a never-configured servo
fails calibration with exactly this error:

    FeetechMotorsBus motor check failed on port '<port>':
    Missing motor IDs:
      - 6 (expected model: 777)
    Full found motor list (id: model_number):
    {}

("found: {}" because calibration only PINGS the ID it expects (6) -- it
never broadcasts, so a motor sitting at any other ID looks like nothing is
there at all, not "wrong ID".)

This burns the target ID (and the bus's default baud-rate) into whatever
SINGLE motor is currently connected -- run it once per fresh claw board,
with ONLY that one servo wired to the controller/adapter, before ever
opening Start Calibration in the K12 app. Deliberately talks to the bus
directly (FeetechMotorsBus), not through ClawFollower/Robot -- this is a
one-shot hardware-register write with no calibration file or camera
involved, so there's nothing to gain from that extra machinery.

Usage:
    python -m lelab.scripts.set_claw_motor_id --port COM8
"""

from __future__ import annotations

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# Must match claw_follower.py's own Motor(6, "sts3215", ...) exactly, or a
# servo set up here would still fail that file's calibration check.
GRIPPER_MOTOR_ID = 6
GRIPPER_MODEL = "sts3215"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="Serial port the claw's controller board is on, e.g. COM8.")
    args = parser.parse_args()

    bus = FeetechMotorsBus(
        port=args.port,
        motors={"gripper": Motor(GRIPPER_MOTOR_ID, GRIPPER_MODEL, MotorNormMode.RANGE_0_100)},
    )

    input(
        f"Connect ONLY the claw's servo to the controller board on {args.port} "
        "(no other motors on the bus), then press enter."
    )
    bus.setup_motor("gripper")
    print(f"Done -- the connected servo is now ID {GRIPPER_MOTOR_ID}. Start Calibration in the K12 app should work now.")


if __name__ == "__main__":
    main()
