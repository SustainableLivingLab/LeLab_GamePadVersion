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

"""Single-servo "claw only" follower for the K12 app's Lesson 2.2.

Some K12 stations only have a single Feetech STS3215 claw servo on an
adapter board -- the same servo model and adapter used on motor 6 of the
real SO-101 arm, just not attached to the other five joints. `SOFollower`
(lerobot's SO-100/101 follower base class) already implements `connect`,
`calibrate`, `configure`, `get_observation`, `send_action` and `disconnect`
generically over whatever motors live in `self.bus.motors` -- the 6-motor
dict is hardcoded only in `SOFollower.__init__`. So this subclass exists
purely to replace that dict with a single "gripper" entry; every other
behavior (including LeLab's web calibration flow in `calibrate.py`, which
also iterates `self.device.bus.motors` generically) is inherited unchanged.

Registered as robot type "claw_follower" -- see `ClawFollowerConfig` below.
"""

from __future__ import annotations

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so_follower.so_follower import SOFollower

from .config_claw_follower import ClawFollowerConfig


class ClawFollower(SOFollower):
    """Single-motor ("gripper" only) variant of `SOFollower`."""

    config_class = ClawFollowerConfig
    # Deliberately kept as "so_follower" (SOFollower's own name), NOT "claw_follower":
    # `Robot.__init__` derives `calibration_dir` from `self.name`, and LeLab's
    # `FOLLOWER_CONFIG_PATH` / `setup_calibration_files` / `is_robot_record_clean`
    # (lelab/utils/config.py) all hardcode that same "so_follower" calibration
    # directory. Reusing it means a claw robot's calibration file lands wherever
    # LeLab already looks -- distinguished from arm robots purely by its own unique
    # config id, same as any two arm robot records already coexist there today.
    # Giving this its own directory name would require teaching config.py about
    # per-robot-type paths for no benefit.
    name = "so_follower"

    def __init__(self, config: ClawFollowerConfig):
        # Deliberately skip SOFollower.__init__ (it hardcodes a 6-motor bus)
        # and call up to Robot.__init__ directly instead.
        super(SOFollower, self).__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        from lerobot.cameras import make_cameras_from_configs

        self.cameras = make_cameras_from_configs(config.cameras)
