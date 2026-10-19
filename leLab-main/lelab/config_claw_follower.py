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

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig


@RobotConfig.register_subclass("claw_follower")
@dataclass
class ClawFollowerConfig(RobotConfig, SOFollowerConfig):
    """Config for `ClawFollower` -- a single claw servo on its own adapter
    board, not attached to a full SO-101 arm. Shares every field with
    `SOFollowerConfig` (port, cameras, max_relative_target, ...); only the
    motor set differs, and that's decided in `ClawFollower` itself."""

    pass
