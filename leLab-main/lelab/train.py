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

"""Training-specific helpers: the request schema and the LeRobot CLI builder.

The actual job lifecycle (subprocess management, registry, log streaming)
lives in app/jobs.py.
"""

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from lelab.jobs import JobTarget

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


class TrainingRequest(BaseModel):
    # Dataset configuration
    dataset_repo_id: str
    dataset_revision: str | None = None
    dataset_root: str | None = None
    dataset_episodes: list[int] | None = None
    dataset_image_transforms_enable: bool = False

    # Policy configuration. policy_type trains that architecture from scratch;
    # policy_path instead fine-tunes a pretrained checkpoint (Hub id or local
    # dir, e.g. SMOLVLA_BASE) and takes precedence -- its own config says
    # which policy type it is.
    policy_type: str = "act"
    policy_path: str | None = None

    # Core training parameters
    steps: int = 10000
    batch_size: int = 8
    seed: int | None = 1000
    num_workers: int = 4

    # Logging and checkpointing
    log_freq: int = 250
    save_freq: int = 1000
    env_eval_freq: int = 0
    save_checkpoint: bool = True

    # Output configuration
    output_dir: str = "outputs/train"
    resume: bool = False
    job_name: str | None = None

    # Weights & Biases
    wandb_enable: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_notes: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str | None = "online"
    wandb_disable_artifact: bool = False

    # Environment / evaluation
    env_type: str | None = None
    env_task: str | None = None
    eval_n_episodes: int = 10
    eval_batch_size: int = 50
    eval_use_async_envs: bool = False

    # Policy-specific
    policy_device: str | None = "cuda"
    policy_use_amp: bool = False
    # Hub upload (set by HfCloudJobRunner; not exposed in the form)
    policy_push_to_hub: bool = False
    policy_repo_id: str | None = None

    # GR00T-specific policy options (only emitted when policy_type == "groot";
    # these flags don't exist on other policy configs, so draccus would reject
    # them). All optional — unset fields fall back to the groot config defaults.
    policy_base_model_path: str | None = None
    policy_embodiment_tag: str | None = None
    policy_chunk_size: int | None = None
    policy_n_action_steps: int | None = None
    policy_use_relative_actions: bool | None = None
    policy_relative_exclude_joints: list[str] | None = None
    policy_use_bf16: bool | None = None

    # Optimizer
    optimizer_type: str | None = "adam"
    optimizer_lr: float | None = None
    optimizer_weight_decay: float | None = None
    optimizer_grad_clip_norm: float | None = None

    # Advanced
    use_policy_training_preset: bool = True
    config_path: str | None = None


SMOLVLA_BASE = "lerobot/smolvla_base"

_IMAGE_PREFIX = "observation.images."


def _read_json_file(repo_or_dir: str, filename: str, repo_type: str, revision: str | None = None) -> dict:
    """A small JSON file from a local directory, or from the Hub (cached)."""
    local = Path(repo_or_dir) / filename
    if local.is_file():
        return json.loads(local.read_text(encoding="utf-8"))
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_or_dir, filename=filename, repo_type=repo_type, revision=revision)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dataset_camera_keys(request: TrainingRequest) -> list[str]:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    root = request.dataset_root or str(HF_LEROBOT_HOME / request.dataset_repo_id)
    if (Path(root) / "meta" / "info.json").is_file():
        info = _read_json_file(root, "meta/info.json", "dataset")
    else:
        info = _read_json_file(request.dataset_repo_id, "meta/info.json", "dataset", request.dataset_revision)
    return sorted(key for key in info.get("features", {}) if key.startswith(_IMAGE_PREFIX))


def camera_rename_map(request: TrainingRequest) -> dict[str, str]:
    """Map the dataset's camera keys onto the pretrained policy's own.

    A pretrained checkpoint keeps the camera names it was trained with (e.g.
    smolvla_base's generic camera1/2/3), and lerobot refuses to fine-tune on a
    dataset whose cameras are named differently unless given a --rename_map.
    Cameras are paired in sorted-name order; a dataset with fewer cameras than
    the checkpoint is fine (SmolVLA only needs one), more is an error.
    """
    policy_cfg = _read_json_file(request.policy_path, "config.json", "model")
    policy_cams = sorted(
        key
        for key, feature in (policy_cfg.get("input_features") or {}).items()
        if feature.get("type") == "VISUAL"
    )
    dataset_cams = _dataset_camera_keys(request)
    if not policy_cams or set(dataset_cams) <= set(policy_cams):
        return {}
    if len(dataset_cams) > len(policy_cams):
        raise ValueError(
            f"Dataset has {len(dataset_cams)} cameras but {request.policy_path} only takes {len(policy_cams)}."
        )
    return dict(zip(dataset_cams, policy_cams, strict=False))


def _require_smolvla_extra_if_needed(request: TrainingRequest) -> None:
    """SmolVLA's dependencies (transformers etc.) are an optional extra, so a
    normal install doesn't download them. Fail the request up front with the
    install command (a ValueError -> HTTP 400 in server.py) rather than
    letting the training subprocess die on an ImportError minutes in."""
    if request.policy_path:
        policy_type = _read_json_file(request.policy_path, "config.json", "model").get("type")
    else:
        policy_type = request.policy_type
    if policy_type != "smolvla":
        return

    import importlib.util

    missing = [name for name in ("transformers", "num2words") if importlib.util.find_spec(name) is None]
    if missing:
        raise ValueError(
            "SmolVLA training needs extra packages that aren't installed ("
            + ", ".join(missing)
            + '). In the leLab-main folder run: uv pip install -e ".[smolvla]" -- or, for the one-line '
            'install, reinstall with "lelab-gamepad[smolvla] @ git+https://github.com/SustainableLivingLab/'
            'LeLab_GamePadVersion.git@Gokcever1#subdirectory=leLab-main". Then restart LeLab.'
        )


def _require_recorded_episodes(request: TrainingRequest) -> None:
    """A local dataset with 0 episodes (a recording that failed before saving
    any) makes lerobot fall back to downloading it from the Hub, which fails
    with an unhelpful 401. Say what's actually wrong instead."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    from .datasets import local_dataset_episode_count

    root = Path(request.dataset_root) if request.dataset_root else HF_LEROBOT_HOME / request.dataset_repo_id
    if local_dataset_episode_count(root) == 0:
        raise ValueError(
            f'The dataset "{request.dataset_repo_id}" has no recorded episodes -- its recording stopped '
            "before any episode was saved. Pick a dataset with episodes, or record a new one."
        )


def build_training_command(
    request: TrainingRequest,
    output_dir: str,
    python_executable: str = "python",
    job_target: "JobTarget | None" = None,
) -> list[str]:
    """Build the argv list to invoke `<python_executable> -m lerobot.scripts.lerobot_train`.

    `output_dir` is supplied separately from the request so the caller (the
    JobRegistry) can pin it to the per-job directory rather than relying on
    request.output_dir, which the frontend doesn't even send in the new world.

    `python_executable` defaults to "python" for the cloud runner (whose
    container has lerobot on PATH); the local runner must pass sys.executable
    so the subprocess uses the same interpreter as lelab itself — otherwise
    PATH lookup picks up a different env (uv tool venv, miniforge3 base, etc.)
    that lacks lerobot.
    """
    cmd: list[str] = [python_executable, "-m", "lerobot.scripts.lerobot_train"]

    # Dataset
    cmd.extend(["--dataset.repo_id", request.dataset_repo_id])
    if request.dataset_revision:
        cmd.extend(["--dataset.revision", request.dataset_revision])
    if request.dataset_root:
        cmd.extend(["--dataset.root", request.dataset_root])
    if request.dataset_episodes:
        cmd.extend(["--dataset.episodes"] + [str(ep) for ep in request.dataset_episodes])
    if request.dataset_image_transforms_enable:
        cmd.extend(["--dataset.image_transforms.enable", "true"])
    if sys.platform == "win32":
        # lerobot's get_safe_default_video_backend() picks torchcodec whenever
        # it *imports* successfully, but on Windows that import can succeed
        # while the actual native DLL (libtorchcodec_core*.dll, which needs
        # FFmpeg's shared libs) still fails to load at decode time -- crashing
        # training with a FileNotFoundError/OSError once it actually reads a
        # video frame. pyav doesn't have this gap and works reliably on
        # Windows, so force it there rather than trusting the "safe" default.
        cmd.extend(["--dataset.video_backend", "pyav"])

    # Policy
    is_local_run = job_target is None or job_target.runner != "hf_cloud"
    if is_local_run:
        _require_smolvla_extra_if_needed(request)
        _require_recorded_episodes(request)
    if request.policy_path:
        policy_path = request.policy_path
        if sys.platform == "win32" and is_local_run and not Path(policy_path).is_dir():
            # lerobot stores --policy.path as a pathlib.Path, which on Windows
            # turns a Hub id like "lerobot/smolvla_base" into
            # "lerobot\smolvla_base" -- an invalid repo id by the time it
            # downloads the weights. Download it ourselves (cached after the
            # first time) and hand lerobot the local folder instead.
            from huggingface_hub import snapshot_download

            policy_path = snapshot_download(repo_id=policy_path, repo_type="model")
        # Must be the single "--policy.path=<x>" token: lerobot pre-parses it
        # (parser.get_path_arg) and argparse rejects the space-separated form.
        cmd.append(f"--policy.path={policy_path}")
        rename_map = camera_rename_map(request)
        if rename_map:
            cmd.append(f"--rename_map={json.dumps(rename_map)}")
    else:
        cmd.extend(["--policy.type", request.policy_type])
        if request.policy_type == "smolvla":
            # "From scratch" for SmolVLA means a fresh action expert on top of
            # the PRETRAINED SmolVLM backbone. lerobot's own default here is
            # False, which would leave the whole VLM randomly initialised too.
            cmd.extend(["--policy.load_vlm_weights", "true"])

    # Core training params
    cmd.extend(["--steps", str(request.steps)])
    cmd.extend(["--batch_size", str(request.batch_size)])
    cmd.extend(["--num_workers", str(request.num_workers)])
    if request.seed is not None:
        cmd.extend(["--seed", str(request.seed)])

    # Policy device / AMP / hub
    if request.policy_device:
        cmd.extend(["--policy.device", request.policy_device])
    cmd.extend(["--policy.use_amp", "true" if request.policy_use_amp else "false"])
    # On HF Cloud, lerobot's submit_to_hf owns the model repo and sets push_to_hub on
    # the pod itself; _pod_forwarded_args drops any --policy.push_to_hub/--policy.repo_id
    # we'd pass, so we must not emit them. Local runs keep the existing behavior:
    # LeRobot defaults push_to_hub=True and demands --policy.repo_id when so.
    is_cloud = job_target is not None and job_target.runner == "hf_cloud"
    if not is_cloud:
        cmd.extend(["--policy.push_to_hub", "true" if request.policy_push_to_hub else "false"])
        if request.policy_push_to_hub and request.policy_repo_id:
            cmd.extend(["--policy.repo_id", request.policy_repo_id])

    # GR00T-specific policy flags. These options only exist on the groot config,
    # so emit them only for --policy.type=groot; sending them to another policy
    # would make draccus abort the run. Each is skipped when unset so the groot
    # config's own defaults apply.
    if request.policy_type == "groot":
        if request.policy_base_model_path:
            cmd.extend(["--policy.base_model_path", request.policy_base_model_path])
        if request.policy_embodiment_tag:
            cmd.extend(["--policy.embodiment_tag", request.policy_embodiment_tag])
        if request.policy_chunk_size is not None:
            cmd.extend(["--policy.chunk_size", str(request.policy_chunk_size)])
        if request.policy_n_action_steps is not None:
            cmd.extend(["--policy.n_action_steps", str(request.policy_n_action_steps)])
        if request.policy_use_relative_actions is not None:
            cmd.extend(
                ["--policy.use_relative_actions", "true" if request.policy_use_relative_actions else "false"]
            )
        if request.policy_relative_exclude_joints is not None:
            # draccus parses list values from a single JSON token, e.g. '["gripper"]'.
            cmd.extend(
                ["--policy.relative_exclude_joints", json.dumps(request.policy_relative_exclude_joints)]
            )
        if request.policy_use_bf16 is not None:
            cmd.extend(["--policy.use_bf16", "true" if request.policy_use_bf16 else "false"])

    # Logging / checkpointing
    cmd.extend(["--log_freq", str(request.log_freq)])
    cmd.extend(["--save_freq", str(request.save_freq)])
    cmd.extend(["--env_eval_freq", str(request.env_eval_freq)])
    cmd.extend(["--save_checkpoint", "true" if request.save_checkpoint else "false"])

    # Output. On HF Cloud the pod, not this host, runs the trainer: an absolute host
    # output_dir (e.g. ~/.cache/.../outputs/train) is baked into the staged config and
    # the pod crashes trying to mkdir it under /Users. Checkpoints land on the Hub repo
    # anyway, so we omit it for cloud and let lerobot pick its in-pod default.
    if not is_cloud:
        cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--resume", "true" if request.resume else "false"])
    if request.job_name:
        cmd.extend(["--job_name", request.job_name])

    # W&B
    cmd.extend(["--wandb.enable", "true" if request.wandb_enable else "false"])
    if request.wandb_enable:
        if request.wandb_project:
            cmd.extend(["--wandb.project", request.wandb_project])
        if request.wandb_entity:
            cmd.extend(["--wandb.entity", request.wandb_entity])
        if request.wandb_notes:
            cmd.extend(["--wandb.notes", request.wandb_notes])
        if request.wandb_run_id:
            cmd.extend(["--wandb.run_id", request.wandb_run_id])
        if request.wandb_mode:
            cmd.extend(["--wandb.mode", request.wandb_mode])
        cmd.extend(["--wandb.disable_artifact", "true" if request.wandb_disable_artifact else "false"])

    # Env
    if request.env_type:
        cmd.extend(["--env.type", request.env_type])
    if request.env_task:
        cmd.extend(["--env.task", request.env_task])

    # Eval
    cmd.extend(["--eval.n_episodes", str(request.eval_n_episodes)])
    cmd.extend(["--eval.batch_size", str(request.eval_batch_size)])
    cmd.extend(["--eval.use_async_envs", "true" if request.eval_use_async_envs else "false"])

    # Optimizer
    if request.optimizer_type:
        cmd.extend(["--optimizer.type", request.optimizer_type])
    if request.optimizer_lr is not None:
        cmd.extend(["--optimizer.lr", str(request.optimizer_lr)])
    if request.optimizer_weight_decay is not None:
        cmd.extend(["--optimizer.weight_decay", str(request.optimizer_weight_decay)])
    if request.optimizer_grad_clip_norm is not None:
        cmd.extend(["--optimizer.grad_clip_norm", str(request.optimizer_grad_clip_norm)])

    # Advanced
    cmd.extend(["--use_policy_training_preset", "true" if request.use_policy_training_preset else "false"])
    if request.config_path:
        cmd.extend(["--config_path", request.config_path])

    # HF Jobs: --job.target=<flavor> dispatches the run remotely (lerobot commit #3856).
    # Image/timeout use lerobot's JobConfig defaults. lelab tags its jobs; lerobot always
    # adds a "lerobot" tag too. A pod's local checkpoints die with it, so push each one to
    # the model repo's checkpoints/<step>/ tree (the native replacement for lelab's old
    # in-pod uploader) — that's what makes the trained checkpoints reachable afterwards.
    if is_cloud and job_target.flavor:
        cmd.extend(["--job.target", job_target.flavor])
        cmd.extend(["--job.tags", '["lelab"]'])
        # save_checkpoint_to_hub needs policy.repo_id, which submit_to_hf only sets on the
        # fresh-run path; on a resume it isn't set before validate(), so the flag would
        # abort the submit. A resume already pushes back to its source repo, so skip it.
        if request.save_checkpoint and not request.resume:
            cmd.extend(["--save_checkpoint_to_hub", "true"])

    if request.policy_path:
        # With --policy.path, lerobot reads every other --policy.* flag as an
        # override of the loaded config, and only accepts the "--policy.x=y"
        # form for those -- a separate value token is an unrecognized argument.
        merged: list[str] = []
        args = iter(cmd)
        for arg in args:
            if arg.startswith("--policy.") and "=" not in arg:
                merged.append(f"{arg}={next(args)}")
            else:
                merged.append(arg)
        cmd = merged

    return cmd
