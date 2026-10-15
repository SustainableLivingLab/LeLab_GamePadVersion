"""Submit a Hugging Face cloud training run: `lerobot-train --job.target=...`
with two fixes for runs that start from a pretrained model (--policy.path,
e.g. SmolVLA fine-tuned from lerobot/smolvla_base), run as
`python -m lelab.cloud_submit <lerobot-train args>`.

1. Windows paths. lerobot stores --policy.path as a pathlib.Path, so on
   Windows "lerobot/smolvla_base" becomes "lerobot\\smolvla_base" in the
   train_config.json uploaded for the cloud machine, which (Linux) then can't
   resolve it as a Hub model id. The staged config is rewritten with forward
   slashes before upload.
2. The model repo name is supplied by train.build_training_command
   (--policy.repo_id), because lerobot only generates one itself for
   --policy.type runs and otherwise rejects the submit with
   "save_checkpoint_to_hub requires --policy.repo_id". lerobot still names the
   actual repo itself (<user>/<job name>_<time>) when it stages the config.
3. Checkpoint uploads skip the optimizer state (training_state/). It is only
   needed to resume a run exactly, which LeLab doesn't offer, and it is the
   bigger part of a checkpoint: for pi0.5, 15.1 GB of 24.5 GB per checkpoint,
   which filled a free account's 100 GB of private storage mid-run. The
   cloud machine runs `lerobot-train` itself, so the job's command is wrapped
   in a few lines that make its checkpoint uploads ignore that folder.
"""

from __future__ import annotations

import json
from pathlib import Path

# Runs on the cloud machine in place of `lerobot-train <args>`: checkpoint
# uploads (upload_folder into checkpoints/<step>/) leave out training_state/,
# then the real lerobot-train entry point runs with the same arguments. The
# final model push (repo root) is untouched.
_REMOTE_TRAIN = """
import sys
from importlib.metadata import entry_points
from huggingface_hub import HfApi
_upload = HfApi.upload_folder
def upload_folder(self, *args, **kwargs):
    if str(kwargs.get("path_in_repo") or "").startswith("checkpoints/"):
        ignore = kwargs.get("ignore_patterns") or []
        ignore = [ignore] if isinstance(ignore, str) else list(ignore)
        kwargs["ignore_patterns"] = ignore + ["training_state/*", "training_state/**"]
    return _upload(self, *args, **kwargs)
HfApi.upload_folder = upload_folder
print("LeLab: checkpoint uploads skip the optimizer state (training_state/).", flush=True)
train = next(e for e in entry_points(group="console_scripts") if e.name == "lerobot-train").load()
sys.argv[0] = "lerobot-train"
sys.exit(train())
"""


def _install_patches() -> None:
    import lerobot.jobs.hf as hf

    original = hf.build_remote_config_file

    def build_remote_config_file(cfg, repo_id, dest, tags=None):
        out = original(cfg, repo_id, dest, tags)
        data = json.loads(Path(out).read_text())
        policy = data.get("policy") or {}
        pretrained = policy.get("pretrained_path")
        # Only a Hub id with Windows separators; a real local folder is left
        # alone (it can't work on the cloud machine anyway).
        if isinstance(pretrained, str) and "\\" in pretrained and not Path(pretrained).exists():
            policy["pretrained_path"] = pretrained.replace("\\", "/")
            Path(out).write_text(json.dumps(data, indent=4))
        return out

    hf.build_remote_config_file = build_remote_config_file

    original_run_job = hf.run_job

    def run_job(*args, command=None, **kwargs):
        if command and command[0] == "lerobot-train":
            command = ["python", "-c", _REMOTE_TRAIN, *command[1:]]
        return original_run_job(*args, command=command, **kwargs)

    hf.run_job = run_job


def main() -> None:
    _install_patches()
    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
