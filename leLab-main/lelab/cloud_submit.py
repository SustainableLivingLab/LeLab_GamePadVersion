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
"""

from __future__ import annotations

import json
from pathlib import Path


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


def main() -> None:
    _install_patches()
    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
