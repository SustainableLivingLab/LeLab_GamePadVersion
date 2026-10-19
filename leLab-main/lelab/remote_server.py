"""Start, watch and stop the cloud GPU that runs a policy for the arm.

The cloud side is a Hugging Face Job running remote_policy_server.py in the
lerobot GPU image, with its port exposed through the HF jobs proxy (reachable
only with the account's token). LeLab keeps at most one such server; the arm
loop (remote_rollout.py) talks to it once it reports ready.

States reported to the app: off, starting (machine starting), loading (model
loading), ready, error.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# L4: enough GPU memory (24 GB) and RAM (30 GB; loading pi0.5 briefly needs
# ~19 GB) for every policy LeLab trains, at the lowest price that fits.
DEFAULT_FLAVOR = "l4x1"
PRICE_PER_HOUR = {"l4x1": 0.80, "a10g-large": 1.50, "l40sx1": 1.80, "a100-large": 2.50}
IMAGE = "huggingface/lerobot-gpu:latest"
PORT = 8000
# The server turns itself off after this long without arm requests, and the
# job has a hard limit on top, so a forgotten session can't keep billing.
IDLE_S = 600
JOB_TIMEOUT = "3h"

_lock = threading.Lock()
_server: dict[str, Any] = {}


def _hf():
    from huggingface_hub import HfApi, get_token

    return HfApi(), get_token()


def _model_location(policy_ref: str) -> tuple[str, str]:
    """(repo, subfolder) of a cloud training's checkpoint ref."""
    from .rollout import _HUB_REF_RE, _HUB_ROOT_REF_RE

    m = _HUB_REF_RE.match(policy_ref)
    if m:
        return m.group("repo"), f"checkpoints/{m.group('step_dir')}/pretrained_model"
    m = _HUB_ROOT_REF_RE.match(policy_ref)
    if m:
        return m.group("repo"), ""
    raise ValueError("Only trainings saved on Hugging Face can run on the cloud GPU.")


def _health(url: str, token: str) -> dict[str, Any] | None:
    req = urllib.request.Request(url + "/health")
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def start(policy_ref: str, flavor: str = DEFAULT_FLAVOR) -> dict[str, Any]:
    repo, subfolder = _model_location(policy_ref)
    with _lock:
        if _server.get("job_id") and _server.get("policy_ref") == policy_ref:
            return status()
    if _server.get("job_id"):
        stop()

    api, token = _hf()
    code = (Path(__file__).parent / "remote_policy_server.py").read_text(encoding="utf-8")
    job = api.run_job(
        image=IMAGE,
        command=["python", "-u", "-c", code],
        env={"MODEL_REPO": repo, "MODEL_SUBFOLDER": subfolder, "IDLE_S": str(IDLE_S), "PORT": str(PORT)},
        secrets={"HF_TOKEN": token},
        flavor=flavor,
        expose=[PORT],
        timeout=JOB_TIMEOUT,
        labels={"lelab": "remote-policy"},
    )
    with _lock:
        _server.clear()
        _server.update(
            job_id=job.id,
            job_url=job.url,
            policy_ref=policy_ref,
            flavor=flavor,
            started_at=time.time(),
            url=None,
        )
    logger.info("Cloud GPU requested: job %s (%s) for %s", job.id, flavor, policy_ref)
    return status()


def status() -> dict[str, Any]:
    with _lock:
        server = dict(_server)
    if not server.get("job_id"):
        return {"state": "off"}

    base = {
        "policy_ref": server["policy_ref"],
        "flavor": server["flavor"],
        "price_per_hour": PRICE_PER_HOUR.get(server["flavor"]),
        "elapsed_s": round(time.time() - server["started_at"]),
        "job_url": server.get("job_url"),
    }
    api, token = _hf()
    try:
        info = api.inspect_job(job_id=server["job_id"])
    except Exception as exc:
        return {**base, "state": "starting", "message": f"Could not reach Hugging Face: {exc}"}

    stage = info.status.stage
    if stage in ("ERROR", "COMPLETED", "CANCELED", "DELETED"):
        with _lock:
            if _server.get("job_id") == server["job_id"]:
                _server.clear()
        if stage == "ERROR":
            return {**base, "state": "error", "message": info.status.message or "The cloud GPU stopped with an error."}
        # COMPLETED: the server turned itself off after being idle.
        return {"state": "off", "message": "The cloud GPU turned itself off."}

    url = (getattr(info.status, "expose_urls", None) or [None])[0]
    if stage != "RUNNING" or not url:
        return {**base, "state": "starting"}
    with _lock:
        if _server.get("job_id") == server["job_id"]:
            _server["url"] = url
    health = _health(url, token)
    if health is None:
        return {**base, "state": "starting"}
    if health.get("error"):
        return {**base, "state": "error", "message": health["error"]}
    if not health.get("ready"):
        return {**base, "state": "loading"}
    return {**base, "state": "ready"}


def ready_url(policy_ref: str) -> str:
    """The server address to drive the arm with, or ValueError with a reason."""
    current = status()
    if current.get("state") != "ready":
        raise ValueError("The cloud GPU isn't ready yet. Start it first and wait until it says ready.")
    if current.get("policy_ref") != policy_ref:
        raise ValueError("The cloud GPU has a different training loaded. Start it again for this one.")
    with _lock:
        return _server["url"]


def stop() -> dict[str, Any]:
    with _lock:
        job_id = _server.get("job_id")
        _server.clear()
    if job_id:
        api, _ = _hf()
        try:
            api.cancel_job(job_id=job_id)
            logger.info("Cloud GPU stopped: job %s", job_id)
        except Exception as exc:
            logger.warning("Could not cancel cloud GPU job %s: %s", job_id, exc)
    return {"state": "off"}
