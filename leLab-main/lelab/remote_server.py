"""Start, watch and stop the cloud GPU that runs a policy for the arm.

The cloud side is a Hugging Face Job running remote_policy_server.py in the
lerobot GPU image, with its port exposed through the HF jobs proxy (reachable
only with the account's token). LeLab keeps at most one such server; the arm
loop (remote_rollout.py) talks to it once it reports ready.

Picking another checkpoint while the server is on swaps the model on the same
machine (POST /load) instead of starting a new one, which skips the queue and
machine start.

States reported to the app: off, starting (waiting for / starting the
machine), loading (model downloading or loading), ready, error; plus "steps"
for the checklist (wait for a GPU, start the machine, download the model,
load it), with "last time" durations.
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
GPU_NAMES = {"l4x1": "L4", "a10g-large": "A10G", "l40sx1": "L40S", "a100-large": "A100"}
# When Hugging Face has no free GPU of one kind, the request waits in its
# queue (sometimes for half an hour or more). After QUEUE_FALLBACK_S without a
# machine, LeLab cancels it and asks for the next kind here instead (both fit
# every policy LeLab trains, at a higher price).
FALLBACK_FLAVOR = {"l4x1": "a10g-large", "a10g-large": "l40sx1"}
QUEUE_FALLBACK_S = 120
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


def _call(url: str, token: str, path: str, body: dict | None = None, timeout: float = 8) -> dict[str, Any] | None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, method="POST" if body is not None else "GET")
    req.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


# ---- "last time" durations for the checklist ------------------------------


def _timings_file() -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / "lelab_cloud_gpu_timings.json"


def _load_timings() -> dict[str, Any]:
    try:
        return json.loads(_timings_file().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_timing(key: str, step: str, seconds: float) -> None:
    try:
        timings = _load_timings()
        timings.setdefault(key, {})[step] = round(seconds, 1)
        _timings_file().write_text(json.dumps(timings, indent=1), encoding="utf-8")
    except Exception:
        pass


# ---- start / swap / stop ---------------------------------------------------


def start(policy_ref: str, flavor: str = DEFAULT_FLAVOR) -> dict[str, Any]:
    repo, subfolder = _model_location(policy_ref)
    with _lock:
        running = dict(_server)
    # A request that fell back to another GPU kind still counts as this one.
    same_machine = running.get("flavor") == flavor or flavor in running.get("skipped_flavors", [])
    if running.get("job_id") and same_machine:
        if running.get("policy_ref") == policy_ref:
            return status()
        # Same machine, another model: swap it there (no queue, no machine
        # start). If the server isn't reachable yet, status() sends the swap
        # once it is.
        with _lock:
            _server.update(policy_ref=policy_ref, swap_requested=time.time(), swap_sent=False)
        logger.info("Cloud GPU: switching model to %s", policy_ref)
        return status()
    if running.get("job_id"):
        stop()

    job = _run_job(repo, subfolder, flavor)
    now = time.time()
    with _lock:
        _server.clear()
        _server.update(
            job_id=job.id,
            job_url=job.url,
            policy_ref=policy_ref,
            flavor=flavor,
            started_at=now,
            flavor_started_at=now,
            skipped_flavors=[],
            url=None,
            running_at=None,
            reachable_at=None,
            swap_sent=True,  # the first model is loaded from the job's settings
        )
    logger.info("Cloud GPU requested: job %s (%s) for %s", job.id, flavor, policy_ref)
    return status()


def _run_job(repo: str, subfolder: str, flavor: str):
    api, token = _hf()
    code = (Path(__file__).parent / "remote_policy_server.py").read_text(encoding="utf-8")
    return api.run_job(
        image=IMAGE,
        command=["python", "-u", "-c", code],
        env={"MODEL_REPO": repo, "MODEL_SUBFOLDER": subfolder, "IDLE_S": str(IDLE_S), "PORT": str(PORT)},
        secrets={"HF_TOKEN": token},
        flavor=flavor,
        expose=[PORT],
        timeout=JOB_TIMEOUT,
        labels={"lelab": "remote-policy"},
    )


def _fall_back(server: dict[str, Any]) -> None:
    """Cancel a request still waiting for a GPU and ask for the next kind."""
    next_flavor = FALLBACK_FLAVOR[server["flavor"]]
    api, _ = _hf()
    # If this fails, keep the old request rather than risk two machines.
    api.cancel_job(job_id=server["job_id"])
    repo, subfolder = _model_location(server["policy_ref"])
    job = _run_job(repo, subfolder, next_flavor)
    with _lock:
        turned_off = _server.get("job_id") != server["job_id"]
        if not turned_off:
            _server.update(
                job_id=job.id,
                job_url=job.url,
                flavor=next_flavor,
                flavor_started_at=time.time(),
                skipped_flavors=[*_server.get("skipped_flavors", []), server["flavor"]],
            )
    if turned_off:
        # Turned off while switching: drop the new request too.
        api.cancel_job(job_id=job.id)
        return
    logger.info(
        "Cloud GPU: no %s free after %d s, asked for %s instead (job %s)",
        server["flavor"], QUEUE_FALLBACK_S, next_flavor, job.id,
    )


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


# ---- status ----------------------------------------------------------------


def _steps(server: dict, stage: str, health: dict | None, now: float) -> list[dict[str, Any]]:
    """Checklist: wait for a GPU, start the machine, download, load, ready."""
    timings = _load_timings()
    machine_last = timings.get(f"machine:{server['flavor']}", {})
    model_last = timings.get(server["policy_ref"], {})
    swapped = server.get("swap_requested") is not None

    def step(step_id, label, state, seconds=None, last=None, detail=None):
        item = {"id": step_id, "label": label, "state": state}
        if seconds is not None:
            item["seconds"] = round(seconds, 1)
        if last is not None:
            item["last_seconds"] = last
        if detail:
            item["detail"] = detail
        return item

    out = []
    if not swapped:
        running_at, reachable_at = server.get("running_at"), server.get("reachable_at")
        queue_label = "Wait for a free GPU on Hugging Face"
        skipped = server.get("skipped_flavors") or []
        if skipped:
            names = " or ".join(GPU_NAMES.get(f, f) for f in skipped)
            queue_label = f"No {names} GPU was free, so waiting for an {GPU_NAMES.get(server['flavor'], server['flavor'])} instead"
        if stage != "RUNNING":
            out.append(step("queue", queue_label, "active",
                            now - server["started_at"], machine_last.get("queue")))
        else:
            out.append(step("queue", queue_label, "done",
                            (running_at or now) - server["started_at"]))
        if stage != "RUNNING":
            out.append(step("machine", "Start the cloud machine", "pending"))
        elif reachable_at is None:
            out.append(step("machine", "Start the cloud machine", "active",
                            now - (running_at or now), machine_last.get("machine")))
        else:
            out.append(step("machine", "Start the cloud machine", "done", reachable_at - (running_at or reachable_at)))

    phase = (health or {}).get("phase")
    times = (health or {}).get("times") or {}
    model_matches = (health or {}).get("model") == _model_key(server["policy_ref"])
    elapsed = (health or {}).get("phase_elapsed")
    if health is None or not model_matches or phase == "starting":
        download_state, load_state = "pending", "pending"
    elif phase == "downloading":
        download_state, load_state = "active", "pending"
    elif phase == "loading":
        download_state, load_state = "done", "active"
    elif phase == "ready":
        download_state, load_state = "done", "done"
    else:
        download_state, load_state = "done", "pending"

    detail = None
    progress = (health or {}).get("download") or {}
    if download_state == "active" and progress.get("total"):
        detail = f"{progress['done'] / 1e9:.1f} of {progress['total'] / 1e9:.1f} GB"
    out.append(step("download", "Download the model", download_state,
                    elapsed if download_state == "active" else (times.get("download") if download_state == "done" else None),
                    model_last.get("download"), detail))
    out.append(step("load", "Load the AI on the GPU", load_state,
                    elapsed if load_state == "active" else (times.get("load") if load_state == "done" else None),
                    model_last.get("load")))
    return out


def _model_key(policy_ref: str) -> str:
    repo, sub = _model_location(policy_ref)
    return f"{repo}@{sub}"


def status() -> dict[str, Any]:
    with _lock:
        server = dict(_server)
    if not server.get("job_id"):
        return {"state": "off"}

    now = time.time()
    base = {
        "policy_ref": server["policy_ref"],
        "flavor": server["flavor"],
        "price_per_hour": PRICE_PER_HOUR.get(server["flavor"]),
        "elapsed_s": round(now - server["started_at"]),
        "job_url": server.get("job_url"),
    }
    api, token = _hf()
    try:
        info = api.inspect_job(job_id=server["job_id"])
    except Exception as exc:
        return {**base, "state": "starting", "message": f"Could not reach Hugging Face: {exc}", "steps": []}

    stage = info.status.stage
    if (
        stage == "SCHEDULING"
        and server["flavor"] in FALLBACK_FLAVOR
        and now - server.get("flavor_started_at", server["started_at"]) > QUEUE_FALLBACK_S
    ):
        # Only one status poll switches (the app polls from several places).
        with _lock:
            claimed = _server.get("job_id") == server["job_id"] and not _server.get("falling_back")
            if claimed:
                _server["falling_back"] = True
        if claimed:
            try:
                _fall_back(server)
                return status()
            except Exception as exc:
                logger.warning("Cloud GPU: could not ask for another GPU kind: %s", exc)
                # Keep waiting on this one; try again after another wait.
                with _lock:
                    if _server.get("job_id") == server["job_id"]:
                        _server["flavor_started_at"] = now
            finally:
                with _lock:
                    _server.pop("falling_back", None)

    if stage in ("ERROR", "COMPLETED", "CANCELED", "DELETED"):
        with _lock:
            if _server.get("job_id") == server["job_id"]:
                _server.clear()
        if stage == "ERROR":
            return {**base, "state": "error", "message": info.status.message or "The cloud GPU stopped with an error."}
        # COMPLETED: the server turned itself off after being idle.
        return {"state": "off", "message": "The cloud GPU turned itself off."}

    url = (getattr(info.status, "expose_urls", None) or [None])[0]
    health = None
    if stage == "RUNNING":
        with _lock:
            if _server.get("job_id") == server["job_id"] and _server.get("running_at") is None:
                _server["running_at"] = now
                _save_timing(f"machine:{server['flavor']}", "queue",
                             now - server.get("flavor_started_at", server["started_at"]))
            server = dict(_server)
        if url:
            health = _call(url, token, "/health")
            with _lock:
                if _server.get("job_id") == server["job_id"]:
                    _server["url"] = url
                    if health is not None and _server.get("reachable_at") is None:
                        _server["reachable_at"] = now
                        _save_timing(f"machine:{server['flavor']}", "machine", now - (_server["running_at"] or now))
                server = dict(_server)
            # A model switch asked for before the server could be reached.
            if health is not None and not server.get("swap_sent"):
                repo, sub = _model_location(server["policy_ref"])
                if _call(url, token, "/load", {"repo": repo, "subfolder": sub}) is not None:
                    with _lock:
                        _server["swap_sent"] = True
                    health = _call(url, token, "/health")

    steps = _steps(server, stage, health, now)
    if health is not None and health.get("phase") == "ready" and health.get("model") == _model_key(server["policy_ref"]):
        # Remember how long this model took, for "last time" next time.
        times = health.get("times") or {}
        with _lock:
            saved = _server.get("timings_saved_for")
            if saved != server["policy_ref"]:
                _server["timings_saved_for"] = server["policy_ref"]
                for step_id in ("download", "load"):
                    if step_id in times:
                        _save_timing(server["policy_ref"], step_id, times[step_id])
        return {**base, "state": "ready", "steps": steps}
    if health is not None and health.get("error") and health.get("model") == _model_key(server["policy_ref"]):
        return {**base, "state": "error", "message": health["error"], "steps": steps}
    if health is None:
        return {**base, "state": "starting", "steps": steps}
    return {**base, "state": "loading", "steps": steps}


def ready_url(policy_ref: str) -> str:
    """The server address to drive the arm with, or ValueError with a reason."""
    current = status()
    if current.get("state") != "ready":
        raise ValueError("The cloud GPU isn't ready yet. Start it first and wait until it says ready.")
    if current.get("policy_ref") != policy_ref:
        raise ValueError("The cloud GPU has a different training loaded. Load this one first.")
    with _lock:
        return _server["url"]
