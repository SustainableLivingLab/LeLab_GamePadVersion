"""Model server for running a trained policy on a Hugging Face Jobs GPU.

This file is sent to the cloud machine as the job's command (it runs in the
`huggingface/lerobot-gpu` image, so it imports only what that image has) and
serves the policy over plain HTTP on PORT, which the job exposes through the
HF jobs proxy (only callers with the account's HF token get through).

  GET  /health  -> {"ready": bool, "phase": "downloading"|"loading"|"ready"|"error",
                    "model": "repo@subfolder", "download": {"done", "total"},
                    "times": {"download": s, "load": s}, "error": str|None, ...}
  POST /act     -> body: {"images": {name: base64 JPEG}, "state": [floats],
                          "task": str}
                   reply: {"actions": [[...], ...], "names": [...],
                           "infer_ms": float}
  POST /load    -> body: {"repo": str, "subfolder": str}. Swaps to another
                   checkpoint on the same machine (no new machine to start).

The /act reply is the whole planned chunk, already un-normalised into robot
units and ordered like the dataset's action names, so the laptop only has to
play it. The process exits after IDLE_S seconds without /act requests (once
the model is ready) so a forgotten session doesn't keep a paid GPU running.

Environment: MODEL_REPO, MODEL_SUBFOLDER (e.g. checkpoints/002000/pretrained_model,
or empty for the repo root), PORT (8000), IDLE_S (600), HF_TOKEN (secret).
MODEL_PATH (a local folder instead of MODEL_REPO) and DEVICE (cuda) are for
testing on a PC.
"""

import base64
import hashlib
import http.server
import io
import json
import os
import threading
import time
import traceback

DEVICE = os.environ.get("DEVICE", "cuda")
STATE = {
    "ready": False,
    "phase": "starting",
    "phase_started": time.time(),
    "error": None,
    "model": None,
    "download": None,
    "times": {},
    "loaded_in_s": None,
}
LAST_USE = [time.time()]
LOCK = threading.Lock()
MODEL = {}
_DOWNLOAD = {}
_LOADING = threading.Lock()


def log(message):
    print(f"[remote-policy] {message}", flush=True)


def _phase(name):
    STATE["phase"] = name
    STATE["phase_started"] = time.time()


def _download_progress():
    folder, total = _DOWNLOAD.get("dir"), _DOWNLOAD.get("total")
    if not folder:
        return None
    done = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                done += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return {"done": done, "total": total}


def load(repo, sub):
    """Download (if needed) and load a checkpoint; replaces the current one
    only once the new one is ready."""
    with _LOADING:
        started = time.time()
        STATE.update(ready=False, error=None, model=f"{repo}@{sub}" if repo else None)
        try:
            import torch
            from huggingface_hub import HfApi, hf_hub_download, snapshot_download
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors

            if os.environ.get("MODEL_PATH"):
                path = os.environ["MODEL_PATH"]
            else:
                _phase("downloading")
                local_dir = os.path.join("/tmp/models", hashlib.sha1(repo.encode()).hexdigest()[:12])
                prefix = f"{sub}/" if sub else ""
                try:
                    info = HfApi().model_info(repo, files_metadata=True)
                    total = sum((f.size or 0) for f in info.siblings if f.rfilename.startswith(prefix))
                except Exception:
                    total = None
                _DOWNLOAD.update(dir=os.path.join(local_dir, sub) if sub else local_dir, total=total)
                snapshot_download(repo, allow_patterns=[f"{sub}/*"] if sub else None, local_dir=local_dir)
                path = os.path.join(local_dir, sub) if sub else local_dir
                STATE["times"]["download"] = round(time.time() - started, 1)
                log(f"downloaded {repo}/{sub} in {STATE['times']['download']:.0f}s")

            _phase("loading")
            load_started = time.time()
            # Free the old model first (a GPU may not hold two big ones).
            with LOCK:
                MODEL.clear()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            cfg = PreTrainedConfig.from_pretrained(path)
            cfg.device = DEVICE
            policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(DEVICE).eval()
            pre, post = make_pre_post_processors(
                cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": DEVICE}}
            )

            # The dataset's action names give the order of the action vector.
            names = None
            try:
                with open(os.path.join(path, "train_config.json")) as fh:
                    dataset_repo = json.load(fh)["dataset"]["repo_id"]
                info_file = hf_hub_download(dataset_repo, "meta/info.json", repo_type="dataset")
                with open(info_file) as fh:
                    names = json.load(fh)["features"]["action"]["names"]
            except Exception as exc:  # the client can still play actions by position
                log(f"could not read action names: {exc}")

            image_keys = sorted(k for k, f in cfg.input_features.items() if str(f.type).endswith("VISUAL"))
            with LOCK:
                MODEL.update(policy=policy, pre=pre, post=post, names=names, image_keys=image_keys, torch=torch)
            STATE["times"]["load"] = round(time.time() - load_started, 1)
            _phase("ready")
            STATE.update(ready=True, loaded_in_s=round(time.time() - started, 1))
            LAST_USE[0] = time.time()
            log(f"ready in {STATE['loaded_in_s']}s: {cfg.type}, cameras {image_keys}")
        except Exception as exc:
            _phase("error")
            STATE["error"] = f"{type(exc).__name__}: {exc}"
            log("load failed:\n" + traceback.format_exc())
        finally:
            _DOWNLOAD.clear()


def decode_jpeg(data):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


def act(body):
    import numpy as np
    from lerobot.policies.utils import prepare_observation_for_inference

    with LOCK:
        model = dict(MODEL)
    torch = model["torch"]
    images = body["images"]
    # Camera names: as sent when the model knows them, otherwise paired in
    # sorted order (a fine-tuned SmolVLA calls them camera1, camera2, ...).
    sent = sorted(images)
    obs = {}
    for i, key in enumerate(model["image_keys"]):
        name = key.split(".")[-1]
        source = name if name in images else (sent[i] if i < len(sent) else None)
        if source is not None:
            obs[key] = decode_jpeg(images[source])
    obs["observation.state"] = np.asarray(body["state"], dtype=np.float32)

    started = time.time()
    with torch.inference_mode():
        batch = prepare_observation_for_inference(obs, torch.device(DEVICE), body.get("task", ""), "so_follower")
        batch = model["pre"](batch)
        chunk = model["policy"].predict_action_chunk(batch)  # (1, steps, dims)
        chunk = model["post"](chunk)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    actions = chunk.squeeze(0).float().cpu().tolist()
    return {"actions": actions, "names": model["names"], "infer_ms": round((time.time() - started) * 1000, 1)}


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    def do_GET(self):
        health = dict(STATE)
        health["phase_elapsed"] = round(time.time() - STATE["phase_started"], 1)
        if STATE["phase"] == "downloading":
            health["download"] = _download_progress()
        self._send(200, health)

    def do_POST(self):
        if self.path.startswith("/load"):
            try:
                body = self._body()
                repo, sub = body["repo"], body.get("subfolder", "")
            except Exception as exc:
                self._send(400, {"error": f"bad request: {exc}"})
                return
            if STATE["model"] == f"{repo}@{sub}" and (STATE["ready"] or _LOADING.locked()):
                self._send(200, {"ok": True, "already": True})
                return
            LAST_USE[0] = time.time()
            threading.Thread(target=load, args=(repo, sub), daemon=True).start()
            self._send(200, {"ok": True})
            return
        if not STATE["ready"]:
            self._send(503, {"error": STATE["error"] or "Model is still loading."})
            return
        LAST_USE[0] = time.time()
        try:
            self._send(200, act(self._body()))
        except Exception as exc:
            log("act failed:\n" + traceback.format_exc())
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args):
        pass


def idle_watch():
    idle_s = float(os.environ.get("IDLE_S", "600"))
    failed_at = None
    while True:
        time.sleep(10)
        if STATE["error"] and not _LOADING.locked():
            # Keep answering /health for a few minutes so the error can be
            # read (and another model loaded instead).
            failed_at = failed_at or time.time()
            if time.time() - failed_at > 300:
                log("stopping after load failure: " + STATE["error"])
                os._exit(1)
        else:
            failed_at = None
            if STATE["ready"] and time.time() - LAST_USE[0] > idle_s:
                log(f"stopping: no requests for {idle_s:.0f}s")
                os._exit(0)


if __name__ == "__main__":
    threading.Thread(
        target=load, args=(os.environ.get("MODEL_REPO", ""), os.environ.get("MODEL_SUBFOLDER", "").strip("/")), daemon=True
    ).start()
    threading.Thread(target=idle_watch, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    log(f"listening on {port}")
    # HOST=127.0.0.1 for testing on a PC (no firewall prompt).
    http.server.ThreadingHTTPServer((os.environ.get("HOST", "0.0.0.0"), port), Handler).serve_forever()
