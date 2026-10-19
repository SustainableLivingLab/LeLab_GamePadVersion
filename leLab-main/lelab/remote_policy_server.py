"""Model server for running a trained policy on a Hugging Face Jobs GPU.

This file is sent to the cloud machine as the job's command (it runs in the
`huggingface/lerobot-gpu` image, so it imports only what that image has) and
serves the policy over plain HTTP on PORT, which the job exposes through the
HF jobs proxy (only callers with the account's HF token get through).

  GET  /health  -> {"ready": bool, "error": str|None, ...}
  POST /act     -> body: {"images": {name: base64 JPEG}, "state": [floats],
                          "task": str}
                   reply: {"actions": [[...], ...], "names": [...],
                           "infer_ms": float}

The reply is the whole planned chunk, already un-normalised into robot units
and ordered like the dataset's action names, so the laptop only has to play
it. The process exits after IDLE_S seconds without /act requests (once the
model is ready) so a forgotten session doesn't keep a paid GPU running.

Environment: MODEL_REPO, MODEL_SUBFOLDER (e.g. checkpoints/002000/pretrained_model,
or empty for the repo root), PORT (8000), IDLE_S (600), HF_TOKEN (secret).
MODEL_PATH (a local folder instead of MODEL_REPO) and DEVICE (cuda) are for
testing on a PC.
"""

import base64
import http.server
import io
import json
import os
import threading
import time
import traceback

DEVICE = os.environ.get("DEVICE", "cuda")
STATE = {"ready": False, "error": None, "loaded_in_s": None, "model": None}
LAST_USE = [time.time()]
LOCK = threading.Lock()
MODEL = {}


def log(message):
    print(f"[remote-policy] {message}", flush=True)


def load():
    started = time.time()
    try:
        import torch
        from huggingface_hub import hf_hub_download, snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        repo = os.environ.get("MODEL_REPO", "")
        sub = os.environ.get("MODEL_SUBFOLDER", "").strip("/")
        if os.environ.get("MODEL_PATH"):
            path = os.environ["MODEL_PATH"]
        else:
            local = snapshot_download(repo, allow_patterns=[f"{sub}/*"] if sub else None)
            path = os.path.join(local, sub) if sub else local
            log(f"downloaded {repo}/{sub} in {time.time() - started:.0f}s")

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
            info = hf_hub_download(dataset_repo, "meta/info.json", repo_type="dataset")
            with open(info) as fh:
                names = json.load(fh)["features"]["action"]["names"]
        except Exception as exc:  # the client can still play actions by position
            log(f"could not read action names: {exc}")

        image_keys = sorted(k for k, f in cfg.input_features.items() if str(f.type).endswith("VISUAL"))
        MODEL.update(policy=policy, pre=pre, post=post, names=names, image_keys=image_keys, torch=torch)
        STATE.update(ready=True, loaded_in_s=round(time.time() - started, 1), model=f"{cfg.type} {repo}/{sub}")
        log(f"ready in {STATE['loaded_in_s']}s: {cfg.type}, cameras {image_keys}")
    except Exception as exc:
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        log("load failed:\n" + traceback.format_exc())


def decode_jpeg(data):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


def act(body):
    import numpy as np
    from lerobot.policies.utils import prepare_observation_for_inference

    torch = MODEL["torch"]
    images = body["images"]
    # Camera names: as sent when the model knows them, otherwise paired in
    # sorted order (a fine-tuned SmolVLA calls them camera1, camera2, ...).
    sent = sorted(images)
    obs = {}
    for i, key in enumerate(MODEL["image_keys"]):
        name = key.split(".")[-1]
        source = name if name in images else (sent[i] if i < len(sent) else None)
        if source is not None:
            obs[key] = decode_jpeg(images[source])
    obs["observation.state"] = np.asarray(body["state"], dtype=np.float32)

    started = time.time()
    with LOCK, torch.inference_mode():
        batch = prepare_observation_for_inference(obs, torch.device(DEVICE), body.get("task", ""), "so_follower")
        batch = MODEL["pre"](batch)
        chunk = MODEL["policy"].predict_action_chunk(batch)  # (1, steps, dims)
        chunk = MODEL["post"](chunk)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
    actions = chunk.squeeze(0).float().cpu().tolist()
    return {"actions": actions, "names": MODEL["names"], "infer_ms": round((time.time() - started) * 1000, 1)}


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, STATE)

    def do_POST(self):
        if not STATE["ready"]:
            self._send(503, {"error": STATE["error"] or "Model is still loading."})
            return
        LAST_USE[0] = time.time()
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self._send(200, act(body))
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
        if STATE["error"]:
            # Keep answering /health for a minute so the error can be read.
            failed_at = failed_at or time.time()
            if time.time() - failed_at > 60:
                log("stopping after load failure: " + STATE["error"])
                os._exit(1)
        elif STATE["ready"] and time.time() - LAST_USE[0] > idle_s:
            log(f"stopping: no requests for {idle_s:.0f}s")
            os._exit(0)


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    threading.Thread(target=idle_watch, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    log(f"listening on {port}")
    http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
