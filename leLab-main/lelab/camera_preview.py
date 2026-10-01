"""Low-res live preview of a camera, served by LeLab itself.

The K12 frontend normally previews cameras in the browser, but Chrome on
Windows sometimes lists only one of two identical USB cameras (OpenCV via
DirectShow still sees both). For a camera the browser can't see, the page
points an <img> at /camera-preview/{index} instead.

Only one program can hold a camera on Windows, so every preview is shut down
(and its camera released) before recording or a policy run opens the
cameras -- see ``stop_all_previews``.
"""

import logging
import threading
import time
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_FPS = 15
_lock = threading.Lock()
# Per camera index: a generation number. Starting a new preview of an index,
# or stop_all_previews(), bumps it and the running generator for the old
# generation exits and releases the camera.
_generation: dict[int, int] = {}
# Indices whose capture is currently open.
_open: set[int] = set()


def _wait_released(indices: set[int], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with _lock:
            if not (_open & indices):
                return
        time.sleep(0.05)
    logger.warning("camera preview: %s still open after %.1fs", sorted(_open & indices), timeout_s)


def stop_all_previews(timeout_s: float = 2.0) -> None:
    """Stop every preview and wait for their cameras to be released."""
    with _lock:
        indices = set(_open)
        for index in list(_generation):
            _generation[index] += 1
    if indices:
        _wait_released(indices, timeout_s)


def preview_frames(index: int, is_blocked) -> Iterator[bytes]:
    """MJPEG frames from camera `index` until superseded, stopped, or
    `is_blocked()` (recording / a policy run took over) turns true."""
    import cv2

    with _lock:
        generation = _generation.get(index, 0) + 1
        _generation[index] = generation
    # A previous preview of this index (e.g. the page re-rendered) must let
    # go first, or this open fails.
    _wait_released({index}, 2.0)

    # DirectShow needs COM on this (worker-pool) thread, or the open fails
    # with "backend is generally available but can't be used to capture by
    # index". Left initialized -- pool threads are reused.
    try:
        import ctypes

        ctypes.windll.ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
    except Exception:
        pass
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    with _lock:
        _open.add(index)
    try:
        if not cap.isOpened():
            logger.warning("camera preview: could not open camera %s", index)
            return
        # Small + MJPG: two USB cameras on one controller can't both stream
        # full-size uncompressed video.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        interval = 1.0 / _FPS
        while True:
            with _lock:
                if _generation.get(index) != generation:
                    break
            if is_blocked():
                break
            ok, frame = cap.read()
            if not ok:
                time.sleep(interval)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(interval)
    finally:
        cap.release()
        with _lock:
            _open.discard(index)
