"""Low-res live preview of a camera, served by LeLab itself.

The K12 frontend's Lesson 5 shows its camera previews through this instead of
the browser: two identical USB cameras behind one USB 2.0 hub can't both
stream to the browser (it only gets uncompressed video), while OpenCV can ask
for compressed MJPG, as recording does, and both fit. The preview also opens
exactly the camera index recording will use.

Only one program can hold a camera on Windows, so every preview must let go
before recording or a policy run opens the cameras -- see
``stop_all_previews``. Each camera is read by its own background thread
(``_Reader``) rather than inside the HTTP response generator: a generator only
runs while the browser keeps pulling frames, so when the page stopped
watching (e.g. the preview box hid itself for an AI run) the generator sat
paused with the camera still open and nothing could tell it to stop -- the
policy run then hung opening that camera. A thread can always be told to
stop, and stops on its own once nobody has pulled a frame for a moment.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

_FPS = 15
# A reader nobody has pulled a frame from for this long releases its camera.
_IDLE_STOP_S = 2.0
_lock = threading.Lock()
# Cameras are opened one at a time (see _open_camera).
_open_lock = threading.Lock()
# A camera released a moment ago (page refresh, the other preview, a
# recording that just ended) often can't be opened again straight away on
# Windows; try a few times before reporting it as unavailable.
_OPEN_ATTEMPTS = 5
_OPEN_RETRY_S = 0.5
# Live reader per camera index.
_readers: dict[int, _Reader] = {}


class _Reader:
    """Owns one camera: opens it, keeps the latest JPEG, releases it on stop."""

    def __init__(self, index: int, is_blocked: Callable[[], bool]) -> None:
        self.index = index
        self.is_blocked = is_blocked
        self.stop_event = threading.Event()
        self.released = threading.Event()
        self.frame: bytes | None = None
        self.frame_seq = 0
        self.last_pull = time.monotonic()
        self.failed = False
        self.thread = threading.Thread(target=self._run, name=f"camera-preview-{index}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _open_camera(self, cv2):
        """Open the camera small + MJPG (two USB cameras on one hub can't
        both stream full-size uncompressed video), retrying while it's still
        being released elsewhere. Opens are serialized: two cameras on one
        hub opening at the same instant is when one of them tends to fail."""
        for attempt in range(1, _OPEN_ATTEMPTS + 1):
            if self.stop_event.is_set() or self.is_blocked():
                return None
            with _open_lock:
                cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    if attempt > 1:
                        logger.info("camera preview: camera %s opened on attempt %d", self.index, attempt)
                    return cap
                cap.release()
            self.stop_event.wait(_OPEN_RETRY_S)
        return None

    def _run(self) -> None:
        import cv2

        cap = None
        try:
            # DirectShow needs COM on this thread, or the open fails with
            # "backend is generally available but can't be used to capture by
            # index".
            try:
                import ctypes

                ctypes.windll.ole32.CoInitializeEx(None, 0x2)  # COINIT_APARTMENTTHREADED
            except Exception:
                pass
            cap = self._open_camera(cv2)
            if cap is None:
                logger.warning("camera preview: could not open camera %s", self.index)
                self.failed = True
                return
            interval = 1.0 / _FPS
            while not self.stop_event.is_set():
                if self.is_blocked() or time.monotonic() - self.last_pull > _IDLE_STOP_S:
                    break
                ok, frame = cap.read()
                if ok:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        self.frame = buf.tobytes()
                        self.frame_seq += 1
                self.stop_event.wait(interval)
        except Exception as exc:
            logger.warning("camera preview %s stopped: %s", self.index, exc)
            self.failed = True
        finally:
            if cap is not None:
                cap.release()
            with _lock:
                if _readers.get(self.index) is self:
                    _readers.pop(self.index, None)
            self.released.set()


def stop_all_previews(timeout_s: float = 3.0, settle_s: float = 1.0) -> None:
    """Stop every preview and wait until their cameras are released, plus
    `settle_s`: on Windows a camera closed a moment ago often can't be opened
    again straight away, and recording/rollout open them right after this."""
    with _lock:
        readers = list(_readers.values())
    for reader in readers:
        reader.stop()
    deadline = time.monotonic() + timeout_s
    for reader in readers:
        if not reader.released.wait(max(0.0, deadline - time.monotonic())):
            logger.warning("camera preview: camera %s still open after %.1fs", reader.index, timeout_s)
    if readers:
        time.sleep(settle_s)


def preview_frames(index: int, is_blocked: Callable[[], bool]) -> Iterator[bytes]:
    """MJPEG stream of camera `index` until the reader stops (superseded,
    stopped for a recording/policy run, or `is_blocked()`)."""
    with _lock:
        old = _readers.get(index)
    if old is not None:
        # A previous preview of this index (the page re-rendered or retried)
        # must let go first, or this open fails.
        old.stop()
        old.released.wait(3.0)
    reader = _Reader(index, is_blocked)
    with _lock:
        _readers[index] = reader
    reader.start()

    sent_seq = 0
    while True:
        reader.last_pull = time.monotonic()
        if reader.frame_seq != sent_seq and reader.frame is not None:
            sent_seq = reader.frame_seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + reader.frame + b"\r\n"
        elif reader.released.is_set():
            return
        else:
            time.sleep(0.02)
