"""Finding Lesson 6's coloured cubes in the top camera's picture.

Lesson 6 sorts two cubes (e.g. white and orange) with one AI that takes a
task sentence per cube. While the AI runs, CubeWatcher looks at the table a
few times a second, finds the cubes between the baskets and, once it clearly
sees one there, switches the run's task to that cube's sentence, so the
student only has to put a cube down.

Colours are learned by example, not hard-coded (lighting changes how a colour
looks): the student clicks each cube, then each basket, in the camera
preview, and sample() reads the colour there, the table's around it and the
size and outline of the patch of that colour. Each look then gives every
pixel of the search area to the closest of those colours (table or a cube),
in Lab colour space, where "closest" matches how different colours look,
with lightness counting less than hue so dimmer or brighter lighting doesn't
swap one cube for another. Patches about the size of a cube are cubes.

The search area is a circle around the middle spot (where the cubes were
clicked) reaching most of the way to the baskets, minus the baskets' own
outlines: the baskets are the same colours as the cubes, and a cube that is
already in its basket is done, not a new job.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lelab.utils import live_task

logger = logging.getLogger(__name__)

# Set by LeLab for an AI run when the camera should pick the task: the
# watcher's settings, and where to report what it sees (a JSON file; the
# camera picture it looked at goes next to it as .jpg, so the app can show the
# picture and the cubes found while the run holds the camera).
CUBE_WATCH_ENV = "LELAB_CUBE_WATCH"
CUBE_SEEN_ENV = "LELAB_CUBE_SEEN"
# Width of that picture: the app's camera box is small.
SEEN_PICTURE_WIDTH = 320

# Sizes as fractions of the picture's width. A clicked colour is read in a
# small patch where it was clicked, the table's in a ring well outside it.
CUBE_PATCH = 0.02
TABLE_RING = (0.10, 0.13)
# Looks work on a small copy of the picture: fast enough for a few a second.
WORK_WIDTH = 160
# The search circle reaches this share of the way from the middle spot to
# the baskets (or has this radius, as a share of the width, if the baskets
# weren't taught).
SEARCH_REACH = 0.8
FALLBACK_SEARCH_RADIUS = 0.15
# Basket outlines are grown by this share of the width, so a basket's rim
# and shadow aren't taken for a cube.
BASKET_MARGIN = 0.02
# A patch counts as a cube when its size is between these multiples of the
# taught cube's size (a cube nearer the camera looks bigger, a hand-held one
# may be partly hidden). Without a taught size, this share of the picture.
SIZE_RANGE = (0.3, 3.0)
FALLBACK_CUBE_AREA = 0.004
# A pixel is a cube's colour only if it is this close to it, as a share of
# the gap between that colour and the nearest other taught colour; things
# that are neither cube nor table (a hand, a sleeve) then match nothing.
COLOR_MARGIN = 0.6
# Looks in a row that must agree before the task changes (a few per second),
# so a hand or the claw passing over the table doesn't flip it. A cube the
# claw holds is hidden under it, so the task doesn't flip mid-pick either.
STABLE_LOOKS = 3
# Weight of lightness against hue (Lab's L against a, b) when comparing
# colours: lighting mostly changes how light a colour looks, so a white cube
# in a dim room should still be nearer white than orange.
LIGHTNESS_WEIGHT = 0.35
_WEIGHTS = np.array([LIGHTNESS_WEIGHT, 1.0, 1.0], dtype=np.float32)


def _median_rgb(pixels: np.ndarray) -> list[int]:
    return [int(v) for v in np.median(pixels.reshape(-1, 3), axis=0)]


def _circle_mask(h: int, w: int, cx: float, cy: float, r_min: float, r_max: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy)
    return (d >= r_min) & (d <= r_max)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """Weighted Lab of RGB pixels (any shape ending in 3)."""
    shape = rgb.shape
    flat = np.ascontiguousarray(rgb.reshape(1, -1, 3), dtype=np.uint8)
    return (cv2.cvtColor(flat, cv2.COLOR_RGB2LAB)[0].astype(np.float32) * _WEIGHTS).reshape(shape)


def _small(frame_rgb: np.ndarray) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if w <= WORK_WIDTH:
        return frame_rgb
    return cv2.resize(frame_rgb, (WORK_WIDTH, round(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)


def _clean(mask: np.ndarray) -> np.ndarray:
    """Drop specks and fill pinholes in a patch mask."""
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def sample(frame_bgr: np.ndarray, x: float, y: float) -> dict[str, Any]:
    """What was clicked at (x, y) (fractions of width / height): its colour
    and the table's around it (RGB), and the patch of that colour there: its
    size (share of the picture) and outline (points as fractions), or None
    when no patch stands out from the table."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cx, cy = x * w, y * h
    patch = rgb[_circle_mask(h, w, cx, cy, 0, max(2.0, CUBE_PATCH * w))]
    table = rgb[_circle_mask(h, w, cx, cy, TABLE_RING[0] * w, TABLE_RING[1] * w)]
    if patch.size == 0 or table.size == 0:
        raise ValueError("That spot is too close to the edge of the picture.")
    color, table_color = _median_rgb(patch), _median_rgb(table)
    out: dict[str, Any] = {"cube": color, "table": table_color, "area": 0.0, "outline": None}

    small = _small(rgb)
    sh, sw = small.shape[:2]
    refs = _lab(np.array([color, table_color], dtype=np.uint8))
    gap = float(np.linalg.norm(refs[0] - refs[1]))
    dist = np.linalg.norm(_lab(small) - refs[0], axis=2)
    mask = _clean(dist < max(1.0, 0.5 * gap))
    count, labels = cv2.connectedComponents(mask)
    # The patch under the click, or the one nearest it within the read patch.
    px, py = min(sw - 1, int(x * sw)), min(sh - 1, int(y * sh))
    near = labels[_circle_mask(sh, sw, px, py, 0, max(1.5, CUBE_PATCH * sw * 1.5))]
    near = near[near > 0]
    if count <= 1 or near.size == 0:
        return out
    label = int(np.bincount(near).argmax())
    ys, xs = np.nonzero(labels == label)
    out["area"] = float(len(xs)) / (sw * sh)
    hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))[:, 0, :]
    out["outline"] = [[round(float(px_) / sw, 4), round(float(py_) / sh, 4)] for px_, py_ in hull]
    return out


class CubeWatcher:
    """Finds the cubes between the baskets and decides when to switch tasks.

    config (from the app): {"camera": "top", "x": 0.5, "y": 0.6,
    "table": [r, g, b], "cubes": [{"name": "white", "rgb": [r, g, b],
    "area": 0.006, "task": "..."}, ...], "baskets": [{"x": 0.2, "y": 0.5,
    "outline": [[x, y], ...]}, ...]}. x / y are fractions of the picture,
    area a share of it; area and baskets may be missing.
    """

    def __init__(self, config: dict[str, Any], current_task: str):
        self.camera = config.get("camera", "top")
        self.x = float(config["x"])
        self.y = float(config["y"])
        self.cubes = [c for c in config["cubes"] if c.get("rgb")]
        if not self.cubes:
            raise ValueError("no cube colours to look for")
        self.baskets = [b for b in config.get("baskets") or [] if "x" in b and "y" in b]
        # Reference colours: table first, then each cube.
        self._refs = _lab(np.array([config["table"], *[c["rgb"] for c in self.cubes]], dtype=np.uint8))
        gaps = np.linalg.norm(self._refs[:, None, :] - self._refs[None, :, :], axis=2)
        np.fill_diagonal(gaps, np.inf)
        self._max_dist = COLOR_MARGIN * gaps.min(axis=1)
        self._search: np.ndarray | None = None
        self.seen: str | None = None
        self.found: list[dict[str, Any]] = []
        self.task = current_task
        self._streak: tuple[str | None, int] = (None, 0)
        # False until the camera has first seen a cube.
        self._decided = False

    def search_radius(self, aspect: float) -> float:
        """Search circle radius as a share of the picture's width (aspect =
        height / width, the fractions' y unit against x)."""
        if not self.baskets:
            return FALLBACK_SEARCH_RADIUS
        reach = [np.hypot(b["x"] - self.x, (b["y"] - self.y) * aspect) for b in self.baskets]
        return SEARCH_REACH * float(np.mean(reach))

    def _search_mask(self, h: int, w: int) -> np.ndarray:
        if self._search is None or self._search.shape != (h, w):
            mask = _circle_mask(h, w, self.x * w, self.y * h, 0, self.search_radius(h / w) * w).astype(
                np.uint8
            )
            for basket in self.baskets:
                outline = basket.get("outline")
                if outline and len(outline) >= 3:
                    points = np.array([[px * w, py * h] for px, py in outline], dtype=np.int32)
                    cv2.fillConvexPoly(mask, points, 0)
                    cv2.polylines(mask, [points], True, 0, thickness=max(1, round(2 * BASKET_MARGIN * w)))
            self._search = mask.astype(bool)
        return self._search

    def find(self, frame_rgb: np.ndarray) -> list[dict[str, Any]]:
        """The cube-sized patches of each cube's colour in the search area,
        nearest the middle spot first: name, centre, box (fractions)."""
        small = _small(frame_rgb)
        h, w = small.shape[:2]
        search = self._search_mask(h, w)
        lab = _lab(small[search])
        dist = np.linalg.norm(lab[:, None, :] - self._refs[None, :, :], axis=2)
        nearest = dist.argmin(axis=1)
        close = dist[np.arange(len(nearest)), nearest] < self._max_dist[nearest]
        found = []
        for k, cube in enumerate(self.cubes, start=1):
            mask = np.zeros((h, w), dtype=bool)
            mask[search] = (nearest == k) & close
            count, _, stats, centroids = cv2.connectedComponentsWithStats(_clean(mask))
            size = float(cube.get("area") or FALLBACK_CUBE_AREA) * w * h
            for i in range(1, count):
                left, top, bw, bh, area = (int(v) for v in stats[i])
                if not SIZE_RANGE[0] * size <= area <= SIZE_RANGE[1] * size:
                    continue
                cx, cy = (float(v) for v in centroids[i])
                found.append(
                    {
                        "name": cube["name"],
                        "x": round(cx / w, 4),
                        "y": round(cy / h, 4),
                        "box": [round(left / w, 4), round(top / h, 4), round(bw / w, 4), round(bh / h, 4)],
                        "_d": float(np.hypot(cx / w - self.x, (cy / h - self.y) * h / w)),
                    }
                )
        found.sort(key=lambda f: f["_d"])
        for f in found:
            del f["_d"]
        return found

    def look(self, frame_rgb: np.ndarray) -> str | None:
        """Name of the cube to do now (nearest the middle spot), or None."""
        self.found = self.find(frame_rgb)
        self.seen = self.found[0]["name"] if self.found else None
        return self.seen

    def update(self, frame_rgb: np.ndarray) -> str | None:
        """Look again; returns the new task when it should change, else None."""
        seen = self.look(frame_rgb)
        name, count = self._streak
        if seen != name:
            logger.info("Camera: %s", f"sees the {seen} cube" if seen else "no cube between the baskets")
        self._streak = (seen, count + 1 if seen == name else 1)
        # The very first look counts on its own: waiting for more looks lets
        # the arm head off for the wrong cube.
        needed = 1 if not self._decided else STABLE_LOOKS
        if seen is None or self._streak[1] < needed:
            return None
        self._decided = True
        task = next(c.get("task") for c in self.cubes if c["name"] == seen)
        if not task or task == self.task:
            return None
        self.task = task
        logger.info("Camera sees the %s cube: task is now %r", seen, task)
        return task

    def view(self, aspect: float = 0.75) -> dict[str, Any]:
        """What the app draws over the camera picture."""
        return {
            "seen": self.seen,
            "cubes": self.found,
            "search": {"x": self.x, "y": self.y, "r": round(self.search_radius(aspect), 4)},
        }


def make_watcher(task: str, camera_keys: list[str]) -> CubeWatcher | None:
    """The run's watcher, if LeLab asked for one and its camera is on."""
    path = os.environ.get(CUBE_WATCH_ENV)
    if not path:
        return None
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        watcher = CubeWatcher(config, task)
        if watcher.camera not in camera_keys:
            logger.warning(
                "Cube colours: camera %r isn't on, so the task won't change by itself", watcher.camera
            )
            return None
        logger.info("Cube colours: looking for %s between the baskets", [c["name"] for c in watcher.cubes])
        return watcher
    except Exception as exc:  # noqa: BLE001 - the run works without it
        logger.warning("Cube colours: not watching (%s)", exc)
        return None


def watch(watcher: CubeWatcher, obs: dict) -> None:
    """Look at the run's newest observation: switch the task if the camera
    says so, and report what it saw (and its picture) to LeLab."""
    frame = obs[watcher.camera]
    task_file = Path(os.environ[live_task.ENV_VAR]) if os.environ.get(live_task.ENV_VAR) else None
    # The task may have been switched with the app's buttons since the last
    # look; the camera must still switch it back when it sees the other cube.
    try:
        if task_file is not None:
            watcher.task = task_file.read_text(encoding="utf-8").strip() or watcher.task
    except OSError:
        pass
    task = watcher.update(frame)
    if task and task_file is not None:
        live_task.write(task_file, task)
    seen_path = os.environ.get(CUBE_SEEN_ENV)
    if not seen_path:
        return
    h, w = frame.shape[:2]
    small = cv2.resize(
        frame, (SEEN_PICTURE_WIDTH, round(h * SEEN_PICTURE_WIDTH / w)), interpolation=cv2.INTER_AREA
    )
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
    # Write then rename, so LeLab never reads half a file. On Windows the
    # rename fails while LeLab has the old file open; the next look retries.
    for path, data in (
        (Path(seen_path), json.dumps(watcher.view(h / w)).encode("utf-8")),
        (Path(seen_path).with_suffix(".jpg"), buf.tobytes() if ok else None),
    ):
        if data is None:
            continue
        try:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError:
            pass
