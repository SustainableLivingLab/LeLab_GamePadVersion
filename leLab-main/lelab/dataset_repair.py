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

"""Rebuild the episode index of a recording that was never finalized.

``LeRobotDataset.finalize()`` is what writes ``meta/episodes/`` and the parquet
footers; frames, videos and ``meta/info.json`` are written as the session runs.
A recording interrupted before finalize therefore leaves a directory that looks
complete but carries no episode index, and ``LeRobotDataset`` reads that as
"not available locally" and tries to download it from the Hub — a 404 for a
dataset that was never pushed.

Recovery is bounded by the parquet footer: a data file the writer was still
appending to when it died cannot be read at all, so its frames and every frame
after it are gone.
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHUNK_FILE_RE = re.compile(r"chunk-(\d+)/file-(\d+)\.")


class DatasetRepairError(Exception):
    """The dataset is damaged beyond what can be rebuilt from disk."""


def _chunk_file(path: Path) -> tuple[int, int]:
    match = _CHUNK_FILE_RE.search(path.as_posix())
    if match is None:
        raise DatasetRepairError(f"Unexpected dataset file layout: {path}")
    return int(match.group(1)), int(match.group(2))


def _video_files(root: Path, video_key: str) -> list[tuple[int, int, float]]:
    """Return (chunk_index, file_index, duration_s) per encoded video file."""
    import av

    files = []
    for path in sorted((root / "videos" / video_key).rglob("*.mp4"), key=_chunk_file):
        try:
            with av.open(str(path)) as container:
                duration = float(container.duration) / av.time_base if container.duration else 0.0
        except Exception as e:
            # An unclosed mp4 has no moov atom; treat it as carrying no episodes.
            logger.warning("Could not read %s (%s) — episodes in it are unrecoverable", path, e)
            duration = 0.0
        chunk_index, file_index = _chunk_file(path)
        files.append((chunk_index, file_index, duration))
    return files


def _episodes_from_data(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Reconstruct one row per episode from the readable data files.

    Also returns the trailing files that have no parquet footer — the reader
    globs every file under ``data/``, so they have to be moved out of the way.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    # Written with the first saved episode, so its absence means the session
    # died before one completed.
    if not (root / "meta" / "tasks.parquet").exists():
        return [], []

    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_by_index = {int(index): task for task, index in tasks["task_index"].items()}

    data_files = sorted((root / "data").rglob("*.parquet"), key=_chunk_file)
    rows: list[dict[str, Any]] = []
    next_index = 0
    for position, path in enumerate(data_files):
        try:
            frames = pq.read_table(path, columns=["episode_index", "task_index"]).to_pandas()
        except Exception as e:
            # No footer: the writer died mid-file. Nothing from here on is
            # readable, since only the newest file is ever left open.
            logger.warning("Data file %s is truncated (%s) — recovering the episodes before it", path, e)
            return rows, data_files[position:]

        chunk_index, file_index = _chunk_file(path)
        for episode_index, group in frames.groupby("episode_index", sort=True):
            length = len(group)
            rows.append(
                {
                    "episode_index": int(episode_index),
                    "tasks": [task_by_index[int(i)] for i in sorted(group["task_index"].unique())],
                    "length": length,
                    "data/chunk_index": chunk_index,
                    "data/file_index": file_index,
                    "dataset_from_index": next_index,
                    "dataset_to_index": next_index + length,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
            next_index += length
    return rows, []


def _assign_videos(rows: list[dict[str, Any]], root: Path, video_key: str, fps: int) -> int:
    """Fill in the video columns for `video_key`, in place.

    Episodes are concatenated into a video file until it hits the size limit,
    so each episode's window is its predecessors' durations within that file.
    Returns how many leading rows are actually covered by encoded video.
    """
    files = _video_files(root, video_key)
    cursor = 0
    offset = 0.0
    tolerance = 1.0 / fps

    for covered, row in enumerate(rows):
        duration = row["length"] / fps
        while cursor < len(files) and offset + duration > files[cursor][2] + tolerance:
            cursor += 1
            offset = 0.0
        if cursor >= len(files):
            return covered

        chunk_index, file_index, _ = files[cursor]
        row[f"videos/{video_key}/chunk_index"] = chunk_index
        row[f"videos/{video_key}/file_index"] = file_index
        row[f"videos/{video_key}/from_timestamp"] = offset
        row[f"videos/{video_key}/to_timestamp"] = offset + duration
        offset += duration

    return len(rows)


def _trim_data(root: Path, keep_through: int) -> int:
    """Drop data rows belonging to episodes past `keep_through`.

    The reader loads every row under data/ and push_to_hub uploads the whole
    directory, so rows the rebuilt index no longer references would otherwise
    travel with the dataset. Returns the number of frames dropped.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    dropped = 0
    for path in sorted((root / "data").rglob("*.parquet"), key=_chunk_file):
        table = pq.read_table(path)
        kept = table.filter(pc.less_equal(table["episode_index"], keep_through))
        if kept.num_rows == table.num_rows:
            continue
        dropped += table.num_rows - kept.num_rows
        if kept.num_rows == 0:
            path.unlink()
        else:
            pq.write_table(kept, path, compression="snappy", use_dictionary=True)
    return dropped


def _episode_video_frames(root: Path, video_key: str, row: dict[str, Any]) -> "np.ndarray":  # noqa: F821
    """Decode a sample of the frames one episode references, as (N, C, H, W) in [0, 1]."""
    import av
    import numpy as np

    from lerobot.datasets.compute_stats import sample_indices

    file_index = row[f"videos/{video_key}/file_index"]
    start = row[f"videos/{video_key}/from_timestamp"]
    end = row[f"videos/{video_key}/to_timestamp"]

    path = next((root / "videos" / video_key).rglob(f"*file-{file_index:03d}.mp4"))
    with av.open(str(path)) as container:
        frames = [
            frame
            for frame in container.decode(container.streams.video[0])
            if start <= float(frame.time) < end
        ]

    if not frames:
        raise DatasetRepairError(f"No decodable frames for {video_key} in episode {row['episode_index']}")

    sampled = [frames[i] for i in sample_indices(len(frames))]
    return np.stack([frame.to_ndarray(format="rgb24").transpose(2, 0, 1) for frame in sampled]) / 255.0


def _recompute_stats(root: Path, info: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Rewrite meta/stats.json over the retained frames only.

    stats.json is aggregated as each episode is saved, so it still describes
    episodes that were lost with the interruption, and training normalizes with
    it. Rebuilt the way recording builds it, per episode then aggregated, so the
    numbers match what the same episodes would have produced on their own.
    """
    import numpy as np
    import pyarrow.parquet as pq

    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.io_utils import write_stats

    columns: dict[str, np.ndarray] = {}  # noqa: F821
    for path in sorted((root / "data").rglob("*.parquet"), key=_chunk_file):
        table = pq.read_table(path)
        for key in table.schema.names:
            values = np.stack(table[key].to_numpy(zero_copy_only=False).tolist())
            columns[key] = np.concatenate([columns[key], values]) if key in columns else values

    per_episode = []
    for row in rows:
        episode_stats = {}
        for key, feature in info["features"].items():
            if feature["dtype"] in {"string", "language"}:
                continue

            if feature["dtype"] in {"image", "video"}:
                values = _episode_video_frames(root, key, row)
                episode_stats[key] = {
                    name: value if name == "count" else np.squeeze(value, axis=0)
                    for name, value in get_feature_stats(values, axis=(0, 2, 3), keepdims=True).items()
                }
            elif key in columns:
                values = columns[key][row["dataset_from_index"] : row["dataset_to_index"]]
                episode_stats[key] = get_feature_stats(values, axis=0, keepdims=values.ndim == 1)
        per_episode.append(episode_stats)

    write_stats(aggregate_stats(per_episode), root)


def _real_video_frame_count(path: Path) -> int:
    """Actual decodable frame count of an mp4 -- container metadata (``stream.frames``)
    is frequently 0 or wrong for files this app encodes, so decode for real."""
    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


def _clamp_row_video_windows(
    row: dict[str, Any],
    root: Path,
    info: dict[str, Any],
    video_keys: list[str],
    frame_counts: dict[tuple[str, int], int],
) -> int:
    """Clamp one episode row's video windows in place. Returns how many were clamped."""
    fps = info["fps"]
    clamped = 0
    for video_key in video_keys:
        to_key = f"videos/{video_key}/to_timestamp"
        from_key = f"videos/{video_key}/from_timestamp"
        file_index = row.get(f"videos/{video_key}/file_index")
        if file_index is None or to_key not in row or row[to_key] is None:
            continue

        cache_key = (video_key, int(file_index))
        if cache_key not in frame_counts:
            path = next((root / "videos" / video_key).rglob(f"*file-{file_index:03d}.mp4"), None)
            frame_counts[cache_key] = _real_video_frame_count(path) if path else 0
        num_frames = frame_counts[cache_key]

        last_frame_needed = round(row[to_key] * fps) - 1
        if last_frame_needed >= num_frames:
            safe_to_timestamp = num_frames / fps
            if safe_to_timestamp <= (row.get(from_key) or 0.0):
                # Nothing usable left in this file for this episode/key;
                # leave it be rather than produce a negative-length window.
                continue
            logger.warning(
                "%s episode %s video %s: to_timestamp %.4fs needs frame %d but file-%03d.mp4 "
                "only has %d real frames -- clamping to %.4fs",
                root.name,
                row.get("episode_index"),
                video_key,
                row[to_key],
                last_frame_needed,
                file_index,
                num_frames,
                safe_to_timestamp,
            )
            row[to_key] = safe_to_timestamp
            clamped += 1
    return clamped


def fix_video_window_overruns_in_place(root: Path, info: dict[str, Any]) -> int:
    """Clamp video windows that ask for frames past what the file has, rewriting
    each existing meta/episodes/*.parquet file in place (every other column,
    including stats/*, passes through untouched).

    Continue Recording restarts camera capture per key independently, so two
    cameras recording the "same" episode can end up a frame or two apart --
    one video_key's file ends up shorter than the episode's to_timestamp
    implies. The dataloader then indexes past the end of that video and
    crashes (IndexError from torchcodec) partway through training, often long
    after the dataset was already pushed to the Hub. Clamping every video
    key's to_timestamp to the tightest real duration available keeps all
    cameras requesting frames that actually exist, at the cost of at most a
    couple of frames of trailing video per affected episode.

    Returns how many (episode, video_key) windows were clamped across the
    whole dataset.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    if not video_keys:
        return 0

    frame_counts: dict[tuple[str, int], int] = {}
    total_clamped = 0
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet"), key=_chunk_file):
        table = pq.read_table(path)
        rows = table.to_pylist()
        file_clamped = sum(
            _clamp_row_video_windows(row, root, info, video_keys, frame_counts) for row in rows
        )
        if file_clamped:
            pq.write_table(
                pa.Table.from_pylist(rows, schema=table.schema),
                path,
                compression="snappy",
                use_dictionary=True,
            )
        total_clamped += file_clamped
    return total_clamped


def _write_episodes(root: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    episodes_dir = root / "meta" / "episodes"
    shutil.rmtree(episodes_dir, ignore_errors=True)
    path = episodes_dir / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy", use_dictionary=True)


def repair_local_dataset(repo_id: str) -> str | None:
    """Rebuild `repo_id`'s episode index when the recording never finalized.

    Returns a description of what was recovered, or None when the dataset is
    absent or already readable. Raises DatasetRepairError when nothing is left
    to recover.
    """
    from lerobot.datasets.io_utils import load_episodes
    from lerobot.utils.constants import HF_LEROBOT_HOME

    cache_root = Path(HF_LEROBOT_HOME).resolve()
    root = (cache_root / repo_id).resolve()

    # repo_id reaches here straight from the request body, and this function
    # rewrites info.json and renames files. Keep it inside the cache.
    if root == cache_root or cache_root not in root.parents:
        raise DatasetRepairError(f"Invalid dataset id: {repo_id}")

    if not (root / "meta" / "info.json").exists():
        return None

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))

    try:
        episodes_df = load_episodes(root)
    except Exception as e:
        logger.info("%s has no usable episode index (%s) — rebuilding it from disk", repo_id, e)
        episodes_df = None

    if episodes_df is not None:
        # The index loaded fine, but it can still record a video window that
        # asks for more frames than a camera's file actually has (see
        # fix_video_window_overruns_in_place). That's a silent corruption
        # load_episodes can't detect on its own, so always check it even on
        # the happy path.
        clamped = fix_video_window_overruns_in_place(root, info)
        if clamped == 0:
            return None
        message = f"Clamped {clamped} video window(s) in {repo_id} that pointed past the end of a camera's recording"
        logger.info(message)
        return message

    fps = info["fps"]
    recorded_episodes = info.get("total_episodes", 0)

    rows, truncated = _episodes_from_data(root)
    for video_key, feature in info["features"].items():
        if feature["dtype"] == "video":
            del rows[_assign_videos(rows, root, video_key, fps) :]

    if not rows:
        raise DatasetRepairError(
            f"{repo_id} was interrupted before any episode was fully written to disk, "
            "and the partial files it left behind cannot be read. Re-record it."
        )

    # The reader loads every parquet file under data/, so an unreadable one
    # would still break it. Keep the bytes, just out of the glob's reach.
    for path in truncated:
        path.rename(path.with_suffix(".parquet.unreadable"))

    lost = recorded_episodes - len(rows)
    if lost > 0:
        # Episodes the videos don't cover still have their frames in the
        # readable shards; drop them so data, index and stats agree.
        _trim_data(root, rows[-1]["episode_index"])
        _recompute_stats(root, info, rows)

    # _assign_videos trusts each episode's row count (from data/) to compute
    # to_timestamp; a camera that recorded one fewer real frame than its row
    # count implies needs the same clamp as the happy path above.
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    frame_counts: dict[tuple[str, int], int] = {}
    for row in rows:
        _clamp_row_video_windows(row, root, info, video_keys, frame_counts)

    _write_episodes(root, rows)

    info["total_episodes"] = len(rows)
    info["total_frames"] = sum(row["length"] for row in rows)
    info["splits"] = {"train": f"0:{len(rows)}"}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    message = f"Recovered {len(rows)} episode(s) from an interrupted recording of {repo_id}"
    if lost > 0:
        message += f"; {lost} episode(s) were lost because the recording was cut short"
    logger.info(message)
    return message
