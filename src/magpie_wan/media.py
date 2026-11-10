"""Small, offline media validation helpers shared by render and provider clients."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class InvalidVideoError(ValueError):
    """Raised when a video artifact is missing, incomplete, or unreadable."""


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_video_file(path, require_audio=False):
    """Use ffprobe to verify a readable video stream and a positive duration."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise InvalidVideoError(f"video file is missing or empty: {path}")

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise InvalidVideoError("ffprobe is required to validate video output")

    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,duration:format=duration",
             "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise InvalidVideoError(f"could not run ffprobe: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffprobe failed"
        raise InvalidVideoError(f"invalid video {path}: {detail}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise InvalidVideoError(f"ffprobe returned invalid JSON for {path}") from exc

    streams = data.get("streams") or []
    kinds = {stream.get("codec_type") for stream in streams}
    if "video" not in kinds:
        raise InvalidVideoError(f"no video stream in {path}")
    if require_audio and "audio" not in kinds:
        raise InvalidVideoError(f"no audio stream in {path}")

    duration = (data.get("format") or {}).get("duration")
    if duration is None:
        durations = [stream.get("duration") for stream in streams if stream.get("duration")]
        duration = max(durations, key=float) if durations else None
    try:
        seconds = float(duration)
    except (TypeError, ValueError):
        seconds = 0
    if not math.isfinite(seconds) or seconds <= 0:
        raise InvalidVideoError(f"video duration is missing or invalid in {path}")
    return {"duration": seconds, "streams": kinds}


def save_video_bytes(content, destination, require_audio=False):
    """Write bytes to a temporary file, validate them, then atomically publish."""
    if not content:
        raise InvalidVideoError("downloaded video is empty")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".partial.mp4",
                                         delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        validate_video_file(tmp_path, require_audio=require_audio)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return destination
