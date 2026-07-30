"""Video decoding, normalised to what the visual front end expects.

Arbitrary video is messy: variable frame rate, rotation metadata, exotic codecs,
odd pixel formats. Rather than handle each case downstream, everything is pushed
through ffmpeg once and comes out as a plain RGB array at a fixed frame rate.

ffmpeg comes from the ``imageio-ffmpeg`` wheel, so a system install is optional.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from pathlib import Path

import numpy as np

from .constants import MODEL_FPS


class VideoError(RuntimeError):
    """Raised when a video cannot be probed or decoded."""


def ffmpeg_binary() -> str:
    """Return a usable ffmpeg executable, preferring the bundled wheel."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - only when the wheel is absent
        from shutil import which

        found = which("ffmpeg")
        if not found:
            raise VideoError(
                "No ffmpeg available. Install it with: pip install imageio-ffmpeg"
            )
        return found


@dataclasses.dataclass(frozen=True)
class VideoInfo:
    """What we could learn about a video without decoding all of it."""

    path: Path
    width: int
    height: int
    fps: float
    duration: float

    @property
    def frame_count(self) -> int:
        return int(round(self.duration * self.fps))


_STREAM_RE = re.compile(
    r"Stream #\d+:\d+.*?Video:.*?(?P<w>\d{2,5})x(?P<h>\d{2,5})[^,]*(?:,[^,]*)*?,\s*"
    r"(?P<fps>[\d.]+)\s*fps",
    re.DOTALL,
)
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d\.\d+)")


def probe(path: str | Path) -> VideoInfo:
    """Read dimensions, frame rate and duration from a video file."""
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"No such video file: {path}")

    proc = subprocess.run(
        [ffmpeg_binary(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    # ffmpeg with no output file always exits non-zero; the metadata we want is
    # on stderr regardless, so parse first and only complain if it isn't there.
    err = proc.stderr

    stream = _STREAM_RE.search(err)
    if not stream:
        raise VideoError(f"No decodable video stream found in {path}")

    duration = 0.0
    if (m := _DURATION_RE.search(err)) is not None:
        hours, minutes, seconds = int(m.group(1)), int(m.group(2)), float(m.group(3))
        duration = hours * 3600 + minutes * 60 + seconds

    return VideoInfo(
        path=path,
        width=int(stream.group("w")),
        height=int(stream.group("h")),
        fps=float(stream.group("fps")),
        duration=duration,
    )


def decode_rgb(
    path: str | Path,
    fps: float = MODEL_FPS,
    max_seconds: float | None = None,
) -> np.ndarray:
    """Decode a video to ``(T, H, W, 3)`` uint8 RGB at a constant frame rate.

    Resampling to a constant rate here — rather than by dropping frames later —
    is what makes variable-frame-rate phone recordings behave. ffmpeg duplicates
    or drops frames as needed so that output frame *i* is at time *i/fps*.
    """
    path = Path(path)
    info = probe(path)

    cmd = [ffmpeg_binary(), "-hide_banner", "-loglevel", "error"]
    if max_seconds is not None:
        cmd += ["-t", str(max_seconds)]
    cmd += [
        "-i",
        str(path),
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise VideoError(f"ffmpeg failed to decode {path}: {detail}")

    frame_bytes = info.width * info.height * 3
    if frame_bytes == 0:
        raise VideoError(f"Video {path} reports zero-sized frames")

    total = len(proc.stdout) // frame_bytes
    if total == 0:
        raise VideoError(f"Video {path} decoded to zero frames")

    usable = total * frame_bytes
    frames = np.frombuffer(proc.stdout[:usable], dtype=np.uint8)
    return frames.reshape(total, info.height, info.width, 3)
