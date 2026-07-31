"""Fetch a clip from a video URL, with its captions when they exist.

Captions are the point of this module as much as the video is. Lip reading
output is fluent and confident whether or not it read anything, so the single
most useful thing you can do with a public video is compare the model's guess
against a transcript somebody else already wrote. That turns an unfalsifiable
sentence into a measured error rate.

Downloading from most video sites is against their terms of service. That is the
caller's decision; this module does not pretend otherwise.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path


class SourceError(RuntimeError):
    """Raised when a URL cannot be fetched."""


@dataclasses.dataclass(frozen=True)
class Caption:
    """One caption line, timed relative to the start of the fetched clip."""

    start: float
    end: float
    text: str


@dataclasses.dataclass
class Clip:
    """A downloaded clip and whatever context came with it."""

    path: Path
    title: str
    url: str
    start: float
    duration: float
    captions: list[Caption] = dataclasses.field(default_factory=list)
    caption_source: str | None = None

    @property
    def has_captions(self) -> bool:
        return bool(self.captions)

    def caption_text(self) -> str:
        return " ".join(c.text for c in self.captions).strip()


def _require_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise SourceError(
            "Fetching from a URL needs yt-dlp:\n\n    pip install yt-dlp\n"
        ) from exc
    return yt_dlp


_TAG = re.compile(r"<[^>]+>")


def _parse_vtt(text: str, offset: float = 0.0) -> list[Caption]:
    """Parse WebVTT into captions, shifted so the clip starts at zero.

    Deliberately tolerant: caption files in the wild carry styling blocks,
    positioning cues and inline tags, and none of that should stop us reading
    the words.
    """

    def seconds(stamp: str) -> float:
        stamp = stamp.replace(",", ".")
        parts = stamp.strip().split(":")
        try:
            values = [float(p) for p in parts]
        except ValueError:
            return -1.0
        total = 0.0
        for value in values:
            total = total * 60 + value
        return total

    captions: list[Caption] = []
    seen: set[tuple[float, str]] = set()

    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        timing = next((ln for ln in lines if "-->" in ln), None)
        if timing is None:
            continue
        raw_start, _, raw_end = timing.partition("-->")
        start = seconds(raw_start.split()[0] if raw_start.split() else "")
        end_token = raw_end.split()[0] if raw_end.split() else ""
        end = seconds(end_token)
        if start < 0 or end < 0:
            continue

        body = " ".join(
            _TAG.sub("", ln) for ln in lines[lines.index(timing) + 1 :]
        )
        body = " ".join(body.split())
        if not body:
            continue

        # Rolling auto-captions repeat the previous line as context; drop those.
        key = (round(start, 2), body)
        if key in seen:
            continue
        seen.add(key)

        captions.append(Caption(max(0.0, start - offset), max(0.0, end - offset), body))

    return captions


def _load_captions(directory: Path, offset: float) -> tuple[list[Caption], str | None]:
    """Find and parse whatever subtitle file yt-dlp wrote, if any."""
    for pattern, label in (("*.en*.vtt", "captions"), ("*.vtt", "captions")):
        for path in sorted(directory.glob(pattern)):
            try:
                parsed = _parse_vtt(path.read_text(encoding="utf-8", errors="replace"), offset)
            except OSError:
                continue
            if parsed:
                kind = "auto-generated" if ".orig." not in path.name else "uploaded"
                return parsed, f"{label} ({kind})"
    return [], None


# YouTube serves different player APIs to different clients, and which ones are
# reachable from a given IP changes constantly. Trying several costs a few
# seconds and rescues a lot of otherwise-dead fetches.
_PLAYER_CLIENTS = (None, "tv", "android", "ios", "web_safari")

# Signatures that mean "the site refused us", not "the URL is wrong".
_BLOCKED_SIGNS = (
    "sign in to confirm",
    "not a bot",
    "http error 403",
    "unable to download api page",
    "failed to extract any player response",
    "requested format is not available",
    "this content isn't available",
    "video unavailable",
)


def explain_failure(error_text: str) -> str:
    """Turn a yt-dlp failure into something worth acting on."""
    lowered = error_text.lower()

    if any(sign in lowered for sign in _BLOCKED_SIGNS):
        return (
            "The site refused the download. YouTube blocks requests coming from "
            "datacenters, which is where this app runs — so public videos often "
            "fail here even though they play fine in your browser.\n\n"
            "Download the clip on your own device and upload the file instead. "
            "That path does not touch the site at all and always works."
        )
    if "is not a valid url" in lowered:
        return "That does not look like a video link. Paste the full URL, starting with https://."
    if "private" in lowered or "members-only" in lowered:
        return "That video is private or members-only, so it cannot be fetched."
    if "no video formats" in lowered:
        return "No downloadable video was found at that link."
    return (
        "The download failed. If this is a YouTube link it is most likely their "
        "block on datacenter traffic — download the clip yourself and upload the "
        "file instead."
    )


def fetch_clip(
    url: str,
    workdir: str | Path,
    start: float = 0.0,
    duration: float = 30.0,
    want_captions: bool = True,
) -> Clip:
    """Download a section of a video, plus its captions where available.

    Only the requested section is fetched where the site supports it, which
    keeps a 30-second look at a two-hour video cheap.
    """
    yt_dlp = _require_yt_dlp()

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if duration <= 0:
        raise SourceError("duration must be greater than zero")
    start = max(0.0, start)

    options = {
        "outtmpl": str(workdir / "clip.%(ext)s"),
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "download_ranges": yt_dlp.utils.download_range_func(
            None, [(start, start + duration)]
        ),
        "force_keyframes_at_cuts": True,
    }
    if want_captions:
        options.update(
            {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en.*", "en"],
                "subtitlesformat": "vtt",
            }
        )

    info = None
    last_error: Exception | None = None
    for client in _PLAYER_CLIENTS:
        attempt = dict(options)
        if client:
            attempt["extractor_args"] = {"youtube": {"player_client": [client]}}
        try:
            with yt_dlp.YoutubeDL(attempt) as ydl:
                info = ydl.extract_info(url, download=True)
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 - yt-dlp raises many types
            last_error = exc
            # Clear partial output so the next attempt starts clean.
            for leftover in workdir.glob("clip.*"):
                leftover.unlink(missing_ok=True)

    if last_error is not None:
        raise SourceError(explain_failure(str(last_error))) from last_error

    videos = [
        p for p in sorted(workdir.iterdir())
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
    ]
    if not videos:
        raise SourceError(f"Nothing downloadable was found at {url}")

    captions, caption_source = ([], None)
    if want_captions:
        captions, caption_source = _load_captions(workdir, start)

    return Clip(
        path=videos[0],
        title=(info or {}).get("title") or url,
        url=url,
        start=start,
        duration=duration,
        captions=captions,
        caption_source=caption_source,
    )
